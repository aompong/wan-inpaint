import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import load_video, export_to_video
import os
import time
from diffusers.video_processor import VideoProcessor
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import UMT5EncoderModel
from transformers import AutoTokenizer
from accelerate import cpu_offload
from tqdm import tqdm
from utils import seed_everything

seed_everything(0)

# export HF_HUB_CACHE=/ist-nas/ist-share/vision/huggingface_hub/
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

vae = None
scheduler = None
tokenizer = None
text_encoder = None
transformer = None

vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample) # 4
vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample) # 8
scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", model_max_length=512, local_files_only=True)
text_encoder = UMT5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE, local_files_only=True)
transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=DTYPE, local_files_only=True)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=DTYPE)

models = [text_encoder, transformer, vae]
for m in models:
    cpu_offload(m, DEVICE) # automatically move unused models to cpu

video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
mask_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial, do_binarize=True, do_normalize=False, do_convert_grayscale=True)


@torch.no_grad()
def sdedit_video_inpainting_pipeline(
    input_video_frames,   # List[PIL.Image.Image]
    input_mask_frames,    # b f c h w
    height,
    width,
    prompt,
    neg_prompt,
    strength,
    num_inference_steps,
    guidance_scale,
    dir='output/',
):
    # 1. timesteps
    scheduler.set_timesteps(num_inference_steps, DEVICE)
    original_timesteps = scheduler.timesteps

    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    start_step = max(num_inference_steps - init_timestep, 0) 
    timesteps = original_timesteps[start_step * scheduler.order :] # what's scheduler's order?
    num_inference_steps = num_inference_steps - start_step
    latent_timestep = timesteps[:1] # shape: [1]

    # 2. video latents
    video_tensor = video_processor.preprocess_video(input_video_frames, height, width).to(DEVICE, dtype=torch.float32) # [B, C, F, H, W], range [0, 1]

    num_channels_latents = transformer.config.in_channels # 16
    shape = (
        1,                                                              # b
        num_channels_latents,                                           # latent c
        (video_tensor.size(2) - 1) // vae_scale_factor_temporal + 1,    # latent f
        height // vae_scale_factor_spatial,                             # latent h
        width // vae_scale_factor_spatial,                              # latent w
    )
    initial_latents = [vae.encode(vid.unsqueeze(0)).latent_dist.sample() for vid in video_tensor]
    initial_latents = torch.cat(initial_latents, dim=0).to(DTYPE)

    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(DEVICE, DTYPE))
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(DEVICE, DTYPE)
    init_latents = (initial_latents - latents_mean) * latents_std # why? similar to sd vae's scaling factor?

    noise = torch.randn(shape).to(DEVICE)
    latents = scheduler.add_noise(init_latents, noise, latent_timestep)

    # 3. mask latents
    mask_tensor = mask_processor.preprocess_video(input_mask_frames, height, width).to(DEVICE) # [B, 1, F, H, W], (0, 1)
    
    mask = torch.nn.functional.interpolate(
        mask_tensor, 
        size=shape[2:],
        mode='trilinear'
    ).to(DEVICE, DTYPE)

    # 4. text embeddings
    pos_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    pos_input_ids = pos_inputs.input_ids
    pos_attention_mask = pos_inputs.attention_mask # [1, 512]
    pos_seq_lens = pos_attention_mask.gt(0).sum(dim=1).long()
    pos_text_embeds = text_encoder(pos_input_ids.to(DEVICE), pos_attention_mask.to(DEVICE)).last_hidden_state 
    pos_text_embeds = pos_text_embeds.to(DEVICE, DTYPE) # [1, 512, 4096]
    pos_trimmed_embedds = [u[:v] for u, v in zip(pos_text_embeds, pos_seq_lens)] # [1, seq_len, 4096]
    prompt_embeds = torch.stack([
        torch.cat([u, u.new_zeros(tokenizer.model_max_length - u.size(0), u.size(1))]) for u in pos_trimmed_embedds
    ], dim=0) # [1, 512, 4096]

    neg_inputs = tokenizer(neg_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    neg_input_ids = neg_inputs.input_ids
    neg_attention_mask = neg_inputs.attention_mask
    neg_seq_lens = neg_attention_mask.gt(0).sum(dim=1).long()
    neg_text_embeds = text_encoder(neg_input_ids.to(DEVICE), neg_attention_mask.to(DEVICE)).last_hidden_state
    neg_text_embeds = neg_text_embeds.to(DEVICE, DTYPE)
    neg_trimmed_embeds = [u[:v] for u, v in zip(neg_text_embeds, neg_seq_lens)]
    negative_prompt_embeds = torch.stack([torch.cat([
        u, u.new_zeros(tokenizer.model_max_length - u.size(0), u.size(1))]) 
        for u in neg_trimmed_embeds
    ], dim=0)

    prompt_embeds.to(DTYPE)
    negative_prompt_embeds.to(DTYPE)

    # 5. denoising (reverse sde)
    progress_bar = tqdm(timesteps, total=num_inference_steps)
    for i, t in enumerate(progress_bar):
        latent_model_input = latents.to(DTYPE)

        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(1), # convert scalar to [t]
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        noise_pred_uncond = transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(1),
            encoder_hidden_states=negative_prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

        # x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # sdedit
        init_latents_proper = init_latents
        init_mask = mask
        if i < len(timesteps) - 1:
            noise_timestep = timesteps[i+1]
            init_latents_proper = scheduler.add_noise(init_latents_proper, noise, torch.tensor([noise_timestep]))
        latents = (1-init_mask) * init_latents_proper + init_mask * latents

    # 6. Decode
    latents = latents.to(vae.dtype)
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype))
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents/latents_std + latents_mean

    output = vae.decode(latents, return_dict=False)[0] # [1, 3, F, H, W]
    output = video_processor.postprocess_video(output) # (F, 3, H, W)
    return output

def main():
    dir = f'output/sdedit/{time.strftime("%m%d")}/{time.strftime("%H%M")}'
    os.makedirs(dir, exist_ok=True)

    path = "input/landmark/process/16fps_720x1280_crop_41/man_video.mp4"
    input_video_frames = load_video(path) 
    input_mask_frames = load_video(path.replace('video', 'mask'))

    height=1280
    width=720
    strength = 0.7
    num_inference_steps = 10
    guidance_scale = 7.0
    prompt = "A man is speaking straight to the camera. He is bald, has beard, and is wearing a white shirt. His mouth opens and closes, naturally revealing his teeth as he gives his speech. He is eloquently pronouncing each word, moving his head and changing his facial expression as he talks."
    neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    inpainted_video = sdedit_video_inpainting_pipeline(
        input_video_frames,
        input_mask_frames,
        height=height,
        width=width,
        prompt=prompt,
        neg_prompt=neg_prompt, 
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        dir=dir,
    )[0]
    export_to_video(inpainted_video, f'{dir}/out_{height}x{width}_str{strength}_infstp{num_inference_steps}_gs{guidance_scale}.mp4', fps=16)

if __name__ == "__main__":
    main()
