import torch
from diffusers.utils import load_video, export_to_video
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanVideoToVideoPipeline, UniPCMultistepScheduler

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
)
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipe = WanVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True
)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=flow_shift
)
# change to pipe.to("cuda") if you have sufficient VRAM
pipe.enable_model_cpu_offload()

prompt = "A robot standing on a mountain top. The sun is setting in the background"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
)
output = pipe(
    video=video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=512,
    guidance_scale=7.0,
    strength=0.7,
).frames[0]

export_to_video(output, "output/example/v2v/robot.mp4", fps=16)
