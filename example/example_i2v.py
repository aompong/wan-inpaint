import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanImageToVideoPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel, CLIPVisionModel

# model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32, local_files_only=True
)
text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)

transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")

max_area = 720 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "wan-i2v.mp4", fps=16)
