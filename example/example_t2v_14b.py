import torch

from diffusers import WanPipeline
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
pipe = WanPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16, local_files_only=True)
pipe.enable_model_cpu_offload()

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 80

frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
export_to_video(frames, "wan-t2v-14b-80f.mp4", fps=16)
