"""
Apply all model fixes from working pod setup
Exact replica of the inline fixes from RUNPOD_DEPLOY.md
"""
import re

models_path = '/app/SwiftEdit/models.py'
with open(models_path, 'r') as f:
    content = f.read()

# Fix 1: Use laion OpenCLIP
print("[Fix 1] Applying laion OpenCLIP...")
content = re.sub(
    r"image_encoder = CLIPVisionModelWithProjection\.from_pretrained\('openai/clip-vit-large-patch14'\)",
    "image_encoder = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')",
    content
)

# Fix 2: Use community VAE
print("[Fix 2] Applying community VAE...")
content = re.sub(
    r"vae = AutoencoderKL\.from_pretrained\('stabilityai/sd-vae-ft-mse'\)",
    "vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix')",
    content
)

# Fix 3: Manual scheduler creation
print("[Fix 3] Applying manual scheduler...")
content = re.sub(
    r"scheduler = DDIMScheduler\.from_pretrained\('runwayml/stable-diffusion-v1-5', subfolder='scheduler'\)",
    "scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False)",
    content
)

# Fix 4: SD 2 base model
print("[Fix 4] Applying SD 2 base model...")
content = re.sub(
    r"'stabilityai/stable-diffusion-2-inpainting'",
    "'stabilityai/stable-diffusion-2-base'",
    content
)

with open(models_path, 'w') as f:
    f.write(content)

print('[SwiftEdit] ✅ All 4 fixes applied successfully!')
