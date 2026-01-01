"""
Pre-download all HuggingFace models needed by SwiftEdit.
This runs during Docker build to create a fully self-contained image.
"""

print("=" * 60)
print("Pre-downloading HuggingFace models for SwiftEdit")
print("=" * 60)

try:
    print("\n[1/4] Downloading CLIP (text/image encoder)...")
    from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoTokenizer
    CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    CLIPTextModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    print("✓ CLIP downloaded")

    print("\n[2/4] Downloading VAE...")
    from diffusers import AutoencoderKL
    AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix')
    print("✓ VAE downloaded")

    print("\n[3/4] Downloading SD 2 base model...")
    from diffusers import UNet2DConditionModel
    CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='text_encoder')
    AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='tokenizer')
    UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='unet')
    AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='vae')
    print("✓ SD 2 base downloaded")

    print("\n[4/4] Manual scheduler (no download needed)")
    print("✓ Scheduler will be created at runtime")

    print("\n" + "=" * 60)
    print("✓ SUCCESS: All HuggingFace models pre-downloaded!")
    print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ ERROR: {str(e)}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    exit(1)

