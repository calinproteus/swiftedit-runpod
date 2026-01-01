"""
Pre-download all HuggingFace models needed by SwiftEdit.
This runs during Docker build to create a fully self-contained image.
"""
import sys

print("=" * 60)
print("Pre-downloading HuggingFace models for SwiftEdit")
print("=" * 60)

# Test imports first
try:
    print("\n[0/4] Testing imports...")
    from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel
    print("✓ All imports successful")
except ImportError as e:
    print(f"\n✗ IMPORT ERROR: {str(e)}")
    print("Make sure transformers and diffusers are installed")
    sys.exit(1)

# Download each model with individual error handling
try:
    print("\n[1/4] Downloading CLIP (text/image encoder)...")
    print("  - CLIPVisionModelWithProjection...")
    CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    print("  - CLIPTextModel...")
    CLIPTextModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    print("  - AutoTokenizer...")
    AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    print("✓ CLIP downloaded successfully")
except Exception as e:
    print(f"\n✗ CLIP DOWNLOAD FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/4] Downloading VAE...")
    AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix')
    print("✓ VAE downloaded successfully")
except Exception as e:
    print(f"\n✗ VAE DOWNLOAD FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[3/4] Downloading SD 2 model...")
    print("  - text_encoder...")
    CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2', subfolder='text_encoder')
    print("  - tokenizer...")
    AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2', subfolder='tokenizer')
    print("  - unet...")
    UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2', subfolder='unet')
    print("  - vae...")
    AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2', subfolder='vae')
    print("✓ SD 2 downloaded successfully")
except Exception as e:
    print(f"\n✗ SD 2 BASE DOWNLOAD FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] Manual scheduler (no download needed)")
print("✓ Scheduler will be created at runtime")

print("\n" + "=" * 60)
print("✓ SUCCESS: All HuggingFace models pre-downloaded!")
print("=" * 60)

