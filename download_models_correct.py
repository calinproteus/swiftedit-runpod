"""
CORRECT SwiftEdit Model Downloads
Based on actual SwiftEdit architecture (SD 1.5 + SwiftBrushv2)

We ALREADY have in swiftedit_weights/:
- sbv2_0.5/ (SwiftBrushv2 UNet) ✅
- inverse_ckpt-120k/ ✅
- ip_adapter_ckpt-90k/ ✅

We NEED to download:
- SD 1.5 VAE
- SD 1.5 text encoder
- SD 1.5 tokenizer
- CLIP (for image encoding)
"""
import sys

print("=" * 60)
print("Pre-downloading SD 1.5 components for SwiftEdit")
print("=" * 60)

# Test imports
try:
    print("\n[0/3] Testing imports...")
    from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoTokenizer
    from diffusers import AutoencoderKL
    print("✓ All imports successful")
except ImportError as e:
    print(f"\n✗ IMPORT ERROR: {str(e)}")
    sys.exit(1)

# Download SD 1.5 components
try:
    print("\n[1/3] Downloading SD 1.5 VAE...")
    # Try CompVis first (community maintained)
    try:
        AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae')
        print("✓ SD 1.5 VAE downloaded (CompVis)")
    except:
        # Fallback to stabilityai/sd-vae-ft-mse (improved VAE)
        AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
        print("✓ SD 1.5 VAE downloaded (stabilityai improved)")
except Exception as e:
    print(f"\n✗ VAE DOWNLOAD FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/3] Downloading SD 1.5 text encoder + tokenizer...")
    # Try CompVis first
    try:
        CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='text_encoder')
        AutoTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='tokenizer')
        print("✓ SD 1.5 text components downloaded (CompVis)")
    except:
        # Fallback to openai CLIP
        CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        print("✓ SD 1.5 text components downloaded (OpenAI CLIP)")
except Exception as e:
    print(f"\n✗ TEXT ENCODER DOWNLOAD FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[3/3] Downloading CLIP vision encoder...")
    # For IP-Adapter image encoding
    CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
    print("✓ CLIP vision encoder downloaded")
except Exception as e:
    print(f"\n✗ CLIP DOWNLOAD FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ SUCCESS: All SD 1.5 components downloaded!")
print("=" * 60)
print("\nSwiftEdit will use:")
print("- SwiftBrushv2 UNet (from swiftedit_weights/sbv2_0.5/)")
print("- SD 1.5 VAE (just downloaded)")
print("- SD 1.5 text encoder (just downloaded)")
print("- CLIP vision (just downloaded)")

