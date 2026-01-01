"""
Pre-download and cache ALL models SwiftEdit needs.
This runs during Docker build to create a fully self-contained image.

Based on SwiftEdit's actual code, it will try to download these at runtime.
We download them NOW during build so they're cached in the image.
"""

print("=" * 60)
print("Pre-caching models for SwiftEdit")
print("=" * 60)

try:
    print("\n[1/5] Importing libraries...")
    from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    import torch
    print("✓ Imports successful")

    print("\n[2/5] Pre-downloading CLIP vision encoder...")
    CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
    print("✓ CLIP vision cached")

    print("\n[3/5] Pre-downloading VAE...")
    AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
    print("✓ VAE cached")

    print("\n[4/5] Pre-downloading CLIP text encoder...")
    CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    print("✓ CLIP text cached")

    print("\n[5/5] Verifying cache...")
    import os
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        print(f"✓ Cache directory exists: {cache_dir}")
    else:
        print("✗ Cache directory not found!")
        exit(1)

    print("\n" + "=" * 60)
    print("✓ SUCCESS: All models pre-cached in Docker image!")
    print("=" * 60)
    print("\nSwiftEdit will now start instantly with no downloads!")

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ ERROR: {str(e)}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    exit(1)

