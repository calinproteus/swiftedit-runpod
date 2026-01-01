"""
Pre-download and cache ALL models SwiftEdit needs.
Based on ACTUAL models.py code from SwiftEdit repo.

InverseModel uses: stabilityai/sd-turbo
AuxiliaryModel uses: Manojb/stable-diffusion-2-1-base (community mirror) + h94/IP-Adapter
"""

print("=" * 60)
print("Pre-caching models for SwiftEdit (CORRECT)")
print("=" * 60)

try:
    print("\n[1/6] Importing libraries...")
    from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    import torch
    print("✓ Imports successful")

    print("\n[2/6] Pre-downloading SD-Turbo components (for InverseModel)...")
    DDPMScheduler.from_pretrained('stabilityai/sd-turbo', subfolder='scheduler')
    AutoencoderKL.from_pretrained('stabilityai/sd-turbo', subfolder='vae')
    AutoTokenizer.from_pretrained('stabilityai/sd-turbo', subfolder='tokenizer')
    CLIPTextModel.from_pretrained('stabilityai/sd-turbo', subfolder='text_encoder')
    print("✓ SD-Turbo components cached")

    print("\n[3/6] Pre-downloading SD 2.1-base scheduler (for AuxiliaryModel)...")
    DDPMScheduler.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder='scheduler')
    print("✓ SD 2.1-base scheduler cached")

    print("\n[4/6] Pre-downloading SD 2.1-base VAE (for AuxiliaryModel)...")
    AutoencoderKL.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder='vae')
    print("✓ SD 2.1-base VAE cached")

    print("\n[5/6] Pre-downloading SD 2.1-base text components (for AuxiliaryModel)...")
    AutoTokenizer.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder='tokenizer')
    CLIPTextModel.from_pretrained('Manojb/stable-diffusion-2-1-base', subfolder='text_encoder')
    print("✓ SD 2.1-base text components cached")

    print("\n[6/6] Pre-downloading IP-Adapter image encoder (for AuxiliaryModel)...")
    CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter', subfolder='models/image_encoder')
    print("✓ IP-Adapter image encoder cached")

    print("\n" + "=" * 60)
    print("✓ SUCCESS: All SwiftEdit models pre-cached!")
    print("=" * 60)
    print("\nModels cached:")
    print("- stabilityai/sd-turbo (InverseModel)")
    print("- Manojb/stable-diffusion-2-1-base (AuxiliaryModel - community mirror)")
    print("- h94/IP-Adapter/models/image_encoder (AuxiliaryModel)")

except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ ERROR: {str(e)}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    exit(1)

