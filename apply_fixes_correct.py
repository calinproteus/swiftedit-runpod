"""
CORRECT Apply Fixes for SwiftEdit
Based on actual SwiftEdit architecture (SD 1.5, not SD 2!)

SwiftEdit was designed for SD 1.5 + SwiftBrushv2.
We should NOT change the base model references!
"""
import re

models_path = '/app/SwiftEdit/models.py'
with open(models_path, 'r') as f:
    content = f.read()

# Fix 1: Fix import paths (ip_adapter directory structure)
print("[Fix 1] Fixing import paths...")
content = content.replace('from src.ip_adapter.attention_processor', 'from src.attention_processor')
content = content.replace('from src.ip_adapter.mask_attention_processor', 'from src.mask_attention_processor')

# Fix 2: Use better CLIP (optional, only if SwiftEdit defaults to openai)
print("[Fix 2] Checking CLIP encoder...")
# Only replace if using the basic openai CLIP
if "'openai/clip-vit-large-patch14'" in content:
    print("  - Found openai CLIP, keeping it (SD 1.5 standard)")
else:
    print("  - CLIP already configured")

# Fix 3: Use better VAE (optional)
print("[Fix 3] Checking VAE...")
if "'stabilityai/sd-vae-ft-mse'" in content:
    print("  - Found stabilityai VAE, keeping it (SD 1.5 improved)")
else:
    print("  - VAE already configured")

# Fix 4: Manual scheduler (if needed)
print("[Fix 4] Checking scheduler...")
if "DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5'" in content:
    # Replace with manual scheduler to avoid download
    content = content.replace(
        "DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')",
        "DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False)"
    )
    print("  - Replaced scheduler with manual config")
else:
    print("  - Scheduler already configured")

# Write changes
with open(models_path, 'w') as f:
    f.write(content)

print('\n[SwiftEdit] ✅ All fixes applied!')
print('\nNOTE: SwiftEdit uses SD 1.5 architecture by design.')
print('We did NOT change the base model (SD 1.5 is correct!)')

