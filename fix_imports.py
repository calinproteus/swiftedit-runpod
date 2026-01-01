"""
Fix SwiftEdit import paths and model references.
The repo has incorrect import paths for ip_adapter modules
and uses non-existent model identifiers.
"""

models_path = '/app/SwiftEdit/models.py'

print("=" * 60)
print("Patching models.py for correct imports and model paths...")
print("=" * 60)

with open(models_path, 'r') as f:
    content = f.read()

# Fix 1: Import paths (src.ip_adapter -> src)
print("\n[1/2] Fixing import paths...")
content = content.replace('from src.ip_adapter.attention_processor', 'from src.attention_processor')
content = content.replace('from src.ip_adapter.mask_attention_processor', 'from src.mask_attention_processor')
print("✓ Import paths fixed")

# Fix 2: Model identifiers (use community mirror for SD 2.1-base)
print("\n[2/2] Fixing model identifiers...")
content = content.replace(
    'model_name="stabilityai/stable-diffusion-2-1-base"',
    'model_name="Manojb/stable-diffusion-2-1-base"'
)
print("✓ Model identifiers fixed (using Manojb/stable-diffusion-2-1-base)")

with open(models_path, 'w') as f:
    f.write(content)

print("\n" + "=" * 60)
print("✓ SUCCESS: models.py fully patched!")
print("=" * 60)

