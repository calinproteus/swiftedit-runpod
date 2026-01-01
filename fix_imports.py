"""
Fix SwiftEdit import paths.
The repo has incorrect import paths for ip_adapter modules.
"""

models_path = '/app/SwiftEdit/models.py'

print("Fixing import paths in models.py...")

with open(models_path, 'r') as f:
    content = f.read()

# Fix import paths
content = content.replace('from src.ip_adapter.attention_processor', 'from src.attention_processor')
content = content.replace('from src.ip_adapter.mask_attention_processor', 'from src.mask_attention_processor')

with open(models_path, 'w') as f:
    f.write(content)

print("✓ Import paths fixed!")

