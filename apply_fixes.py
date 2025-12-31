"""
Apply all model fixes from working pod setup
"""

# Fix 1: Update to use laion OpenCLIP
print("Applying Fix 1: OpenCLIP...")
with open('/app/SwiftEdit/models.py', 'r') as f:
    content = f.read()

content = content.replace(
    'self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")',
    'self.text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")'
)

content = content.replace(
    'self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")',
    'self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")'
)

with open('/app/SwiftEdit/models.py', 'w') as f:
    f.write(content)

# Fix 2: Use SD 2 base model (from working pod configuration)
print("Applying Fix 2: Community components + SD 2 base...")
with open('/app/SwiftEdit/models.py', 'r') as f:
    content = f.read()

# Fix the model_name: Change to stable-diffusion-2-base (working config)
# Original SwiftEdit uses: stabilityai/stable-diffusion-2-inpainting
# Working pod uses: stabilityai/stable-diffusion-2-base
content = content.replace(
    'stabilityai/stable-diffusion-2-inpainting',
    'stabilityai/stable-diffusion-2-base'
)

# Also fix any references to 2-1-base (which doesn't exist)
content = content.replace(
    'stabilityai/stable-diffusion-2-1-base',
    'stabilityai/stable-diffusion-2-base'
)

# Replace AuxiliaryModel init to use community components
old_vae = 'self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)'
new_vae = 'self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)'

if old_vae in content:
    content = content.replace(old_vae, new_vae)

# Update tokenizer and text encoder in AuxiliaryModel
old_tok = 'self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")'
new_tok = 'self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")'

old_text = 'self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")'
new_text = 'self.text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")'

if old_tok in content:
    content = content.replace(old_tok, new_tok)
if old_text in content:
    content = content.replace(old_text, new_text)

with open('/app/SwiftEdit/models.py', 'w') as f:
    f.write(content)

# Fix 3: Fix import paths
print("Applying Fix 3: Import paths...")
with open('/app/SwiftEdit/models.py', 'r') as f:
    content = f.read()

content = content.replace('from src.ip_adapter.attention_processor', 'from src.attention_processor')
content = content.replace('from src.ip_adapter.mask_attention_processor', 'from src.mask_attention_processor')

with open('/app/SwiftEdit/models.py', 'w') as f:
    f.write(content)

print("All fixes applied successfully!")

