# Alternative Model Sources (If HuggingFace Unavailable)

## Models SwiftEdit Needs

### 1. CLIP Vision Encoder
**Original:** `openai/clip-vit-large-patch14`

**Alternatives:**
- Direct from OpenAI: https://openaipublic.azureedge.net/clip/models/
- Civitai mirrors
- Local copy: Can be downloaded once and bundled

### 2. VAE
**Original:** `stabilityai/sd-vae-ft-mse`

**Alternatives:**
- Civitai: Search "SD VAE"
- Direct links from community repos
- Can use default VAE if needed

### 3. Text Encoder
**Original:** `openai/clip-vit-large-patch14`

**Alternatives:**
- Same as CLIP vision (same model, different component)

---

## Solution 1: Bundle Models in Repo

**Download locally, commit to repo:**
```bash
# On your machine:
cd swiftedit-runpod
mkdir -p models_cache

# Download models
python3 << EOF
from transformers import CLIPVisionModelWithProjection
model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
model.save_pretrained('./models_cache/clip-vit-large')
EOF

# Commit to repo
git add models_cache/
git commit -m "Bundle models"
```

**Then in Dockerfile:**
```dockerfile
COPY models_cache/ /root/.cache/huggingface/
```

---

## Solution 2: Download from Direct URLs

**Find direct download links:**
```dockerfile
RUN wget https://direct-url-to-model.tar.gz && \
    tar -xzf model.tar.gz -C /root/.cache/huggingface/
```

---

## Solution 3: Skip Pre-caching (Your Original Research!)

**Just let SwiftEdit use its defaults:**
- If models aren't available, SwiftEdit might have fallbacks
- Or it fails with clear error telling us what it needs
- Then we know exactly what to bundle

---

## What Should We Do?

**Tell me what error the current build shows** and I'll provide the exact solution!

