# Alternative: Use Local Model Cache

Instead of downloading from HuggingFace during Docker build, you can:

## Option 1: Copy Local HuggingFace Cache

If you have models cached locally at `~/.cache/huggingface/`:

```dockerfile
# In Dockerfile, add before handler:
COPY huggingface_cache/ /root/.cache/huggingface/
```

**Pros:**
- No internet needed during build
- Faster builds (no downloads)
- Works offline

**Cons:**
- Need to download locally first
- Larger Git repo (if you commit cache)

---

## Option 2: Download from Custom URL

If you host models yourself:

```dockerfile
RUN wget https://your-server.com/stable-diffusion-2.tar.gz && \
    tar -xzf stable-diffusion-2.tar.gz -C /root/.cache/huggingface/
```

**Pros:**
- Control over hosting
- Can use faster CDN
- No HuggingFace dependency

**Cons:**
- Need to host files (~5GB)
- Maintain file structure

---

## Option 3: Use Different Model Variants

SwiftEdit works with **any SD2-based model**:

```python
# Instead of base SD2:
UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2')

# You could use:
UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-1')  # Slightly better
# OR
UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-depth')  # Depth-aware
# OR any SD2 fine-tune from HuggingFace or Civitai
```

**Constraints:**
- Must be SD2 architecture (not SD 1.5, SDXL, or Flux)
- Must have same components (unet, vae, text_encoder, tokenizer)

---

## Option 4: Use Model from Civitai

Many SD2 models on Civitai work:

```dockerfile
RUN wget "https://civitai.com/api/download/models/[MODEL_ID]" -O model.safetensors && \
    # Convert and place in cache
```

**Pros:**
- More model variety
- Community fine-tunes
- Sometimes better quality

**Cons:**
- Need conversion to diffusers format
- Less standardized structure

---

## What We're Using Now

```dockerfile
# Current approach: Download from HuggingFace during build
RUN python3 download_models.py
```

This downloads:
- `stabilityai/stable-diffusion-2` (base model)
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (CLIP encoder)
- `madebyollin/sdxl-vae-fp16-fix` (better VAE)

**Why this approach:**
- Simple and reliable
- No manual hosting
- Works with standard tools
- Models are cached in Docker image

