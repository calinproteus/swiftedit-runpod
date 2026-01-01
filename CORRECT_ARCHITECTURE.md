# SwiftEdit Correct Architecture

## What We Discovered

### ❌ What I Was Doing Wrong:
- Forcing SwiftEdit to use **Stable Diffusion 2.x**
- Following a "working pod" config that was incorrect
- Not reading the actual SwiftEdit documentation

### ✅ What SwiftEdit Actually Uses:
- **Stable Diffusion 1.5** (base architecture)
- **SwiftBrushv2** (sbv2_0.5 - one-step diffusion UNet)
- **SD 1.5 lineage** (via SwiftBrushv2 → SD-Turbo → SD 1.5)

---

## SwiftEdit Architecture

```
┌─────────────────────────────────────────┐
│ SwiftEdit Custom Weights (~15GB)        │
│ Already downloaded from GitHub!         │
├─────────────────────────────────────────┤
│ sbv2_0.5/                               │ ← SwiftBrushv2 UNet (SD 1.5 based)
│ inverse_ckpt-120k/                      │ ← Inverse diffusion model
│ ip_adapter_ckpt-90k/                    │ ← Image prompt adapter
└─────────────────────────────────────────┘
              ↓ (needs these from HuggingFace)
┌─────────────────────────────────────────┐
│ SD 1.5 Base Components (~3GB)           │
│ Need to download                        │
├─────────────────────────────────────────┤
│ VAE (image encoder/decoder)             │
│ Text Encoder (CLIP)                     │
│ Tokenizer                               │
│ Scheduler (can be hardcoded)            │
└─────────────────────────────────────────┘
```

---

## What We Need to Download

### We HAVE (from GitHub):
- ✅ `swiftedit_weights/sbv2_0.5/` - SwiftBrushv2 UNet
- ✅ `swiftedit_weights/inverse_ckpt-120k/`
- ✅ `swiftedit_weights/ip_adapter_ckpt-90k/`

### We NEED (from HuggingFace/Community):

#### Option A: CompVis (Community Maintained)
```python
# VAE
AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae')

# Text Encoder + Tokenizer
CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='text_encoder')
AutoTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='tokenizer')

# CLIP Vision
CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
```

#### Option B: Improved Components
```python
# Better VAE
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')

# Text Encoder + Tokenizer (same as above)
CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
```

---

## Model Compatibility

### ✅ Compatible:
- **SD 1.5** (what SwiftEdit was trained on)
- **SD 1.5 fine-tunes** (Realistic Vision, DreamShaper, etc.)
- **SD-Turbo** (SD 1.5 distilled)
- Any model with **SD 1.5 architecture**

### ❌ NOT Compatible:
- **SDXL** (different architecture: 2048-dim embeddings, dual text encoders)
- **SD 2.x** (768-dim but different UNet structure)
- **SD 3.x** (completely different MMDiT architecture)
- **Flux** (completely different)

**Why:** SwiftBrushv2 (sbv2_0.5) was specifically trained on SD 1.5 UNet architecture. The weights expect exact layer dimensions/counts.

---

## What Needs to Change

### 1. Replace `apply_fixes.py`
**Old (WRONG):**
- Forced SD 2.x models
- Replaced SD 1.5 scheduler

**New (CORRECT):**
- Keep SD 1.5 references
- Only fix import paths
- Optionally use manual scheduler (to avoid download)

### 2. Replace `download_models.py`
**Old (WRONG):**
```python
# Downloaded SD 2.x components
CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2', ...)
```

**New (CORRECT):**
```python
# Download SD 1.5 components
AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae')
# Or use improved VAE: 'stabilityai/sd-vae-ft-mse'
```

---

## References

- **SwiftEdit Paper:** [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_SwiftEdit_Lightning_Fast_Text-Guided_Image_Editing_via_One-Step_Diffusion_CVPR_2025_paper.pdf)
- **SwiftEdit GitHub:** https://github.com/Qualcomm-AI-research/SwiftEdit
- **SwiftBrushv2 Project:** https://swiftbrushv2.github.io/
- **SwiftBrushv2 Paper:** Shows it's based on SD-Turbo (SD 1.5 lineage)

---

## Why This Matters

Using the **wrong base model (SD 2.x)**:
- ❌ Incompatible architecture
- ❌ Dimension mismatches
- ❌ Poor/broken results
- ❌ Wasted time debugging

Using the **correct base model (SD 1.5)**:
- ✅ Matches SwiftEdit training
- ✅ Compatible dimensions
- ✅ Best quality results
- ✅ Fast inference (~0.4s as advertised)

