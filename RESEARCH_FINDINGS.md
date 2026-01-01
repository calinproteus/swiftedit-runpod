# SwiftEdit Model Dependencies - Research Findings

## What We Know For Sure

### ✅ Confirmed: SwiftEdit Weights Contain
```
swiftedit_weights/
├── inverse_ckpt-120k/      ← Inverse diffusion model
├── sbv2_0.5/               ← SwiftBrushv2 (one-step UNet)
└── ip_adapter_ckpt-90k/    ← IP-Adapter weights
```

### ❓ Unknown: What AuxiliaryModel() Downloads

Looking at the handler code:
```python
# Line 91: Loads from swiftedit_weights
_inverse_model = InverseModel(inverse_ckpt)

# Line 94: NO PATH - must download something!
_aux_model = AuxiliaryModel()  

# Line 99: Loads from swiftedit_weights
_ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, ...)
```

**The critical question:** What does `AuxiliaryModel()` load when called with no parameters?

---

## Two Possible Scenarios

### Scenario A: Auto-Download at Runtime (Your Research is Correct)
```
First Run:
1. SwiftEdit loads swiftedit_weights ✅
2. AuxiliaryModel() downloads VAE/CLIP/etc from HuggingFace (~3GB)
3. Models cached in ~/.cache/huggingface/

Subsequent Runs:
1. Uses cached models ✅
2. No downloads needed
```

**If this is true:**
- ✅ No pre-download needed
- ✅ Simpler Dockerfile
- ❌ First run takes 2-3 minutes longer

### Scenario B: Pre-Download Needed
```
Build Time:
1. Download SD components to Docker image ✅

Runtime:
1. AuxiliaryModel() loads from cache ✅
2. Instant startup
```

**If this is true:**
- ✅ Faster cold starts
- ❌ More complex build
- ❌ Larger Docker image

---

## How to Find Out

### Method 1: Check SwiftEdit models.py (Need Original Repo)
Look at the `AuxiliaryModel.__init__()` method to see what it loads.

### Method 2: Let Current Build Run
When the build completes (it's currently running), check logs for:
```
[SwiftEdit] Loading models...
Downloading: openai/clip-vit-large-patch14  ← If you see this, auto-download confirmed
```

### Method 3: Test Locally (If You Have Python)
```bash
pip install transformers diffusers torch
python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"
# Check if it downloads (watch network activity)
```

---

## My Recommendation

**CANCEL THE CURRENT BUILD** and try the **simplest approach first:**

### Minimal Dockerfile:
```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    transformers==4.37.2 \
    diffusers==0.22.0 \
    accelerate \
    ftfy \
    einops \
    runpod

# Clone SwiftEdit
RUN git clone https://github.com/Qualcomm-AI-research/SwiftEdit.git /app/SwiftEdit

# Download SwiftEdit weights
WORKDIR /app/SwiftEdit
RUN [download swiftedit_weights as we do now]

# Copy handler
COPY handler.py /app/

CMD ["python", "-u", "/app/handler.py"]
```

**NO `apply_fixes.py`** - Let it use defaults  
**NO `download_models.py`** - Let it auto-download  

**Then test:**
- If it works: Your research was right! ✅
- If it fails with "model not found": We know what to pre-download

---

## What I Got Wrong

1. ❌ Assumed SwiftEdit needs SD 2.x (wrong, it's SD 1.5)
2. ❌ Assumed we need to pre-download models (maybe not needed)
3. ❌ Followed "working pod" config without understanding it
4. ❌ Made the Dockerfile unnecessarily complex

**I should have started simple and only added complexity when needed.**

---

## Next Steps (Your Choice)

**Option A: STOP current build, try minimal Dockerfile**
- Fastest way to find the truth
- Less time wasted if I'm still wrong

**Option B: Let current build finish, check logs**
- See what actually gets downloaded
- Learn from what happens

**Option C: You tell me what to research next**
- I'll search for specific technical details
- You guide the investigation

**Which option?**

