# SwiftEdit RunPod Deployment - Complete Guide

Deploy SwiftEdit to RunPod Serverless in 3 steps. No GitHub needed!

---

## Step 1: Go to RunPod

1. Open: https://www.runpod.io/console/serverless
2. Click your endpoint: `ei14pwe8ohlx13`
3. Click **"Edit"**

---

## Step 2: Configure Endpoint

Set these settings:

- **Source Type:** `Dockerfile` (not GitHub!)
- **Container Disk:** `50 GB` (important!)
- **GPU Type:** `RTX A4000` (or better)
- **Max Workers:** `1`
- **Idle Timeout:** `60` seconds

---

## Step 3: Paste Dockerfile

In the Dockerfile field, paste this **ENTIRE** content:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.37.2 \
    accelerate \
    ftfy \
    diffusers==0.22.0 \
    huggingface-hub==0.25.2 \
    einops \
    safetensors \
    numpy \
    Pillow \
    runpod

# Clone SwiftEdit repository
RUN git clone https://github.com/Qualcomm-AI-research/SwiftEdit.git /app/SwiftEdit

# Apply model fixes inline
RUN cd /app/SwiftEdit && python3 << 'PYTHON_SCRIPT'
import re

models_path = 'models.py'
with open(models_path, 'r') as f:
    content = f.read()

# Fix 1: Use laion OpenCLIP
content = re.sub(
    r"image_encoder = CLIPVisionModelWithProjection\.from_pretrained\('openai/clip-vit-large-patch14'\)",
    "image_encoder = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')",
    content
)

# Fix 2: Use community VAE
content = re.sub(
    r"vae = AutoencoderKL\.from_pretrained\('stabilityai/sd-vae-ft-mse'\)",
    "vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix')",
    content
)

# Fix 3: Manual scheduler creation
content = re.sub(
    r"scheduler = DDIMScheduler\.from_pretrained\('runwayml/stable-diffusion-v1-5', subfolder='scheduler'\)",
    "scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False)",
    content
)

# Fix 4: SD 2 base model
content = re.sub(
    r"'stabilityai/stable-diffusion-2-inpainting'",
    "'stabilityai/stable-diffusion-2-base'",
    content
)

with open(models_path, 'w') as f:
    f.write(content)

print('[SwiftEdit] Applied all model fixes')
PYTHON_SCRIPT

# Download model weights (RunPod has 10Gbps+ connection, downloads in 2-3 minutes)
RUN cd /app/SwiftEdit && \
    echo "[1/6] Downloading weights from GitHub releases..." && \
    wget --retry-connrefused --waitretry=5 --read-timeout=60 --timeout=60 -t 5 \
        https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-aa && \
    wget --retry-connrefused --waitretry=5 --read-timeout=60 --timeout=60 -t 5 \
        https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ab && \
    wget --retry-connrefused --waitretry=5 --read-timeout=60 --timeout=60 -t 5 \
        https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ac && \
    wget --retry-connrefused --waitretry=5 --read-timeout=60 --timeout=60 -t 5 \
        https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ad && \
    wget --retry-connrefused --waitretry=5 --read-timeout=60 --timeout=60 -t 5 \
        https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ae && \
    echo "[2/6] Combining parts..." && \
    cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz && \
    echo "[3/6] Extracting weights (4GB compressed -> 15GB extracted)..." && \
    tar zxf swiftedit_weights.tar.gz && \
    echo "[4/6] Cleaning up temporary files..." && \
    rm -f swiftedit_weights.tar.gz.part-* swiftedit_weights.tar.gz && \
    echo "[5/6] Verifying weights..." && \
    ls -lh swiftedit_weights/ && \
    echo "[6/6] ✓ Model weights ready!"

# Create handler.py inline
RUN cat > /app/handler.py << 'HANDLER_PY'
"""
SwiftEdit RunPod Serverless Handler
Handles image editing requests using SwiftEdit model.
"""

import os
import sys
import time
import base64
import io
from typing import Optional

# Add SwiftEdit to path
sys.path.insert(0, '/app/SwiftEdit')

import runpod
import torch
from PIL import Image

# Global model instances
_models_loaded = False
_inverse_model = None
_aux_model = None
_ip_sb_model = None
_load_time = None

def check_weights_exist():
    """Check if model weights are present (baked into container)."""
    WEIGHTS_ROOT = '/app/SwiftEdit/swiftedit_weights'
    
    if os.path.exists(WEIGHTS_ROOT) and os.path.exists(os.path.join(WEIGHTS_ROOT, 'inverse_ckpt-120k')):
        print("[SwiftEdit] Model weights found (pre-loaded in container)")
        return True
    
    print("[SwiftEdit] ERROR: Model weights not found!")
    return False

def load_models():
    """Load SwiftEdit models into GPU memory."""
    global _models_loaded, _inverse_model, _aux_model, _ip_sb_model, _load_time
    
    if _models_loaded:
        return
    
    if not check_weights_exist():
        raise RuntimeError("Model weights not found in container")
    
    print("[SwiftEdit] Loading models...")
    start = time.perf_counter()
    
    from models import InverseModel, AuxiliaryModel, IPSBV2Model
    
    WEIGHTS_ROOT = '/app/SwiftEdit/swiftedit_weights'
    
    inverse_ckpt = os.path.join(WEIGHTS_ROOT, "inverse_ckpt-120k")
    _inverse_model = InverseModel(inverse_ckpt)
    
    _aux_model = AuxiliaryModel()
    
    path_unet_sb = os.path.join(WEIGHTS_ROOT, "sbv2_0.5")
    ip_ckpt = os.path.join(WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    _ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, _aux_model, with_ip_mask_controller=True)
    
    _load_time = time.perf_counter() - start
    _models_loaded = True
    print(f"[SwiftEdit] Models loaded in {_load_time:.2f}s")

def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    if image_data.startswith('data:'):
        image_data = image_data.split(',', 1)[1]
    
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

def encode_image(image: Image.Image, format: str = 'PNG') -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def to_binary(pix, threshold=0.5):
    """Binary threshold for mask."""
    if float(pix) > threshold:
        return 1.0
    else:
        return 0.0

@torch.no_grad()
def edit_image(
    pil_image: Image.Image,
    src_prompt: str,
    edit_prompt: str,
    scale_edit: float = 0.2,
    scale_non_edit: float = 1.0,
    mask_threshold: float = 0.5,
    scale_ta: float = 1.0
) -> Image.Image:
    """
    Edit image using SwiftEdit.
    
    Args:
        pil_image: Source PIL Image (will be resized to 512x512)
        src_prompt: Description of source image
        edit_prompt: Desired edit
        scale_edit: Edit intensity (0-1, default 0.2)
        scale_non_edit: Background preservation (0-1, default 1.0)
        mask_threshold: Mask sensitivity (0-1, default 0.5)
        scale_ta: Text alignment (0-2, default 1.0)
    
    Returns:
        Edited PIL Image (512x512)
    """
    # Ensure models are loaded
    if not _models_loaded:
        load_models()
    
    # Resize to 512x512
    pil_image = pil_image.resize((512, 512))
    
    start_time = time.perf_counter()
    
    # 1. Inverse model: generate initial noise
    result_dict_inv = _inverse_model.inference(
        pil_image, 
        src_prompt, 
        edit_prompt, 
        scale_edit=scale_edit, 
        scale_non_edit=scale_non_edit
    )
    inverted_z = result_dict_inv["inverted_z"]
    edit_guided_z = result_dict_inv["edit_guided_z"]
    
    # 2. Generate edit mask using auxiliary model
    result_dict_aux = _aux_model.inference(pil_image, edit_prompt)
    mask_source = result_dict_aux["mask_source"]
    
    # Apply threshold to mask
    mask_source = mask_source.point(lambda p: to_binary(p, mask_threshold))
    
    # 3. IP-SB model: final image generation
    result_dict_ip = _ip_sb_model.inference(
        pil_image, 
        edit_prompt,
        inverted_z,
        edit_guided_z,
        mask_source,
        scale_ta=scale_ta
    )
    
    edited_img = result_dict_ip["edited_img"]
    
    inference_time = time.perf_counter() - start_time
    print(f"[SwiftEdit] Inference complete in {inference_time:.3f}s")
    
    return edited_img

def handler(job):
    """RunPod handler function."""
    job_input = job['input']
    
    # Handle warm-up requests
    if job_input.get('warmup'):
        load_models()
        return {
            'status': 'ready',
            'load_time': _load_time
        }
    
    # Validate required fields
    if 'image' not in job_input or 'edit_prompt' not in job_input:
        return {'error': 'Missing required fields: image, edit_prompt'}
    
    try:
        # Load models if not loaded
        if not _models_loaded:
            load_models()
        
        # Decode input image
        source_image = decode_image(job_input['image'])
        
        # Extract parameters
        edit_prompt = job_input['edit_prompt']
        source_prompt = job_input.get('source_prompt', '')
        scale_edit = float(job_input.get('scale_edit', 0.2))
        scale_non_edit = float(job_input.get('scale_non_edit', 1.0))
        mask_threshold = float(job_input.get('mask_threshold', 0.5))
        scale_ta = float(job_input.get('scale_ta', 1.0))
        
        # Run inference
        start_time = time.perf_counter()
        edited_image = edit_image(
            source_image,
            source_prompt,
            edit_prompt,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold,
            scale_ta=scale_ta
        )
        total_time = time.perf_counter() - start_time
        
        # Encode result
        result_base64 = encode_image(edited_image, 'PNG')
        
        return {
            'image': result_base64,
            'inference_time': round(total_time, 3),
            'dimensions': '512x512',
            'was_cold_start': _load_time is not None
        }
        
    except Exception as e:
        print(f"[SwiftEdit] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == '__main__':
    print("[SwiftEdit] Starting RunPod serverless handler...")
    runpod.serverless.start({'handler': handler})
HANDLER_PY

# Set environment
ENV PYTHONPATH="/app/SwiftEdit:${PYTHONPATH}"

# RunPod handler entrypoint
CMD ["python", "-u", "/app/handler.py"]
```

---

## Step 4: Deploy

1. Click **"Save"**
2. Click **"Deploy"**
3. Wait **15-20 minutes** for build

### Monitor Build:

Go to **"Logs"** tab and watch for:
```
[1/6] Downloading weights from GitHub releases...
[2/6] Combining parts...
[3/6] Extracting weights...
...
[6/6] ✓ Model weights ready!
```

---

## Step 5: Test

Once build completes, test with:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_RUNPOD_API_KEY' \
  -d '{"input":{"warmup":true}}'
```

**Expected response:**
```json
{
  "output": {
    "status": "ready",
    "load_time": 2.1
  }
}
```

---

## Step 6: Restart Proxy Server

Your proxy server `.env` should have:
```
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_SWIFTEDIT_ENDPOINT=https://api.runpod.ai/v2/your_endpoint_id
```

Just restart it:
```powershell
cd "C:\Users\calin\Desktop\PROTEUS\AI Agent Plugin\proxy-server"
npm run dev
```

---

## Step 7: Test in Figma

1. Reload Figma plugin
2. Go to **Images** → **Live Canvas**
3. Create canvas → **Realtime** tab
4. Generate! ⚡

**Expected:**
- Cold start: ~2-3s
- Warm inference: ~0.4-0.6s

---

## Troubleshooting

### Build fails: "wget: unable to resolve host"
**Solution:** Retry build (GitHub releases temporarily unavailable)

### Build fails: "No space left on device"
**Solution:** Container Disk must be 50GB minimum

### First request times out
**Solution:** Proxy timeout is 120s. First request loads models (~2s). Should work.

---

## Done! 🎉

Your SwiftEdit endpoint is now live!

