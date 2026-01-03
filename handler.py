"""
SwiftEdit RunPod Serverless Handler
Handles image editing requests using SwiftEdit model.
Uses CORRECT 3-step workflow: inverse → auxiliary → IP-SB
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
    
    # Load inverse model
    inverse_ckpt = os.path.join(WEIGHTS_ROOT, "inverse_ckpt-120k")
    _inverse_model = InverseModel(inverse_ckpt)
    
    # Load auxiliary model
    _aux_model = AuxiliaryModel()
    
    # Load IP-Adapter model
    path_unet_sb = os.path.join(WEIGHTS_ROOT, "sbv2_0.5")
    ip_ckpt = os.path.join(WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
    _ip_sb_model = IPSBV2Model(path_unet_sb, ip_ckpt, _aux_model, with_ip_mask_controller=True)
    
    _load_time = time.perf_counter() - start
    _models_loaded = True
    print(f"[SwiftEdit] Models loaded in {_load_time:.2f}s")


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    if image_data.startswith('data:'):
        mime_type = image_data.split(',')[0]
        print(f"[decode_image] Data URI mime type: {mime_type.split(':')[1].split(';')[0]}")
        image_data = image_data.split(',', 1)[1]
    
    print(f"[decode_image] Decoded {len(base64.b64decode(image_data))} bytes")
    image_bytes = base64.b64decode(image_data)
    pil_img = Image.open(io.BytesIO(image_bytes))
    print(f"[decode_image] PIL loaded: size={pil_img.size}, mode={pil_img.mode}, format={pil_img.format}")
    return pil_img


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
    Edit image using SwiftEdit (CORRECT 3-step workflow).
    
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
    
    print(f"[SwiftEdit] ===== STARTING 3-STEP WORKFLOW =====")
    print(f"[SwiftEdit] Prompts: src='{src_prompt}', edit='{edit_prompt}'")
    print(f"[SwiftEdit] Params: scale_edit={scale_edit}, scale_non_edit={scale_non_edit}, mask_threshold={mask_threshold}, scale_ta={scale_ta}")
    
    # STEP 1: Inverse model - generate initial noise
    print(f"[SwiftEdit] Step 1/3: Running inverse model...")
    result_dict_inv = _inverse_model.inference(
        pil_image, 
        src_prompt, 
        edit_prompt, 
        scale_edit=scale_edit, 
        scale_non_edit=scale_non_edit
    )
    inverted_z = result_dict_inv["inverted_z"]
    edit_guided_z = result_dict_inv["edit_guided_z"]
    print(f"[SwiftEdit] Step 1/3 ✓ inverted_z shape={inverted_z.shape}")
    
    # STEP 2: Auxiliary model - generate edit mask
    print(f"[SwiftEdit] Step 2/3: Generating edit mask...")
    result_dict_aux = _aux_model.inference(pil_image, edit_prompt)
    mask_source = result_dict_aux["mask_source"]
    
    # Apply threshold to mask
    mask_source = mask_source.point(lambda p: to_binary(p, mask_threshold))
    print(f"[SwiftEdit] Step 2/3 ✓ mask generated and thresholded")
    
    # STEP 3: IP-SB model - final image generation
    print(f"[SwiftEdit] Step 3/3: Generating final image...")
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
    print(f"[SwiftEdit] ===== WORKFLOW COMPLETE in {inference_time:.3f}s =====")
    
    return edited_img


def handler(event):
    """
    RunPod serverless handler.
    Accepts SwiftEdit-specific parameters.
    """
    job_input = event['input']
    
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
        
        # Scale parameters (0-1 range, sent from client already divided by 100)
        scale_edit = float(job_input.get('scale_edit', 0.2))
        scale_non_edit = float(job_input.get('scale_non_edit', 1.0))
        mask_threshold = float(job_input.get('mask_threshold', 0.5))
        scale_ta = float(job_input.get('scale_ta', 1.0))
        
        print(f"[SwiftEdit] Request params: edit='{edit_prompt}', src='{source_prompt}'")
        print(f"[SwiftEdit] Scales: edit={scale_edit}, non_edit={scale_non_edit}, threshold={mask_threshold}, ta={scale_ta}")
        
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
        
        print(f"[SwiftEdit] Request complete: {total_time:.3f}s")
        
        return {
            'image': result_base64,
            'inference_time': round(total_time, 3),
            'dimensions': '512x512',
            'was_cold_start': _load_time is not None
        }
        
    except Exception as e:
        print(f"[SwiftEdit] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


if __name__ == '__main__':
    print("[SwiftEdit] Starting RunPod serverless handler...")
    runpod.serverless.start({'handler': handler})
