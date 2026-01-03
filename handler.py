"""
SwiftEdit RunPod Serverless Handler
Based on OFFICIAL Qualcomm SwiftEdit infer.py
"""

import os
import sys
import time
import base64
import io

# Add SwiftEdit to path
sys.path.insert(0, '/app/SwiftEdit')

import runpod
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models import *

# Global model instances
_models_loaded = False
_inverse_model = None
_aux_model = None
_ip_sb_model = None
_load_time = None

SWIFTEDIT_WEIGHTS_ROOT = '/app/SwiftEdit/swiftedit_weights'


def check_weights_exist():
    """Check if model weights are present."""
    if os.path.exists(SWIFTEDIT_WEIGHTS_ROOT) and os.path.exists(os.path.join(SWIFTEDIT_WEIGHTS_ROOT, 'inverse_ckpt-120k')):
        print("[SwiftEdit] Model weights found")
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
    
    # Load models (EXACT order from official infer.py)
    inverse_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "inverse_ckpt-120k")
    _inverse_model = InverseModel(inverse_ckpt)
    
    _aux_model = AuxiliaryModel()
    
    path_unet_sb = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "sbv2_0.5")
    ip_ckpt = os.path.join(SWIFTEDIT_WEIGHTS_ROOT, "ip_adapter_ckpt-90k/ip_adapter.bin")
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
    pil_img_cond,
    src_p,
    edit_p,
    scale_ta=1,
    scale_edit=0.2,
    scale_non_edit=1,
    clamp_rate=3.0,
    mask_threshold=0.5,
):
    """
    OFFICIAL SwiftEdit implementation (copied from infer.py).
    
    Args:
        pil_img_cond: PIL Image (will be resized to 512x512)
        src_p: Source Prompt (can be empty)
        edit_p: Edit Prompt
        scale_ta: Text alignment (default 1)
        scale_edit: Edit intensity (default 0.2)
        scale_non_edit: Background preservation (default 1)
        clamp_rate: Mask clamp rate (default 3.0)
        mask_threshold: Mask threshold (default 0.5)
    
    Returns:
        Tensor [C, H, W] in range [-1, 1]
    """
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
    
    # Resize to 512x512 (OFFICIAL)
    pil_img_cond = pil_img_cond.resize((512, 512))
    
    print(f"[SwiftEdit] Edit: '{src_p}' -> '{edit_p}'")
    print(f"[SwiftEdit] Params: scale_edit={scale_edit}, scale_non_edit={scale_non_edit}, scale_ta={scale_ta}")
    
    # Process image (OFFICIAL)
    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1
    
    # Predict inverted noise (OFFICIAL)
    latents = _inverse_model.vae.encode(
        processed_image.to(_inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * _inverse_model.vae.config.scaling_factor
    dub_latents = torch.cat([latents] * 2, dim=0)
    
    input_id = tokenize_captions(_inverse_model.tokenizer, [src_p, edit_p]).to("cuda")
    encoder_hidden_state = _inverse_model.text_encoder(input_id)[0].to(
        dtype=_inverse_model.weight_dtype
    )
    
    predict_inverted_code = _inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to("cuda", dtype=_inverse_model.weight_dtype)
    
    # Estimate editing mask (OFFICIAL)
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")
    
    print(f"[SwiftEdit] Mask: {(mask12 > 0).sum().item()}/{mask12.numel()} active pixels ({(mask12 > 0).sum().item() / mask12.numel() * 100:.1f}%)")
    
    # Edit images (OFFICIAL)
    input_sb = _ip_sb_model.alpha_t * latents + _ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, scale_text_hiddenstate=scale_ta, scale_ip_fg=scale_edit, scale_ip_bg=scale_non_edit
    )
    _ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    
    print(f"[SwiftEdit] Calling gen_img...")
    res_gen_img, _ = _ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )
    
    print(f"[SwiftEdit] gen_img returned tensor: shape={res_gen_img.shape}, dtype={res_gen_img.dtype}")
    
    return res_gen_img


def handler(event):
    """RunPod serverless handler."""
    job_input = event['input']
    
    # Handle warm-up
    if job_input.get('warmup'):
        load_models()
        return {'status': 'ready', 'load_time': _load_time}
    
    # Validate
    if 'image' not in job_input or 'edit_prompt' not in job_input:
        return {'error': 'Missing required fields: image, edit_prompt'}
    
    try:
        # Load models
        if not _models_loaded:
            load_models()
        
        # Decode input
        source_image = decode_image(job_input['image'])
        
        # Extract parameters
        edit_prompt = job_input['edit_prompt']
        source_prompt = job_input.get('source_prompt', '')
        scale_edit = float(job_input.get('scale_edit', 0.2))
        scale_non_edit = float(job_input.get('scale_non_edit', 1.0))
        mask_threshold = float(job_input.get('mask_threshold', 0.5))
        scale_ta = float(job_input.get('scale_ta', 1.0))
        
        # Run OFFICIAL edit_image
        start_time = time.perf_counter()
        result_tensor = edit_image(
            source_image,
            source_prompt,
            edit_prompt,
            scale_ta=scale_ta,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
            mask_threshold=mask_threshold
        )
        
        # Convert tensor to PIL (result is [C, H, W] in [-1, 1])
        # OFFICIAL infer.py uses save_image which expects [C, H, W]
        # For PIL, we need [H, W, C] in [0, 255]
        result_np = result_tensor.cpu().clamp(-1, 1).add(1).div(2).permute(1, 2, 0).numpy()
        result_img = Image.fromarray((result_np * 255).astype('uint8'))
        
        total_time = time.perf_counter() - start_time
        
        # Encode result
        result_base64 = encode_image(result_img, 'PNG')
        
        print(f"[SwiftEdit] Complete: {total_time:.3f}s")
        
        return {
            'image': result_base64,
            'inference_time': round(total_time, 3),
            'dimensions': '512x512'
        }
        
    except Exception as e:
        print(f"[SwiftEdit] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


if __name__ == '__main__':
    print("[SwiftEdit] Starting RunPod serverless handler...")
    runpod.serverless.start({'handler': handler})
