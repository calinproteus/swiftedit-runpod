"""
SwiftEdit RunPod Serverless Handler

Handles image editing requests using SwiftEdit model.
Optimized for ~0.4s inference at 512x512.

API:
    POST /run
    {
        "input": {
            "image": "base64 encoded image",
            "edit_prompt": "what you want",
            "source_prompt": "optional - describes current image",
            "scale_edit": 0.2,        // optional - edit intensity (0-1)
            "scale_non_edit": 1.0,    // optional - background preservation (0-1)
            "mask_threshold": 0.5,    // optional - mask sensitivity (0-1)
            "scale_ta": 1.0           // optional - text alignment (0-2)
        }
    }

Response:
    {
        "output": {
            "image": "base64 encoded result",
            "inference_time": 0.43,
            "was_cold_start": false
        }
    }
"""

import os
import sys
import time
import base64
import io
import subprocess
from typing import Optional

# Add SwiftEdit to path
sys.path.insert(0, '/app/SwiftEdit')

import runpod
import torch
from PIL import Image

# Global model instances (loaded once, reused across requests)
_models_loaded = False
_inverse_model = None
_aux_model = None
_ip_sb_model = None
_load_time = None

def check_weights_exist():
    """
    Check if model weights are present.
    Weights are now baked into the Docker image during build.
    """
    WEIGHTS_ROOT = '/app/SwiftEdit/swiftedit_weights'
    
    # Check if weights exist (should always be true in production)
    if os.path.exists(WEIGHTS_ROOT) and os.path.exists(os.path.join(WEIGHTS_ROOT, 'inverse_ckpt-120k')):
        print("[SwiftEdit] Model weights found (pre-loaded in container)")
        return True
    
    # If we get here, the Docker image was built incorrectly
    print("[SwiftEdit] ERROR: Model weights not found!")
    print("[SwiftEdit] Make sure to run 'download_weights.ps1' before building the Docker image")
    return False

def load_models():
    """Load SwiftEdit models into GPU memory."""
    global _models_loaded, _inverse_model, _aux_model, _ip_sb_model, _load_time
    
    if _models_loaded:
        return
    
    # Check weights are present (should be baked into container)
    if not check_weights_exist():
        raise RuntimeError("Model weights not found in container")
    
    print("[SwiftEdit] Loading models...")
    start = time.perf_counter()
    
    # Import SwiftEdit components
    from models import InverseModel, AuxiliaryModel, IPSBV2Model
    
    WEIGHTS_ROOT = '/app/SwiftEdit/swiftedit_weights'
    
    # Load inverse model
    inverse_ckpt = os.path.join(WEIGHTS_ROOT, "inverse_ckpt-120k")
    _inverse_model = InverseModel(inverse_ckpt)
    
    # Load auxiliary model (with fixed scheduler and community components)
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
    # Handle data URI format
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
    scale_ta: float = 1.0,
    clamp_rate: float = 3.0,
    mask_threshold: float = 0.5,
) -> Image.Image:
    """
    Edit image using SwiftEdit.
    
    Args:
        pil_image: Source image (will be resized to 512x512)
        src_prompt: Description of source image (can be empty)
        edit_prompt: Desired edit
        scale_edit: Edit intensity (0-1, default 0.2)
        scale_non_edit: Background preservation (0-1, default 1.0)
        
    Returns:
        Edited PIL Image
    """
    from models import tokenize_captions, MaskController
    from torchvision.transforms.functional import to_tensor
    from torchvision.utils import save_image
    
    # Convert to RGB and resize to 512x512 (model requirement)
    # Note: Alpha compositing is done client-side before sending
    pil_img_cond = pil_image.convert('RGB').resize((512, 512))
    
    mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 500
    
    # Process image
    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to("cuda") * 2 - 1
    
    # Encode to latent
    latents = _inverse_model.vae.encode(
        processed_image.to(_inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * _inverse_model.vae.config.scaling_factor
    dub_latents = torch.cat([latents] * 2, dim=0)
    
    # Get text embeddings
    input_id = tokenize_captions(_inverse_model.tokenizer, [src_prompt, edit_prompt]).to("cuda")
    encoder_hidden_state = _inverse_model.text_encoder(input_id)[0].to(
        dtype=_inverse_model.weight_dtype
    )
    
    # Predict inverted noise
    predict_inverted_code = _inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to("cuda", dtype=_inverse_model.weight_dtype)
    
    # Estimate editing mask
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to("cuda")
    
    # Edit image
    input_sb = _ip_sb_model.alpha_t * latents + _ip_sb_model.sigma_t * inverted_noise_1
    mask_controller = MaskController(
        mask12, 
        scale_text_hiddenstate=scale_ta, 
        scale_ip_fg=scale_edit, 
        scale_ip_bg=scale_non_edit
    )
    _ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])
    res_gen_img, _ = _ip_sb_model.gen_img(
        pil_image=pil_img_cond, 
        prompts=[src_prompt, edit_prompt], 
        noise=input_sb
    )
    
    # Convert tensor to PIL
    # res_gen_img is a tensor, convert to PIL
    if isinstance(res_gen_img, torch.Tensor):
        # Normalize from [-1, 1] to [0, 1]
        res_gen_img = (res_gen_img.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
        # Convert to PIL
        res_gen_img = res_gen_img.permute(1, 2, 0).numpy()
        res_gen_img = (res_gen_img * 255).astype('uint8')
        res_gen_img = Image.fromarray(res_gen_img)
    
    return res_gen_img


def handler(event):
    """
    RunPod serverless handler.
    
    Expects input with:
        - image: base64 encoded source image
        - edit_prompt: desired edit
        - source_prompt: (optional) description of source
        - scale_edit: (optional) edit intensity 0-1
        - scale_non_edit: (optional) background preservation 0-1
    """
    try:
        # Track if this is a cold start
        was_cold_start = not _models_loaded
        
        # Load models if not already loaded
        load_models()
        
        # Parse input
        input_data = event.get('input', {})
        
        # Handle warm-up ping (no image, just checking if model is loaded)
        if input_data.get('warmup', False) or not input_data.get('image'):
            return {
                "status": "ready",
                "was_cold_start": was_cold_start,
                "load_time": _load_time if _load_time else 0
            }
        
        image_b64 = input_data.get('image')
        edit_prompt = input_data.get('edit_prompt', '')
        source_prompt = input_data.get('source_prompt', '')
        scale_edit = float(input_data.get('scale_edit', 0.2))
        scale_non_edit = float(input_data.get('scale_non_edit', 1.0))
        mask_threshold = float(input_data.get('mask_threshold', 0.5))
        scale_ta = float(input_data.get('scale_ta', 1.0))
        
        if not image_b64:
            return {"error": "No image provided"}
        
        if not edit_prompt:
            return {"error": "No edit_prompt provided"}
        
        # Decode input image
        try:
            input_image = decode_image(image_b64)
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}
        
        # Run inference
        start_time = time.perf_counter()
        
        result_image = edit_image(
            pil_image=input_image,
            src_prompt=source_prompt,
            edit_prompt=edit_prompt,
            scale_edit=scale_edit,
            scale_non_edit=scale_non_edit,
            scale_ta=scale_ta,
            mask_threshold=mask_threshold
        )
        
        inference_time = time.perf_counter() - start_time
        
        # Encode result
        result_b64 = encode_image(result_image, format='PNG')
        
        return {
            "image": result_b64,
            "inference_time": round(inference_time, 3),
            "was_cold_start": was_cold_start,
            "dimensions": "512x512"
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# RunPod serverless entrypoint
runpod.serverless.start({"handler": handler})


