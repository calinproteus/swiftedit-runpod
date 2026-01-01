# SwiftEdit RunPod Serverless - Working Configuration
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

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

# Copy fix script and apply all model fixes
COPY apply_fixes.py /app/
RUN python3 /app/apply_fixes.py

# Download model weights - ONE AT A TIME with retries
WORKDIR /app/SwiftEdit

# Part aa
RUN echo "[1/5] Downloading part-aa..." && \
    for i in 1 2 3 4 5; do \
        curl -L -f --retry 10 --retry-delay 5 --retry-max-time 600 \
            -o swiftedit_weights.tar.gz.part-aa \
            https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-aa \
            && break || sleep 10; \
    done && \
    test -f swiftedit_weights.tar.gz.part-aa && \
    echo "✓ Downloaded part-aa"

# Part ab
RUN echo "[2/5] Downloading part-ab..." && \
    for i in 1 2 3 4 5; do \
        curl -L -f --retry 10 --retry-delay 5 --retry-max-time 600 \
            -o swiftedit_weights.tar.gz.part-ab \
            https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ab \
            && break || sleep 10; \
    done && \
    test -f swiftedit_weights.tar.gz.part-ab && \
    echo "✓ Downloaded part-ab"

# Part ac
RUN echo "[3/5] Downloading part-ac..." && \
    for i in 1 2 3 4 5; do \
        curl -L -f --retry 10 --retry-delay 5 --retry-max-time 600 \
            -o swiftedit_weights.tar.gz.part-ac \
            https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ac \
            && break || sleep 10; \
    done && \
    test -f swiftedit_weights.tar.gz.part-ac && \
    echo "✓ Downloaded part-ac"

# Part ad
RUN echo "[4/5] Downloading part-ad..." && \
    for i in 1 2 3 4 5; do \
        curl -L -f --retry 10 --retry-delay 5 --retry-max-time 600 \
            -o swiftedit_weights.tar.gz.part-ad \
            https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ad \
            && break || sleep 10; \
    done && \
    test -f swiftedit_weights.tar.gz.part-ad && \
    echo "✓ Downloaded part-ad"

# Part ae
RUN echo "[5/5] Downloading part-ae..." && \
    for i in 1 2 3 4 5; do \
        curl -L -f --retry 10 --retry-delay 5 --retry-max-time 600 \
            -o swiftedit_weights.tar.gz.part-ae \
            https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ae \
            && break || sleep 10; \
    done && \
    test -f swiftedit_weights.tar.gz.part-ae && \
    echo "✓ Downloaded part-ae"

# Combine parts
RUN echo "Combining parts..." && \
    cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz && \
    echo "✓ Combined into single archive" && \
    ls -lh swiftedit_weights.tar.gz

# Delete parts to free space BEFORE extraction
RUN echo "Freeing disk space..." && \
    rm -f swiftedit_weights.tar.gz.part-* && \
    echo "✓ Freed 20GB" && \
    df -h

# Extract weights (now we have enough space)
RUN echo "Extracting weights..." && \
    tar zxf swiftedit_weights.tar.gz --no-same-owner && \
    echo "✓ Extracted successfully" && \
    ls -lh swiftedit_weights/

# Delete archive
RUN rm -f swiftedit_weights.tar.gz && \
    echo "✓ Model weights ready!"

# Pre-download ALL HuggingFace models (no runtime downloads!)
RUN python3 -c "\
import torch; \
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, AutoTokenizer; \
from diffusers import AutoencoderKL, UNet2DConditionModel; \
print('[1/4] Downloading CLIP...'); \
CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K'); \
CLIPTextModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K'); \
AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K'); \
print('[2/4] Downloading VAE...'); \
AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix'); \
print('[3/4] Downloading SD 2 base...'); \
CLIPTextModel.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='text_encoder'); \
AutoTokenizer.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='tokenizer'); \
UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='unet'); \
AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-base', subfolder='vae'); \
print('✓ All HuggingFace models pre-downloaded!'); \
"

# Copy handler
WORKDIR /app
COPY handler.py /app/

# Set environment
ENV PYTHONPATH="/app/SwiftEdit:${PYTHONPATH}"

# RunPod handler entrypoint
CMD ["python", "-u", "/app/handler.py"]
