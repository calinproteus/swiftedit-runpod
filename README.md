# SwiftEdit RunPod Deployment - Simplified Approach

## Quick Deploy (Use Pre-built Image)

If the build keeps failing, use this pre-built image instead:

```
stablediffusionapi/sd-image-edit:latest
```

Or try the official PyTorch base with manual setup:

```
pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

## Manual Pod Setup (What Actually Works)

If serverless keeps failing, create a **regular RunPod GPU Pod**:

1. Go to RunPod → Pods → Deploy
2. Choose: **RTX 4000 Ada** or **RTX A4000** 
3. Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`
4. Container Disk: 20 GB
5. Deploy & SSH in

Then run these commands in the pod:

```bash
cd /workspace
git clone https://github.com/Qualcomm-AI-research/SwiftEdit.git
cd SwiftEdit

# Download weights
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-aa
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ab
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ac
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ad
wget https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0/swiftedit_weights.tar.gz.part-ae

# Combine
cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
tar zxf swiftedit_weights.tar.gz

# Install deps
pip install runpod transformers diffusers einops
```

This gives you a running pod with SwiftEdit that you can test directly!

