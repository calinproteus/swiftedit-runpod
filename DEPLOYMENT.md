# SwiftEdit RunPod Deployment Guide

SwiftEdit is now configured to download model weights **during Docker build** on RunPod's servers (fast 10Gbps+ connection).

No local downloads needed! 🚀

---

## Prerequisites

- Git configured
- RunPod account
- GitHub repository: `calinproteus/swiftedit-runpod`

---

## Step 1: Push to GitHub

```powershell
cd "C:\Users\calin\Desktop\PROTEUS\AI Agent Plugin"

git add swiftedit-runpod/
git commit -m "Add SwiftEdit with build-time weight download"
git push origin main
```

**That's it!** No large files to upload.

---

## Step 2: Deploy on RunPod

1. Go to: https://www.runpod.io/console/serverless
2. Click your endpoint: `ei14pwe8ohlx13`
3. Click **"Edit"**
4. Configure:
   - **Source:** GitHub
   - **Repository:** `calinproteus/swiftedit-runpod`
   - **Branch:** `main`
   - **Container Disk:** **50GB** (important! needed for weight extraction)
   - **GPU:** RTX A4000 or better (16GB+ VRAM)
   - **Max Workers:** 1 (increase later if needed)
   - **Idle Timeout:** 60 seconds
5. Click **"Save"** and **"Deploy"**

---

## Step 3: Monitor Build

The build will take **15-20 minutes**:
- ~2 min: Base image pull
- ~10-15 min: Download weights (5 parts, ~20GB total)
- ~2 min: Extract and cleanup
- ~1 min: Final setup

### Watch the Logs:

1. In RunPod dashboard, click your endpoint
2. Go to **"Logs"** tab
3. Look for:
   ```
   Downloading SwiftEdit weights from GitHub releases...
   ✓ Model weights ready!
   ```

---

## Step 4: Verify Deployment

Once build completes, test with:

```bash
curl -X POST https://api.runpod.ai/v2/ei14pwe8ohlx13/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
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

## Step 5: Update Proxy Server

The `.env` file is already configured:
```
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_SWIFTEDIT_ENDPOINT=https://api.runpod.ai/v2/your_endpoint_id
```

Just **restart the proxy server**:
```powershell
cd "C:\Users\calin\Desktop\PROTEUS\AI Agent Plugin\proxy-server"
npm run dev
```

---

## Step 6: Test in Figma

1. Reload Figma plugin
2. Go to **Images** tab → **Live Canvas**
3. Create a new canvas (or edit existing)
4. Switch to **Realtime** tab
5. Should see "SwiftEdit" model
6. Generate! ⚡

**Expected:**
- Cold start: ~2-3 seconds (model loading)
- Warm inference: ~0.4-0.6 seconds

---

## Troubleshooting

### Build fails: "wget: unable to resolve host"
**Solution:** Retry the build (GitHub releases might be temporarily unavailable)

### Build fails: "No space left on device"
**Solution:** Increase Container Disk to 50GB or 100GB in endpoint settings

### Endpoint times out on first request
**Solution:** First request loads models (~2s). The proxy timeout is set to 120s, so this should work. If not, send a warmup request first.

### Models not loading at runtime
**Solution:** Check build logs to ensure "✓ Model weights ready!" message appeared

---

## Performance & Costs

### Performance:
- **Cold start:** ~2-3 seconds
- **Warm inference:** ~0.4-0.6 seconds (512x512)
- **Container stays warm:** 5 minutes after last request

### Costs:
- **Build:** One-time, free
- **Runtime:** Pay per second of GPU usage
- **Storage:** No network volume needed (weights in container)

### Optimization:
- **Idle Timeout:** 60s (stops quickly when not in use)
- **Max Workers:** Start with 1, scale up if needed

---

## Success! 🎉

Your SwiftEdit endpoint is now live with instant cold starts!

**Next:** Test some real-time image edits in the Figma plugin!
