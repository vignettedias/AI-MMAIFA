# MMAIFA Image Authenticity Forensic Reasoning Engine

Evidence-centric implementation of the patented MMAIFA system for distinguishing
real, AI-generated, manipulated, unknown, and review-required images.

## Capabilities

- Pixel stream: legacy EfficientNet/AlexNet/GoogLeNet compatibility plus
  ConvNeXt-V2, Swin V2, DINOv2, CLIP ViT-L/14, SigLIP, frequency-aware ViT,
  cross-scale, patch-forensic, and specialist generator experts.
- Forensics: ELA, FFT, noise, compression, PRNU, SRM, Bayar-like residuals,
  DnCNN-like residuals, diffusion traces, spectral pyramids, provenance,
  manipulation, metadata, and camera-origin consistency.
- Semantic reasoning: CLIP when enabled plus independent auditors for anatomy,
  typography, lighting, reflection, perspective, object relations, human
  realism, contextual logic, scene geometry, and physics.
- Fusion: dynamic evidence weighting, stream reliability, disagreement
  analysis, OOD detection, conformal decision sets, and human-review routing.
- Explainability: forensic saliency maps, spectral anomaly maps, noise
  inconsistency overlays, top evidence, and confidence decomposition.
- Human review: admin feedback, review queue, correction memory, and
  conservative auto-learning.

Architecture details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run Backend

```powershell
cd "D:\Semester 3\CAO\CAO Patent Implementation"
$env:AUTH_ADMIN_KEY="creator-admin-key"
$env:PYTHONPATH=".codex_deps"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## Run Frontend

```powershell
cd "D:\Semester 3\CAO\CAO Patent Implementation\frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

## CLI Prediction

```powershell
python main.py predict --image "SampleImage\testIm5.jpeg" --compact
```

## Evidence API

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/explain" -Form @{file=Get-Item "SampleImage\testIm5.jpeg"}
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ood" -Form @{file=Get-Item "SampleImage\testIm5.jpeg"}
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/heatmap?kind=frequency_anomaly" -Form @{file=Get-Item "SampleImage\testIm5.jpeg"}
```

## Train

```powershell
python main.py train --data-dir data/cifake
python main.py train-meta --data-dir data/cifake
python main.py train-pixel --model all --data-dir data/cifake
```

## Admin Feedback

Only the creator/admin can teach the memory learner. The frontend correction panel sends feedback with the `X-Admin-Key` header.

Default local key:

```text
creator-admin-key
```

Check memory stats:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/memory/stats" -Headers @{"X-Admin-Key"="creator-admin-key"}
```

Check human-review queue:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/review/queue" -Headers @{"X-Admin-Key"="creator-admin-key"}
```
