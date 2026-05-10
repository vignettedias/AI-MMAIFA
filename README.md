# AI Image Authenticity Detector

Multi-stream real-vs-AI image verification system with:

- Pixel ensemble: EfficientNet, AlexNet, GoogLeNet
- Forensics: ELA, FFT, noise, compression, spectral intelligence, camera provenance, manipulation, metadata
- Semantic reasoning: CLIP when enabled, deterministic fallback otherwise
- Synthetic landscape/render cue for fantasy nature scenes
- Synthetic architecture/render cue for generated buildings and plazas
- OOD, uncertainty, reliability, and human-review policy outputs
- Feature fusion and XGBoost/weighted meta-classifier
- Creator-only correction memory plus conservative auto-learning
- React frontend connected to the FastAPI backend

## Run Backend

```powershell
cd "D:\Semester 3\CAO\CAO Patent Implementation"
$env:AUTH_ADMIN_KEY="creator-admin-key"
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
