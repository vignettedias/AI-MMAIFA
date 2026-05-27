# Multi-Modal Image Authenticity Verification Plan

## Objective

Build a modular image authenticity system that distinguishes real photographs from AI-generated images by fusing independent evidence streams:

1. Pixel-level deep model ensemble
2. Forensic artifact analysis
3. Semantic consistency reasoning
4. Feature aggregation
5. Meta-classifier final decision and confidence
6. Creator-controlled dynamic memory for continual correction
7. Uncertainty, OOD detection, and human-review escalation

The architecture preserves the patent idea: multi-modal fusion improves robustness over any single detector.

## Architecture

```text
Input image
  |
  +-- Pixel stream
  |   +-- EfficientNet binary detector
  |   +-- AlexNet binary detector
  |   +-- GoogLeNet binary detector
  |   +-- Pixel ensemble score
  |
  +-- Forensic stream
  |   +-- Error Level Analysis score
  |   +-- FFT frequency artifact score
  |   +-- Noise inconsistency score
  |   +-- Compression/resampling artifact score
  |   +-- Spectral intelligence score
  |   +-- Camera provenance score
  |   +-- Manipulation/blending score
  |   +-- Metadata provenance score
  |   +-- Forensic ensemble score
  |
  +-- Semantic stream
  |   +-- CLIP prompt comparison when enabled
  |   +-- Semantic anomaly fallback
  |   +-- Portrait synthetic cue
  |   +-- Stylized/composite cue
  |   +-- Low-resolution composite cue
  |   +-- Synthetic landscape/render cue
  |   +-- Synthetic architecture/render cue
  |
  +-- Feature fusion
  |   +-- Ordered, clipped [0, 1] feature vector
  |
  +-- Meta-classifier
  |   +-- XGBoost model when trained and trusted
  |   +-- Weighted multi-stream fallback/guard
  |
  +-- Uncertainty and review policy
  |   +-- OOD score
  |   +-- Epistemic/aleatoric uncertainty
  |   +-- Reliability estimate
  |   +-- Human-review queue
  |
  +-- Dynamic memory
      +-- Admin/creator feedback exact-image override
      +-- Nearest feature-vector memory blending
      +-- Conservative auto-learning from high-confidence predictions
```

All fused features are normalized so higher values indicate stronger fake/AI-generation evidence.

## File Structure

```text
.
|-- PLANS.md
|-- main.py
|-- pipeline.py
|-- requirements.txt
|-- api/
|   |-- __init__.py
|   `-- app.py
|-- data/
|   `-- README.md
|-- forensics/
|   |-- __init__.py
|   |-- analyzer.py
|   |-- compression.py
|   |-- ela.py
|   |-- frequency.py
|   `-- noise.py
|-- frontend/
|   |-- package.json
|   |-- src/
|   `-- vite.config.js
|-- fusion/
|   |-- __init__.py
|   `-- aggregator.py
|-- meta/
|   |-- __init__.py
|   |-- classifier.py
|   `-- memory.py
|-- models/
|   `-- README.md
|-- pixel_models/
|   |-- __init__.py
|   |-- alexnet.py
|   |-- base.py
|   |-- efficientnet.py
|   |-- ensemble.py
|   `-- googlenet.py
|-- semantic/
|   |-- __init__.py
|   `-- clip_reasoner.py
|-- tests/
`-- utils/
```

## Training Pipeline

`utils.dataset.DirectoryImageDataset` supports CIFAKE-style and extensible directory layouts. It recursively scans images and infers labels from directory names:

- Real labels: `real`, `authentic`, `photograph`, `human`, `0`
- Fake labels: `fake`, `ai`, `generated`, `synthetic`, `deepfake`, `1`

Pixel models can be trained independently:

```bash
python main.py train-pixel --model efficientnet --data-dir data/cifake
python main.py train-pixel --model alexnet --data-dir data/cifake
python main.py train-pixel --model googlenet --data-dir data/cifake
```

The meta-classifier is trained on aggregated stream features:

```bash
python main.py train-meta --data-dir data/cifake
```

If no real dataset is present, a tiny mock path keeps the pipeline runnable:

```bash
python main.py train --mock-if-missing
```

## Inference Pipeline

```bash
python main.py predict --image path/to/image.jpg
```

Execution:

1. Load image and compute stable content hash.
2. Reuse cached stream features for repeated images when possible.
3. Run EfficientNet, AlexNet, and GoogLeNet pixel streams.
4. Run ELA, FFT, noise, and compression forensic analyzers.
5. Run CLIP or fallback semantic analysis with specialized synthetic cues.
6. Fuse normalized scores into the ordered feature vector.
7. Run the meta-classifier plus weighted evidence guard.
8. Blend with creator/admin memory when exact or nearby learned examples exist.
9. Estimate OOD, epistemic uncertainty, aleatoric uncertainty, and reliability.
10. Apply tiered review policy: REAL, AI_GENERATED, MANIPULATED, or HUMAN_REVIEW_REQUIRED.
11. Return label, confidence, details, feature order, stream scores, review decision, and memory decision.

## API

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict` with multipart field `file`
- `POST /feedback` with `X-Admin-Key` header for creator-only teaching
- `GET /memory/stats` with `X-Admin-Key` header
- `GET /review/queue` with `X-Admin-Key` header

Default local admin key:

```text
creator-admin-key
```

Override it in production:

```bash
set AUTH_ADMIN_KEY=your-secret-key
```

## Dependencies

Core runtime:

- Python 3.10+
- Pillow
- NumPy
- FastAPI
- Uvicorn
- python-multipart
- XGBoost

Frontend:

- React
- Vite
- Tailwind CSS
- Axios

Development/test:

- pytest
- httpx

Optional advanced model backends:

- PyTorch
- Torchvision
- Transformers
