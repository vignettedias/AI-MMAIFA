# MMAIFA – Deepfake Detection Algorithm

## Introduction

**MMAIFA (Multi-Modal AI Inference & Framework Architecture)** is an adaptable framework that uses Artificial Intelligence models to analyze images and videos to detect deepfake content.

---

## Working Process (Step-by-Step)

### Step 1: Data Acquisition
- **Input data** includes:
  - Images (in JPEG, PNG format)
  - Videos (optional)
- File validation process checks image or video:

---

### Step 2: Preprocessing Stage
- Image resizing and normalization
- Face detection and alignment
- Frame extraction from the video (if available)
- Isolating noise/artifacts

---

### Step 3: Feature Extraction
- Identification of facial landmarks
- Texture analysis and pixel pattern recognition
- Transforming to frequency domain (using FFT)
- Extraction of metadata (EXIF, timestamps)

---

### Step 4: AI Inference
- Sending preprocessed input to the AI model:
  - Convolutional Neural Networks (CNNs)
  - Vision Transformers
- Detects:
  - GAN-related artifacts
  - Facial inconsistencies
  - Synthesized features
- Probability output

---

### Step 5: Validation
- Comparison of results from different models
- Threshold implementation
- Calculating confidence level

---

### Step 6: Results Output
- Final decision made between:
  - Real
  - Fake
- Confidence level
