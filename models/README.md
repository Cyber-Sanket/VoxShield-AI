# VoxShield AI – Model Directory

This directory stores trained machine learning and deep learning models for automated audio deepfake detection.

## Academic & Engineering Integrity Rule

> **Important**: In accordance with project integrity standards, VoxShield AI does **not** fake predictions or display randomized confidence scores. 
> If no trained model weights exist in this folder, the system operates in **Demo / Feature Analysis Mode**, presenting verified acoustic signal properties without automated classification.

---

## Supported Model Formats

### 1. Scikit-Learn Pipeline (`.joblib` or `.pkl`)
- **Input**: 52-dimensional acoustic feature vector (MFCC means/stds, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, Spectral Flatness).
- **Target Classes**: `[0, 1]` or `["Real", "Fake"]`.
- **Expected Method**: Supports `.predict()` and optionally `.predict_proba()`.
- **Filename Example**: `voxshield_baseline.joblib`

### 2. PyTorch Neural Classifier (`.pt` or `.pth`)
- **Input**: PyTorch tensor representing either the acoustic feature vector `[Batch, 52]` or raw audio waveform for end-to-end models (Wav2Vec2, ECAPA-TDNN).
- **Output**: 2-class logits or softmax probabilities `[Real, Fake]`.
- **Filename Example**: `voxshield_wav2vec2.pt`

---

## How to Deploy a Model

1. Train a model on an audio deepfake dataset (e.g., [ASVspoof 2019/2021](https://www.asvspoof.org/), [In-the-Wild](https://deepfake-demo.cs.uni-saarland.de/in_the_wild/), or [WaveFake](https://github.com/ecker-lab/wavefake)).
2. Save your trained model directly into this directory:
   ```bash
   # Example:
   copy path/to/my_trained_model.joblib models/voxshield_model.joblib
   ```
3. Navigate to **Audio Analysis** or click **Reload Models** in the dashboard. VoxShield AI will automatically detect the weights and switch to **Model-Based Detection**.
