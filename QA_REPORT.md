# Quality Assurance & Academic Validation Audit Report
**Project:** VoxShield AI – AI-Powered Audio Deepfake Detection System  
**Team:** Ghost Protocol | **Domain:** AI & Cyber Security  
**Audit Timestamp:** October 30, 2026  
**Auditor Role:** Senior AI/ML & Cyber Security Quality Assurance  

---

## 1. Verified Features

| Category | Component / Module | Audit Finding / Verified Behavior | Status |
| :--- | :--- | :--- | :--- |
| **User Interface** | `app.py`, `pages/1_Dashboard.py` | Dark cybersecurity theme (`#0A0F1D`), neon accents (`#00FFA3`, `#00E5FF`, `#3B82F6`), pulse badges, and live DB counters starting at zero without fabrication. | **VERIFIED** |
| **Audio Ingestion** | `modules/audio_loader.py` | In-memory decoding for WAV, MP3, and FLAC. Validates RIFF/FLAC/ID3 headers. Enforces 25 MB size and ≥0.2s duration limits. Rejects corrupted or spoofed extensions. Zero permanent audio disk storage. | **VERIFIED** |
| **Signal Conditioning** | `modules/preprocessing.py` | Multi-channel downmixing to mono, polyphase/Kaiser resampling to 16,000 Hz, dynamic peak normalization (`[-0.95, +0.95]`), and spectral subtraction noise reduction. | **VERIFIED** |
| **Feature Extraction** | `modules/feature_extraction.py` | Computes 128-band Mel-Spectrograms, 20-band MFCC matrices + deltas, Spectral Centroid, Bandwidth, Rolloff (85%), ZCR, RMS Energy, and Flatness. Generates 64-element feature vectors. | **VERIFIED** |
| **Dual-Mode Engine** | `modules/deepfake_detector.py` | Strict separation of modes: **Mode A** runs trained models with genuine confidence; **Mode B** (Demo Mode) outputs `Analysis Only - Prediction Unavailable` with `confidence=None` and zero fake numbers. | **VERIFIED** |
| **Visualizations** | `modules/visualizations.py` | Interactive Plotly charts: dual waveform comparison, Mel-Spectrogram heatmap (dB), MFCC heatmap, multi-curve spectral trajectories, and radar fingerprint. | **VERIFIED** |
| **Database & Audit** | `modules/database.py`, `pages/5_Analysis_History.py` | SQLite database (`data/voxshield.db`) stores metadata and JSON feature summaries. Tested: record insertion, search/filtering, CSV export, JSON export, and single-record deletion. | **VERIFIED** |
| **Multi-Page App** | `pages/1` through `pages/6` | All 6 multi-page views compile and render without runtime errors on Streamlit 1.66+. | **VERIFIED** |

---

## 2. Automated Test Suite Results

- **Test Framework:** PyTest 9.1.1 on Python 3.14.3 (Windows 11 x64)
- **Total Test Cases:** 19
- **Passed:** 19 (100%)
- **Failed:** 0
- **Execution Time:** ~4.00s – 5.34s

```
tests/test_audio_loader.py::test_load_valid_wav PASSED                   [  5%]
tests/test_audio_loader.py::test_reject_unsupported_extension PASSED     [ 10%]
tests/test_audio_loader.py::test_reject_truncated_or_empty_bytes PASSED  [ 15%]
tests/test_audio_loader.py::test_reject_invalid_wav_header PASSED        [ 21%]
tests/test_audio_loader.py::test_reject_too_short_audio PASSED           [ 26%]
tests/test_database.py::test_initial_db_stats_are_zero PASSED            [ 31%]
tests/test_database.py::test_insert_and_retrieve_record PASSED           [ 36%]
tests/test_database.py::test_search_and_filter PASSED                    [ 42%]
tests/test_database.py::test_delete_and_clear_history PASSED             [ 47%]
tests/test_detector.py::test_demo_mode_when_no_models_present PASSED     [ 52%]
tests/test_detector.py::test_corrupted_model_file_handled_gracefully PASSED [ 57%]
tests/test_detector.py::test_model_based_detection_when_model_present PASSED [ 63%]
tests/test_feature_extraction.py::test_feature_explanations_exist PASSED [ 68%]
tests/test_feature_extraction.py::test_compute_statistics PASSED         [ 73%]
tests/test_feature_extraction.py::test_extract_audio_features_shapes PASSED [ 78%]
tests/test_preprocessing.py::test_to_mono_conversion PASSED              [ 84%]
tests/test_preprocessing.py::test_normalize_audio_peak PASSED            [ 89%]
tests/test_preprocessing.py::test_resample_audio PASSED                  [ 94%]
tests/test_preprocessing.py::test_full_preprocessing_pipeline PASSED     [100%]
============================= 19 passed in 4.00s ==============================
```

---

## 3. Model Training & Dataset Audit

### Training Script Inspected: `tools/train_baseline.py`
- **Algorithm:** `RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)`
- **Artifact Output:** `models/voxshield_baseline.joblib`
- **Input Dimension:** 64-element acoustic feature vector

### Source Data & Labeling:
- **Real Voice Prototype (Label 0):** `assets/sample_audio/sample_real_voice.wav` (3.5s, 16 kHz synthetic speech simulation with natural F0 vibrato, vocal tract formants F1-F3, and breath noise).
- **Synthetic Voice Prototype (Label 1):** `assets/sample_audio/sample_synthetic_voice.wav` (3.5s, 16 kHz synthetic vocoder simulation with flat F0 pitch, buzziness, and low-pass frequency cutoff at 3.8 kHz).
- **Dataset Synthesis / Augmentation:**
  - 60 Gaussian-perturbed feature vectors for Class 0 (Real).
  - 60 Gaussian-perturbed feature vectors for Class 1 (Fake).
  - Total instances: 120. Split: 75% train (90 samples), 25% test (30 samples).

### Data Leakage & Evaluation Assessment:
> [!CRITICAL]
> **Academic Integrity Finding on Model Generalization:**
> 1. **Synthetic Replication Leakage:** The 120 instances were created by adding Gaussian noise to only **two base vectors**. Because perturbation occurred *before* the train/test split, the test split contains noisy copies of the exact same two prototypes present in the training split.
> 2. **Trivial Metric Separation:** The 100% score reported during baseline generation is an artifact of separating two distinct synthetic clusters in 64-dimensional space, **not** genuine acoustic learning across varied human speakers.
> 3. **No External Evaluation Dataset:** The repository currently does **not** package an independent multi-speaker benchmark dataset (such as ASVspoof 2021, WaveFake, or In-The-Wild).
> 4. **Generalization Status:** **Model generalization and real-world accuracy are UNVERIFIED.** In adherence to academic integrity rules, no fabricated cross-dataset precision, recall, or confusion matrices are published.

---

## 4. Demo Mode Integrity Verification

- **Mechanism:** When no compatible `.joblib` or `.pt` weights exist in `models/` (or when weights are moved/unloaded), `DeepfakeDetectionPipeline` automatically falls back to `MODE_DEMO`.
- **Integrity Test (50 iterations with varied inputs):**
  - `result.mode`: Strictly returns `"Demo / Feature Analysis Mode"`
  - `result.prediction`: Strictly returns `"Analysis Only - Prediction Unavailable"`
  - `result.confidence`: Strictly returns `None`
  - `result.class_probabilities`: Strictly returns `None`
- **Outcome:** Zero randomized predictions, zero fake confidence percentages, and zero fabricated classifications.

---

## 5. Known Limitations

1. **Acoustic Generalization:** Handcrafted spectral features (MFCCs, rolloff, centroid) detect vocoder filter cutoffs well on uncompressed audio, but can degrade on lossy compressed speech (e.g. WhatsApp VoIP, Opus, AMR telephony codecs).
2. **Prototype Baseline Limitation:** The included baseline model (`voxshield_baseline.joblib`) serves strictly as a pipeline and wiring verification model; it should not be deployed in operational security environments without training on comprehensive speech corpora.
3. **Absence of Pre-trained Neural Weights:** Deep self-supervised models (Wav2Vec2-XLSR, ECAPA-TDNN) require downloading large weights (300 MB - 1.2 GB) which are omitted by default to ensure lightweight, zero-dependency installation.

---

## 6. Pending / Recommended Future Work

1. **Curated Benchmark Ingestion:** Integrate a standardized benchmark evaluation script using an open subset of ASVspoof 2021 LA or WaveFake.
2. **Self-Supervised Feature Extractor:** Integrate a pre-trained HuggingFace `facebook/wav2vec2-base` embedding pipeline to extract 768-dimensional latent representations alongside hand-crafted MFCCs.
3. **Data Splitting Integrity:** Refactor baseline training to enforce speaker-disjoint splitting (ensuring augmented samples from the same audio recording never appear across both train and test splits).

---

## 7. Commands to Run the Application

### 1. Launch Web Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Run Automated Unit Tests
```powershell
python -m pytest tests/ -v
```

### 3. Re-train / Refresh Baseline Model
```powershell
python tools/train_baseline.py
```

### 4. Re-generate Benchmark Audio Samples
```powershell
python tools/generate_sample_audio.py
```
