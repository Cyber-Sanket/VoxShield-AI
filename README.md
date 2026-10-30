# VoxShield AI – AI-Powered Audio Deepfake Detection System

**Detect Synthetic Voices. Protect Digital Trust.**

[![Team](https://img.shields.io/badge/Team-Ghost%20Protocol-00FFA3?style=for-the-badge)](https://github.com/)
[![Domain](https://img.shields.io/badge/Domain-AI%20%26%20Cyber%20Security-00E5FF?style=for-the-badge)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3B82F6?style=for-the-badge)](https://www.python.org/)
[![UI](https://img.shields.io/badge/UI-Streamlit%20Dark%20SOC-A855F7?style=for-the-badge)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic%20Demo-10B981?style=for-the-badge)](LICENSE)

---

## 🛡️ 1. Project Overview

**VoxShield AI** is a cyber-forensics platform developed by team **Ghost Protocol** designed to detect and analyze synthetic speech, voice clones, and audio deepfakes. 

With the rapid emergence of neural text-to-speech (TTS), zero-shot voice cloning, and diffusion-based acoustic vocoders, malicious actors can easily forge biometric voice authentication, execute CEO voice impersonation frauds, and orchestrate social engineering campaigns. 

VoxShield AI provides security analysts, investigators, and researchers with:
- High-fidelity **audio signal conditioning** (resampling, downmixing, normalization, and spectral noise reduction).
- Detailed **time-frequency acoustic forensics** (Mel-Spectrograms, MFCC cepstral matrices, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, and Flatness).
- A **modular detection architecture** enforcing strict academic integrity: separating genuine **Model-Based Detection** from transparent **Demo / Feature Analysis Mode**.
- An auditable **SQLite forensics registry** with live statistics, search, filters, and CSV export.

---

## 🏛️ 2. Core Integrity Principles

> **IMPORTANT ACADEMIC INTEGRITY NOTICE:**
> This system is built for honest academic and cybersecurity research.
> - **Zero Fabricated Metrics**: VoxShield AI never invents random confidence scores or fake predictions.
> - **Clear Operating Modes**:
>   - **Mode A (Model-Based Detection)**: Runs only when trained model weights exist in `models/`. It returns real predictions (`Real`, `Fake`, `Inconclusive`) and genuine model probabilities.
>   - **Mode B (Demo / Feature Analysis Mode)**: Active when no trained weights are present. It extracts and displays all acoustic parameters, explicitly labeled as **"Analysis Only – Prediction Unavailable"**.
> - **Zero Permanent Audio Retention**: Audio is inspected strictly in-memory during session analysis; raw audio is never permanently saved to disk or database.

---

## 🛠️ 3. Technology Stack

- **Core Runtime**: Python 3.12+ (Windows 11 Compatible)
- **Web Interface**: Streamlit with custom CSS cybersecurity dark theme (`assets/style.css`)
- **Signal Processing**: `librosa`, `soundfile`, `scipy.signal`
- **Numerical Processing**: `numpy`, `pandas`
- **Visualization Engines**: `plotly` (interactive HTML5 heatmaps and traces), `matplotlib`
- **Machine Learning**: `scikit-learn` (Random Forest / Logistic baseline classifiers), `torch` architecture compatible
- **Database Engine**: `sqlite3` (forensic audit log, query filters, live counters)
- **Quality Assurance**: `pytest` comprehensive test suite

---

## 📁 4. Project Architecture

```
VoxShield-AI/
│
├── app.py                     # Main application entrypoint & SOC Dashboard
├── config.py                  # Global application constants, paths, and themes
├── requirements.txt           # Python dependency definitions
├── README.md                  # Detailed documentation and setup guide
│
├── modules/                   # Modular backend architecture
│   ├── __init__.py
│   ├── audio_loader.py        # In-memory validation, format & magic byte checking
│   ├── preprocessing.py       # Mono conversion, 16kHz resampling, normalization, noise reduction
│   ├── feature_extraction.py  # MFCC, Mel-Spectrogram, Centroid, Rolloff, ZCR, RMS
│   ├── deepfake_detector.py   # Dual-mode detection engine (Model-based vs. Demo mode)
│   ├── visualizations.py      # Plotly interactive cybersecurity charts
│   ├── database.py            # SQLite audit trail & live metrics manager
│   └── ui_components.py       # Reusable cyber header, metric cards, and sidebar
│
├── pages/                     # Streamlit multi-page interface
│   ├── 1_Dashboard.py         # Home Dashboard & live security metrics
│   ├── 2_Audio_Analysis.py    # Audio upload, conditioning & pipeline execution
│   ├── 3_Feature_Extraction.py # Acoustic feature visualizer & forensic explanations
│   ├── 4_Detection_Results.py # Executive verdict, acoustic report & JSON export
│   ├── 5_Analysis_History.py  # SQLite record viewer, filters, and CSV export
│   └── 6_About.py             # Team Ghost Protocol, domain, objectives & limitations
│
├── models/                    # Model weights storage directory
│   └── README.md              # Instructions for deploying trained weights
│
├── assets/                    # Static styling and acoustic benchmark files
│   ├── style.css              # Dark cybersecurity stylesheet
│   └── sample_audio/          # Pre-synthesized vocal benchmarks
│       ├── sample_real_voice.wav
│       └── sample_synthetic_voice.wav
│
├── tools/                     # Utility scripts
│   ├── generate_sample_audio.py # Synthesizes benchmark WAV audio files
│   └── train_baseline.py       # Trains and exports Scikit-Learn baseline classifier
│
└── tests/                     # Automated unit test suite
    ├── __init__.py
    ├── test_audio_loader.py
    ├── test_preprocessing.py
    ├── test_feature_extraction.py
    ├── test_detector.py
    └── test_database.py
```

---

## 🚀 5. Quickstart & Installation (Windows 11)

### Step 1: Clone or Navigate to Directory
```powershell
cd "e:\Projects\VoxShield AI"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 6. Testing Instructions

### Run Automated Unit Tests
To verify all unit tests (audio loader, preprocessing, features, detector integrity, SQLite):
```powershell
pytest tests/ -v
```

### Testing Audio Analysis in Browser
1. Open the **Audio Analysis** page (`pages/2_Audio_Analysis.py`).
2. You can either:
   - Click the **"Select Built-in Forensic Sample"** tab to test immediately with `sample_real_voice.wav` or `sample_synthetic_voice.wav`.
   - Or upload your own `.wav`, `.mp3`, or `.flac` audio recording (up to 25 MB).
3. Review audio metadata (Duration, Sample Rate, Channels) and play the audio in the browser.
4. Click **"Run VoxShield Analysis Pipeline"**.
5. Observe:
   - Waveform conditioning (Original vs. Processed).
   - Applied preprocessing steps.
   - Transparent mode badge (**Demo / Feature Analysis Mode** or **Model-Based Detection**).
   - Real-time entry recorded in SQLite database.

---

## 🧠 7. Activating Model-Based Detection (Mode A)

By default, VoxShield AI operates in **Demo / Feature Analysis Mode** to uphold academic integrity when no pre-trained weights are present.

To enable **Model-Based Detection**:
1. Run the included baseline model training tool:
   ```powershell
   python tools/train_baseline.py
   ```
   This trains a genuine Random Forest acoustic feature classifier on synthesized voice variations and saves it to:
   `models/voxshield_baseline.joblib`
2. In the dashboard or sidebar, click **"Rescan Models Directory"**.
3. VoxShield AI will detect the newly added model and switch to **Model-Based Detection**, calculating real predictions and genuine confidence scores.

To return to Demo Mode, simply remove or move the `.joblib` / `.pt` file from `models/`.

---

## 📊 8. Forensic Acoustic Features Explained

| Acoustic Descriptor | Physical Meaning | Forensic Deepfake Indicator |
| :--- | :--- | :--- |
| **Mel-Spectrogram** | Energy distribution across perceptual Mel scale | Vocoders (HiFi-GAN, WaveGlow) exhibit checkerboard artifacts and unnatural high-frequency cutoffs. |
| **MFCC (1–20)** | Vocal tract resonance envelope | Synthetic speech displays discontinuities and unnatural variances in high-order coefficients (MFCC 12–20). |
| **Spectral Centroid** | Center of mass ("brightness") of frequency spectrum | Deepfakes often suffer low-pass filtering or erratic high-frequency noise spikes. |
| **Spectral Bandwidth** | Frequency spread around centroid | Natural human speech exhibits dynamic bandwidth shifts during consonant transitions; vocoders are often overly rigid. |
| **Spectral Rolloff (85%)** | Frequency below which 85% energy lies | Telltale steep energy drop-offs indicate internal neural downsampling. |
| **Zero Crossing Rate (ZCR)** | Rate of sign-changes across waveform | Human fricatives ('s', 'sh', 'f') show rich ZCR dynamics; cloned audio often blurs unvoiced transitions. |
| **RMS Energy** | Root Mean Square instantaneous loudness | Cloned voices often lack natural breathing decay and subtle human micro-tremors. |
| **Spectral Flatness** | Measure of tonality vs. noisiness | Identifies vocoder buzziness and phase reconstruction flaws. |

---

## ⚠️ 9. System Limitations

1. **Acoustic Generalization**: Handcrafted spectral features perform reliably on clean recordings, but lossy compression (such as telephony codecs or WhatsApp audio) can attenuate high-frequency vocoder cues.
2. **Evolving Generator Architectures**: Newer continuous flow matching and diffusion vocoders (e.g., BigVGAN, Voicebox) reduce spectral artifacts, necessitating continuous retraining.
3. **Multi-Modal Ensemble Recommended**: For mission-critical forensic deployments, spectral feature analysis should be combined with self-supervised audio representations (Wav2Vec2, HuBERT, ECAPA-TDNN) trained on multi-speaker benchmarks (ASVspoof, In-The-Wild).

---

## 👥 10. Team Credentials

- **Project Name:** VoxShield AI
- **Team Name:** Ghost Protocol
- **Domain:** AI & Cyber Security
- **Version:** 1.0.0 (Windows 11 Compatible)
