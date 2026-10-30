"""
VoxShield AI - About Project Page
Team Ghost Protocol | AI & Cyber Security
Architecture, Technologies, Objectives, Academic Integrity, and Limitations.
"""

import streamlit as st

st.set_page_config(
    page_title="About | VoxShield AI",
    page_icon="🛡️",
    layout="wide"
)

from config import APP_NAME, APP_TAGLINE, TEAM_NAME, DOMAIN, VERSION
from modules.ui_components import apply_custom_css, render_header, render_sidebar_info

apply_custom_css()
render_sidebar_info(st.session_state.get("detector", None).model_status if "detector" in st.session_state else None)
render_header("About Project: Architecture, Methodology & Integrity")

col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown(f"### 🛡️ {APP_NAME} – AI-Powered Audio Deepfake Detection System")
    st.markdown(
        f"""
        **VoxShield AI** is an academic cybersecurity demonstration platform engineered by team **{TEAM_NAME}** 
        to showcase state-of-the-art methodologies in synthetic voice detection, acoustic signal processing, 
        and machine learning-driven digital audio forensics.
        
        As generative artificial intelligence technologies (diffusion audio, neural vocoders like HiFi-GAN and BigVGAN, 
        and zero-shot voice cloning architectures such as XTTS, VALL-E, and ElevenLabs) proliferate, the security threat 
        of synthetic voice fraud, CEO impersonation, and disinformation attacks has escalated exponentially. 
        VoxShield AI demonstrates how forensic spectral analysis and deep learning feature extraction can identify 
        anomalies in synthetic audio signals.
        """
    )

    st.markdown("---")
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    1. **Signal Conditioning Demonstration**: Illustrate how raw audio is standardized through channel downmixing, 16 kHz resampling, dynamic peak scaling, and spectral noise reduction.
    2. **Multi-View Acoustic Analysis**: Provide forensic visualizations of 2D time-frequency structures (Mel-Spectrograms, MFCCs) and 1D temporal trajectories (Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy).
    3. **Academic Integrity & Transparency**: Enforce a strict distinction between true model-evaluated predictions and feature-only demo analysis. Zero fabricated accuracy metrics or randomized classifications.
    4. **Modular Architecture**: Deliver an extensible pipeline supporting plug-and-play integration for PyTorch neural models (Wav2Vec2, ECAPA-TDNN) and Scikit-Learn classifiers.
    5. **Forensic Auditing**: Persist complete analysis metadata in a local SQLite database without compromising privacy by storing raw audio recordings.
    """)

    st.markdown("---")
    st.markdown("### ⚖️ System Limitations & Academic Integrity")
    st.markdown("""
    > [!IMPORTANT]
    > **Academic Integrity Statement:** This is an academic demonstration and engineering prototype. 
    > In strict accordance with research ethics, VoxShield AI never fabricates model accuracy, dataset benchmark scores, 
    > or random "Real/Fake" predictions when trained weights are absent.

    - **Demo Analysis Mode**: When no trained weights are loaded in the `models/` directory, the system operates purely in Feature Analysis Mode. 
      Automated classification is marked as **"Analysis Only – Prediction Unavailable"**.
    - **Acoustic Generalization**: Handcrafted acoustic features (MFCC, Centroid, Rolloff) capture vocoder anomalies well on uncompressed audio, but can suffer degraded performance on lossy audio (e.g. WhatsApp, telephony GSM codecs).
    - **Production Readiness**: For enterprise-grade verification, a multi-modal ensemble combining self-supervised representations (Wav2Vec2.0 / HuBERT) with linear-frequency cepstral coefficients (LFCC) trained on multi-million sample datasets (ASVspoof 2021) is required.
    """)

with col_side:
    st.markdown("### 👥 Project Credentials")
    st.markdown(f"""
    <div class="cyber-card">
        <div class="card-label">Project Name</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00FFA3;">{APP_NAME}</div>
        <hr style="border-color: rgba(30,41,59,0.8); margin: 8px 0;">
        <div class="card-label">Engineering Team</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00E5FF;">{TEAM_NAME}</div>
        <hr style="border-color: rgba(30,41,59,0.8); margin: 8px 0;">
        <div class="card-label">Research Domain</div>
        <div style="font-size: 1.0rem; font-weight: 600; color: #F8FAFC;">{DOMAIN}</div>
        <hr style="border-color: rgba(30,41,59,0.8); margin: 8px 0;">
        <div class="card-label">Current Version</div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #94A3B8;">v{VERSION} (Python 3.12 / Windows 11)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    - **Language**: Python 3.12+ (Windows 11 Compatible)
    - **Frontend / Dashboard**: Streamlit (Cyber Dark Theme)
    - **Audio Processing**: Librosa, SoundFile, SciPy
    - **Mathematics & Data**: NumPy, Pandas
    - **Visualizations**: Plotly Interactive Charts, Matplotlib
    - **ML Frameworks**: Scikit-Learn, PyTorch Architecture
    - **Database**: SQLite 3 (Audit Trail & Live Metrics)
    - **Testing**: PyTest Suite
    """)

st.markdown("---")

# Pipeline Architecture Diagram
st.markdown("### 🏗️ Technical Pipeline Architecture")
st.markdown("""
```
Audio Ingestion (WAV / MP3 / FLAC)
         │
         ▼
[Format & Magic Byte Validator] ── (In-memory, Zero persistent audio file storage)
         │
         ▼
[Signal Conditioning Pipeline]
   ├── Downmix Channels to Single-Channel Mono
   ├── Resample to Standard 16,000 Hz
   ├── Dynamic Range Peak Normalization (0.95)
   └── Spectral Subtraction Noise Reduction
         │
         ▼
[Acoustic Feature Extractor]
   ├── 128-Band Mel-Spectrogram (dB)
   ├── 20-Order MFCC Matrix + Delta Trajectories
   ├── Spectral Centroid, Bandwidth & Rolloff (85%)
   └── Zero Crossing Rate (ZCR), RMS Energy & Spectral Flatness
         │
         ├────────────────────────────────────────┐
         ▼                                        ▼
[Mode A: Model-Based Detection]          [Mode B: Demo Feature Analysis]
   • Checks models/ for trained weights     • No active weights detected
   • Evaluates feature vector / Wav2Vec2    • Transparent Academic Integrity
   • Genuine Real / Fake / Inconclusive     • "Analysis Only – Prediction Unavailable"
   • Real statistical confidence score      • Zero fabricated metrics
         │                                        │
         └───────────────────┬────────────────────┘
                             ▼
               [SQLite Forensics Database]
               [Interactive Cyber Dashboard]
```
""")
