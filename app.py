"""
VoxShield AI – AI-Powered Audio Deepfake Detection System
Main Entrypoint & Security Operations Center (SOC) Dashboard
Team: Ghost Protocol | Domain: AI & Cyber Security
"""

import streamlit as st
from pathlib import Path

# Configure Streamlit page must be first command
st.set_page_config(
    page_title="VoxShield AI | Ghost Protocol",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import (
    APP_NAME, APP_TAGLINE, TEAM_NAME, DOMAIN, VERSION,
    DB_PATH, MODELS_DIR
)
from modules.database import init_db, get_system_stats, get_records_dataframe
from modules.deepfake_detector import DeepfakeDetectionPipeline
from modules.ui_components import apply_custom_css, render_header, render_sidebar_info, render_metric_cards

# Initialize Database and Pipeline State
init_db()

if "detector" not in st.session_state:
    st.session_state.detector = DeepfakeDetectionPipeline()

apply_custom_css()
render_sidebar_info(st.session_state.detector.model_status)
render_header()

# System Metrics Section
st.markdown("### 📊 Live System Metrics")
stats = get_system_stats()
render_metric_cards(stats)

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture & Active Mode Overview
col_left, col_right = st.columns([1.6, 1.0])

with col_left:
    st.markdown("### 🛡️ Mission & Operational Workflow")
    st.markdown(
        f"""
        **{APP_NAME}** is an advanced cyber-defense prototype developed by team **{TEAM_NAME}** 
        to analyze, detect, and mitigate the rising threat of synthetic voice cloning and audio deepfakes. 
        Through high-resolution spectral signal processing, acoustic feature extraction, and modular AI classifiers, 
        VoxShield AI equips security analysts and digital forensics investigators with deep voice verification tools.
        """
    )

    st.markdown("""
    #### ⚙️ Forensic Pipeline Stages:
    1. **Format Validation & Ingestion**: Safely decodes multi-format audio (WAV, MP3, FLAC) in-memory without permanently retaining raw audio bytes.
    2. **Acoustic Conditioning**: Standardizes sampling to 16 kHz, converts channels to mono, normalizes peak amplitude, and applies spectral subtraction noise reduction.
    3. **Multi-View Feature Extraction**: Generates 128-band Mel-Spectrograms, 20-coefficient MFCCs, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, and Flatness.
    4. **Dual-Mode Detection Engine**: Strict separation between genuine **Model-Based Detection** (Wav2Vec2/Scikit-Learn) and transparent **Demo / Feature Analysis Mode**.
    5. **Auditable SQLite History**: Maintains forensic records, metadata, and extracted parameter vectors with CSV export capability.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Launch Audio Analysis Pipeline", use_container_width=True):
            st.switch_page("pages/2_Audio_Analysis.py")
    with c2:
        if st.button("📜 View Analysis History & Reports", use_container_width=True):
            st.switch_page("pages/4_Analysis_History.py")

with col_right:
    st.markdown("### 🔍 Engine Status & Integrity")
    status = st.session_state.detector.model_status

    if status.is_loaded:
        st.markdown(f"""
        <div class="result-banner real">
            <div>
                <div class="result-title" style="color: #10B981;">🟢 Model-Based Detection Active</div>
                <div class="result-subtitle">Model: <code>{status.model_name}</code></div>
                <div style="font-size: 0.82rem; margin-top: 6px; color: #CBD5E1;">
                    Architecture: {status.model_type} &bull; Device: {status.device}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner demo">
            <div>
                <div class="result-title" style="color: #00E5FF;">🔬 Demo / Feature Analysis Mode</div>
                <div class="result-subtitle">Academic Integrity Enforcement</div>
                <div style="font-size: 0.82rem; margin-top: 6px; color: #CBD5E1;">
                    No trained model weights currently detected in <code>models/</code>. 
                    Acoustic feature extraction is fully operational; machine classification is safely disabled.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📁 Storage & Engine Configuration", expanded=True):
        st.markdown(f"- **Database Path:** `{DB_PATH}`")
        st.markdown(f"- **Models Directory:** `{MODELS_DIR}`")
        st.markdown(f"- **Allowed Formats:** `.wav`, `.mp3`, `.flac`")
        st.markdown(f"- **Processing Target:** 16,000 Hz Mono")
        if st.button("🔄 Rescan Models Directory"):
            st.session_state.detector.reload_models()
            st.rerun()

st.markdown("---")

# Recent Activity Table
st.markdown("### 🕒 Recent Analysis Activity")
df_recent = get_records_dataframe()

if df_recent.empty:
    st.info("ℹ️ No audio analysis records found in the database. Counters start at zero. Upload an audio recording in the **Audio Analysis** page to generate genuine records.")
else:
    st.dataframe(
        df_recent.head(10),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Showing 10 most recent forensic entries. Visit **Analysis History** for search, filters, and CSV export.")
