"""
VoxShield AI - Detection Results & Forensic Audit Report Page
Verified classification outputs, forensic notes, model status, and limitations.
Ghost Protocol | AI & Cyber Security
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Detection Results | VoxShield AI",
    page_icon="🛡️",
    layout="wide"
)

from config import (
    LABEL_REAL, LABEL_FAKE, LABEL_INCONCLUSIVE, LABEL_UNAVAILABLE,
    MODE_MODEL, MODE_DEMO, TEAM_NAME, APP_NAME
)
from modules.visualizations import (
    plot_waveform_comparison, plot_mel_spectrogram, plot_mfcc_heatmap
)
from modules.ui_components import apply_custom_css, render_header, render_sidebar_info

apply_custom_css()
render_sidebar_info(st.session_state.get("detector", None).model_status if "detector" in st.session_state else None)
render_header("Detection Results & Cyber Forensic Evaluation")

if "current_analysis" not in st.session_state or st.session_state.current_analysis.get("result") is None:
    st.warning("⚠️ No active detection results found. Please upload or analyze an audio file in **Audio Analysis** first.")
    if st.button("🚀 Go to Audio Analysis"):
        st.switch_page("pages/2_Audio_Analysis.py")
    st.stop()

analysis_data = st.session_state.current_analysis
meta = analysis_data["metadata"]
prep = analysis_data["preprocessed"]
features = analysis_data["features"]
res = analysis_data["result"]
record_id = analysis_data.get("record_id", "N/A")

# 1. Executive Verdict Banner
st.markdown("### 🏛️ Executive Verdict & Forensic Status")

if res.mode == MODE_MODEL:
    if res.prediction == LABEL_REAL:
        banner_class = "real"
        title_color = "#10B981"
        badge_text = "VERIFIED REAL VOICE"
    elif res.prediction == LABEL_FAKE:
        banner_class = "fake"
        title_color = "#EF4444"
        badge_text = "SYNTHETIC DEEPFAKE DETECTED"
    else:
        banner_class = "demo"
        title_color = "#F59E0B"
        badge_text = "INCONCLUSIVE DETERMINATION"

    st.markdown(f"""
    <div class="result-banner {banner_class}">
        <div>
            <div class="result-title" style="color: {title_color};">
                [{badge_text}] {res.prediction.upper()}
            </div>
            <div class="result-subtitle">{res.summary_message}</div>
            <div style="font-size: 0.85rem; margin-top: 8px; color: #CBD5E1;">
                <b>Detection Mode:</b> Model-Based Detection &bull; 
                <b>Model Identifier:</b> <code>{res.model_status.model_name}</code> &bull; 
                <b>Inference Confidence:</b> {res.confidence * 100:.1f}% &bull; 
                <b>Audit ID:</b> <code>{record_id}</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Strict Academic Integrity Banner
    st.markdown(f"""
    <div class="result-banner demo">
        <div>
            <div class="result-title" style="color: #00E5FF;">
                🔬 {LABEL_UNAVAILABLE}
            </div>
            <div class="result-subtitle">{res.summary_message}</div>
            <div style="font-size: 0.85rem; margin-top: 8px; color: #CBD5E1;">
                <b>Detection Mode:</b> {res.mode} &bull; 
                <b>Model Status:</b> {res.model_status.model_name} &bull; 
                <b>Audit ID:</b> <code>{record_id}</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Forensic Metadata & Inspection Grid
col_f1, col_f2 = st.columns([1.2, 1.2])

with col_f1:
    st.markdown("#### 📁 File & Signal Attributes")
    st.markdown(f"""
    - **Filename:** `{meta.filename}`
    - **Duration:** `{meta.duration_seconds:.2f} seconds`
    - **Original Sample Rate:** `{meta.sample_rate:,} Hz`
    - **Processed Rate:** `{prep.sample_rate:,} Hz`
    - **Channels:** `{meta.channels}` (Downmixed to Mono)
    - **Container Format:** `{meta.file_format}`
    - **File Size:** `{meta.file_size_mb:.2f} MB`
    """)

with col_f2:
    st.markdown("#### 🧠 AI Model & Pipeline Metadata")
    st.markdown(f"""
    - **Active Engine Mode:** `{res.mode}`
    - **Model Architecture:** `{res.model_status.model_type}`
    - **Device Runtime:** `{res.model_status.device}`
    - **Acoustic Features Computed:** `52-dimensional vector`
    - **Evaluation Status:** `{'Genuine Model Inference' if res.mode == MODE_MODEL else 'Demo Acoustic Inspection'}`
    """)

st.markdown("---")

# 3. Spectral & Waveform Audit Visualizations
st.markdown("### 📉 Forensic Spectral Signatures")

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.plotly_chart(
        plot_mel_spectrogram(features.mel_spectrogram_db, features.times, sr=prep.sample_rate, title="Mel-Spectrogram (dB)"),
        use_container_width=True
    )

with col_v2:
    st.plotly_chart(
        plot_mfcc_heatmap(features.mfcc, features.times, title="MFCC Matrix (20 Coefficients)"),
        use_container_width=True
    )

st.plotly_chart(
    plot_waveform_comparison(prep.original_signal, prep.original_sr, prep.processed_signal, prep.sample_rate, title="Conditioned Waveform vs Input"),
    use_container_width=True
)

st.markdown("---")

# 4. Forensic Notes & Limitations
col_notes, col_limits = st.columns(2)

with col_notes:
    st.markdown("#### 📝 Forensic Observations")
    for note in res.forensic_notes:
        st.markdown(f"- {note}")

with col_limits:
    st.markdown("#### ⚠️ Operational Limitations & Academic Integrity")
    for lim in res.limitations:
        st.markdown(f"- {lim}")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Export Report
report_dict = {
    "system": APP_NAME,
    "team": TEAM_NAME,
    "audit_id": record_id,
    "filename": meta.filename,
    "duration_seconds": meta.duration_seconds,
    "sample_rate": prep.sample_rate,
    "detection_mode": res.mode,
    "prediction": res.prediction,
    "confidence": res.confidence,
    "model_name": res.model_status.model_name,
    "feature_summary": features.summary_stats
}

report_json = json.dumps(report_dict, indent=2)

st.download_button(
    label="📥 Download Forensic Audit Report (JSON)",
    data=report_json,
    file_name=f"voxshield_forensic_report_{record_id}.json",
    mime="application/json",
    use_container_width=True
)
