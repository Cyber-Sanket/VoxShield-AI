"""
VoxShield AI - Feature Extraction & Acoustic Forensics Page
Deep dive into Mel-Spectrogram, MFCCs, Spectral Centroid, Bandwidth, Rolloff, ZCR, and RMS Energy.
Ghost Protocol | AI & Cyber Security
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Feature Extraction | VoxShield AI",
    page_icon="🛡️",
    layout="wide"
)

from config import SAMPLES_DIR, TARGET_SAMPLE_RATE
from modules.feature_extraction import FEATURE_EXPLANATIONS, extract_audio_features
from modules.audio_loader import load_audio
from modules.preprocessing import preprocess_audio
from modules.visualizations import (
    plot_mel_spectrogram, plot_mfcc_heatmap,
    plot_spectral_dynamics, plot_temporal_dynamics,
    plot_feature_radar
)
from modules.ui_components import apply_custom_css, render_header, render_sidebar_info

apply_custom_css()
render_sidebar_info(st.session_state.get("detector", None).model_status if "detector" in st.session_state else None)
render_header("Acoustic Feature Extraction & Spectral Forensics")

# Ensure an active analysis is available or provide sample loader
if "current_analysis" not in st.session_state:
    st.info("ℹ️ No audio is currently loaded in the active session. You can analyze an audio sample below or go to **Audio Analysis**.")
    sample_files = list(SAMPLES_DIR.glob("*.wav"))
    if sample_files:
        sample_names = [s.name for s in sample_files]
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            chosen = st.selectbox("Select Benchmark Sample to Inspect:", sample_names)
        with col_s2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("⚡ Extract & Visualize Sample Features", type="primary", use_container_width=True):
                with st.spinner("Processing benchmark audio..."):
                    loaded = load_audio(SAMPLES_DIR / chosen)
                    prep = preprocess_audio(loaded, target_sr=TARGET_SAMPLE_RATE)
                    feats = extract_audio_features(prep.processed_signal, prep.sample_rate)
                    st.session_state.current_analysis = {
                        "record_id": "PREVIEW",
                        "metadata": loaded.metadata,
                        "preprocessed": prep,
                        "features": feats,
                        "result": None,
                        "audio_bytes": loaded.file_bytes
                    }
                st.rerun()
    st.stop()

# Retrieve active analysis data
analysis_data = st.session_state.current_analysis
meta = analysis_data["metadata"]
prep = analysis_data["preprocessed"]
features = analysis_data["features"]

st.markdown(f"### 🎵 Active Target: `{meta.filename}` ({meta.duration_seconds:.2f}s, {prep.sample_rate} Hz)")

# Acoustic Radar & Numerical Summary
col_radar, col_table = st.columns([1.1, 1.4])

with col_radar:
    st.markdown("#### 🕸️ Acoustic Distribution Radar")
    radar_fig = plot_feature_radar(features.summary_stats, title=f"Acoustic Fingerprint: {meta.filename}")
    st.plotly_chart(radar_fig, use_container_width=True)

with col_table:
    st.markdown("#### 📋 Statistical Summary Table")
    stat_rows = []
    labels_map = {
        "spectral_centroid": "Spectral Centroid (Hz)",
        "spectral_bandwidth": "Spectral Bandwidth (Hz)",
        "spectral_rolloff": "Spectral Rolloff (Hz)",
        "zero_crossing_rate": "Zero Crossing Rate (ZCR)",
        "rms_energy": "RMS Energy (Loudness)",
        "spectral_flatness": "Spectral Flatness Index",
    }

    for key, label in labels_map.items():
        if key in features.summary_stats:
            s = features.summary_stats[key]
            stat_rows.append({
                "Feature Metric": label,
                "Mean": f"{s['mean']:.3f}",
                "Std Dev": f"{s['std']:.3f}",
                "Min": f"{s['min']:.3f}",
                "Max": f"{s['max']:.3f}",
                "Median": f"{s['median']:.3f}"
            })

    df_stats = pd.DataFrame(stat_rows)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

st.markdown("---")

# Feature Deep Dives with Visualizations & Forensic Interpretations
st.markdown("### 🔬 In-Depth Spectral & Cepstral Visualizations")

tab_mel, tab_mfcc, tab_spectral, tab_temporal = st.tabs([
    "🌈 Mel-Spectrogram (Time-Frequency)",
    "🎛️ MFCC Cepstral Matrix",
    "📈 Spectral Trajectories (Centroid / Rolloff)",
    "⏱️ Temporal Dynamics (RMS / ZCR)"
])

with tab_mel:
    st.plotly_chart(
        plot_mel_spectrogram(features.mel_spectrogram_db, features.times, sr=prep.sample_rate),
        use_container_width=True
    )
    exp = FEATURE_EXPLANATIONS["mel_spectrogram"]
    st.markdown(f"""
    <div class="feature-info-card">
        <div class="feature-info-title">💡 Forensic Relevance: {exp['title']}</div>
        <div class="feature-info-body">
            <b>Signal Characteristic:</b> {exp['role']}<br>
            <b>Deepfake Detection Utility:</b> {exp['forensic_relevance']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab_mfcc:
    st.plotly_chart(
        plot_mfcc_heatmap(features.mfcc, features.times),
        use_container_width=True
    )
    exp = FEATURE_EXPLANATIONS["mfcc"]
    st.markdown(f"""
    <div class="feature-info-card">
        <div class="feature-info-title">💡 Forensic Relevance: {exp['title']}</div>
        <div class="feature-info-body">
            <b>Signal Characteristic:</b> {exp['role']}<br>
            <b>Deepfake Detection Utility:</b> {exp['forensic_relevance']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab_spectral:
    st.plotly_chart(
        plot_spectral_dynamics(
            features.times,
            features.spectral_centroid,
            features.spectral_bandwidth,
            features.spectral_rolloff
        ),
        use_container_width=True
    )
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        exp_c = FEATURE_EXPLANATIONS["spectral_centroid"]
        st.markdown(f"""
        <div class="feature-info-card">
            <div class="feature-info-title">💡 {exp_c['title']}</div>
            <div class="feature-info-body">
                <b>Characteristic:</b> {exp_c['role']}<br>
                <b>Forensics:</b> {exp_c['forensic_relevance']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_e2:
        exp_r = FEATURE_EXPLANATIONS["spectral_rolloff"]
        st.markdown(f"""
        <div class="feature-info-card">
            <div class="feature-info-title">💡 {exp_r['title']}</div>
            <div class="feature-info-body">
                <b>Characteristic:</b> {exp_r['role']}<br>
                <b>Forensics:</b> {exp_r['forensic_relevance']}
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_temporal:
    st.plotly_chart(
        plot_temporal_dynamics(
            features.times,
            features.rms_energy,
            features.zero_crossing_rate
        ),
        use_container_width=True
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        exp_rms = FEATURE_EXPLANATIONS["rms_energy"]
        st.markdown(f"""
        <div class="feature-info-card">
            <div class="feature-info-title">💡 {exp_rms['title']}</div>
            <div class="feature-info-body">
                <b>Characteristic:</b> {exp_rms['role']}<br>
                <b>Forensics:</b> {exp_rms['forensic_relevance']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        exp_z = FEATURE_EXPLANATIONS["zero_crossing_rate"]
        st.markdown(f"""
        <div class="feature-info-card">
            <div class="feature-info-title">💡 {exp_z['title']}</div>
            <div class="feature-info-body">
                <b>Characteristic:</b> {exp_z['role']}<br>
                <b>Forensics:</b> {exp_z['forensic_relevance']}
            </div>
        </div>
        """, unsafe_allow_html=True)
