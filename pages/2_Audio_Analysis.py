"""
VoxShield AI - Audio Analysis & Forensic Ingestion Page
Audio Upload, Signal Conditioning, Feature Processing, and Deepfake Detection Pipeline.
Ghost Protocol | AI & Cyber Security
"""

import io
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Audio Analysis | VoxShield AI",
    page_icon="🛡️",
    layout="wide"
)

from config import (
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, TARGET_SAMPLE_RATE,
    SAMPLES_DIR, LABEL_REAL, LABEL_FAKE, LABEL_INCONCLUSIVE, LABEL_UNAVAILABLE
)
from modules.database import save_analysis_record
from modules.audio_loader import load_audio, AudioLoadingError
from modules.preprocessing import preprocess_audio
from modules.feature_extraction import extract_audio_features
from modules.deepfake_detector import DeepfakeDetectionPipeline
from modules.visualizations import plot_waveform_comparison
from modules.ui_components import apply_custom_css, render_header, render_sidebar_info

if "detector" not in st.session_state:
    st.session_state.detector = DeepfakeDetectionPipeline()

apply_custom_css()
render_sidebar_info(st.session_state.detector.model_status)
render_header("Audio Upload, Forensic Conditioning & Detection Pipeline")

st.markdown("### 🎙️ 1. Audio Source Selection")
st.markdown("Select a built-in forensic sample or upload an external audio recording (`WAV`, `MP3`, or `FLAC`).")

source_tab1, source_tab2 = st.tabs(["📁 Upload Audio File", "🧪 Select Built-in Forensic Sample"])

audio_bytes = None
audio_filename = None

with source_tab1:
    uploaded_file = st.file_uploader(
        "Upload audio recording for forensic inspection",
        type=list(ALLOWED_EXTENSIONS),
        help=f"Maximum file size allowed: {MAX_FILE_SIZE_MB} MB. Audio is processed strictly in-memory."
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        audio_filename = uploaded_file.name

with source_tab2:
    sample_files = list(SAMPLES_DIR.glob("*.wav"))
    if sample_files:
        sample_names = [f.name for f in sample_files]
        chosen_sample = st.selectbox(
            "Choose a pre-synthesized acoustic benchmark audio:",
            sample_names,
            help="Pre-generated acoustic benchmarks with calibrated vocal tract harmonics or vocoder signatures."
        )
        if st.button("Load Selected Sample"):
            sample_path = SAMPLES_DIR / chosen_sample
            with open(sample_path, "rb") as f:
                audio_bytes = f.read()
            audio_filename = chosen_sample
            st.session_state["loaded_sample_name"] = chosen_sample
            st.session_state["loaded_sample_bytes"] = audio_bytes

    if "loaded_sample_bytes" in st.session_state and audio_bytes is None:
        audio_bytes = st.session_state["loaded_sample_bytes"]
        audio_filename = st.session_state.get("loaded_sample_name", "sample.wav")

# Audio validation & playback section
if audio_bytes and audio_filename:
    st.markdown("---")
    st.markdown("### 🔍 2. Audio Metadata & In-Memory Inspection")

    try:
        loaded_audio = load_audio(audio_bytes, filename=audio_filename)
        meta = loaded_audio.metadata

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Filename", meta.filename)
        with col_m2:
            st.metric("Duration", f"{meta.duration_seconds:.2f} s")
        with col_m3:
            st.metric("Sample Rate", f"{meta.sample_rate:,} Hz")
        with col_m4:
            st.metric("Channels", "Mono (1)" if meta.channels == 1 else f"Stereo ({meta.channels})")
        with col_m5:
            st.metric("Format / Size", f"{meta.file_format} ({meta.file_size_mb:.2f} MB)")

        st.markdown("**Browser Audio Playback:**")
        st.audio(audio_bytes, format=f"audio/{meta.file_format.lower()}")

        st.markdown("---")
        st.markdown("### ⚙️ 3. Preprocessing Configuration")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            apply_noise_reduction = st.checkbox(
                "Apply Spectral Subtraction (Noise Reduction)",
                value=True,
                help="Suppresses stationary acoustic background noise to isolate primary voice formants."
            )
        with c_p2:
            normalize_peak = st.checkbox(
                "Dynamic Range Peak Normalization (0.95)",
                value=True,
                help="Scales waveform amplitude to prevent digital clipping."
            )

        # Pipeline Execution Button
        if st.button("🚀 Run VoxShield Analysis Pipeline", use_container_width=True, type="primary"):
            with st.status("Executing VoxShield Cyber Forensics Pipeline...", expanded=True) as status_box:
                st.write("1. 🔄 Standardizing sample rate to 16,000 Hz and converting to mono...")
                preprocessed = preprocess_audio(
                    loaded_audio,
                    target_sr=TARGET_SAMPLE_RATE,
                    apply_noise_reduction=apply_noise_reduction
                )

                st.write("2. 📊 Extracting multi-band acoustic features (MFCCs, Mel-Spectrogram, Centroid, Rolloff, ZCR, RMS)...")
                features = extract_audio_features(
                    preprocessed.processed_signal,
                    sr=preprocessed.sample_rate
                )

                st.write("3. 🛡️ Querying Deepfake Detection Engine...")
                detector: DeepfakeDetectionPipeline = st.session_state.detector
                detection_result = detector.analyze(
                    features=features,
                    signal=preprocessed.processed_signal,
                    sr=preprocessed.sample_rate
                )

                st.write("4. 💾 Storing auditable record in SQLite database...")
                record_id = save_analysis_record(
                    filename=meta.filename,
                    duration=preprocessed.duration_seconds,
                    detection_mode=detection_result.mode,
                    prediction=detection_result.prediction if detection_result.prediction != LABEL_UNAVAILABLE else None,
                    confidence=detection_result.confidence,
                    model_name=detection_result.model_status.model_name,
                    feature_summary=features.summary_stats,
                    sample_rate=preprocessed.sample_rate,
                    channels=1
                )

                status_box.update(label="✅ Analysis Pipeline Completed Successfully!", state="complete", expanded=False)

            # Store in session state for downstream pages
            st.session_state.current_analysis = {
                "record_id": record_id,
                "metadata": meta,
                "preprocessed": preprocessed,
                "features": features,
                "result": detection_result,
                "audio_bytes": audio_bytes
            }

            st.rerun()

    except AudioLoadingError as e:
        st.error(f"❌ **Audio File Error**: {e}")
    except Exception as err:
        st.error(f"❌ **Unexpected Error during pipeline execution**: {err}")

# Display Active Analysis Results if Available
if "current_analysis" in st.session_state:
    data = st.session_state.current_analysis
    res = data["result"]
    prep = data["preprocessed"]
    meta = data["metadata"]

    st.markdown("---")
    st.markdown("### 📊 4. Analysis Results Summary")

    # Mode Banner
    if res.mode == "Model-Based Detection":
        if res.prediction == LABEL_REAL:
            banner_class = "real"
            title_color = "#10B981"
            icon = "✅"
        elif res.prediction == LABEL_FAKE:
            banner_class = "fake"
            title_color = "#EF4444"
            icon = "⚠️"
        else:
            banner_class = "demo"
            title_color = "#F59E0B"
            icon = "⚖️"

        st.markdown(f"""
        <div class="result-banner {banner_class}">
            <div>
                <div class="result-title" style="color: {title_color};">{icon} Model Prediction: {res.prediction.upper()}</div>
                <div class="result-subtitle">{res.summary_message}</div>
                <div style="font-size: 0.85rem; margin-top: 6px; color: #CBD5E1;">
                    Confidence Score: <b>{(res.confidence or 0)*100:.1f}%</b> &bull; Model: <code>{res.model_status.model_name}</code>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-banner demo">
            <div>
                <div class="result-title" style="color: #00E5FF;">🔬 {LABEL_UNAVAILABLE}</div>
                <div class="result-subtitle">{res.summary_message}</div>
                <div style="font-size: 0.82rem; margin-top: 8px; color: #94A3B8;">
                    Mode: <b>{res.mode}</b> &bull; No trained model weights found in <code>models/</code>. 
                    Acoustic features extracted below can be evaluated on the <b>Feature Extraction</b> page.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Dual Waveform Visualization
    st.markdown("#### 🌊 Audio Waveform Conditioning")
    fig_wave = plot_waveform_comparison(
        orig_signal=prep.original_signal,
        orig_sr=prep.original_sr,
        proc_signal=prep.processed_signal,
        proc_sr=prep.sample_rate,
        title=f"Waveform Comparison: {meta.filename}"
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    # Preprocessing log
    with st.expander("📝 View Applied Preprocessing Steps"):
        for step in prep.steps_applied:
            st.markdown(f"- `{step}`")

    # Navigation buttons to deep dive
    st.markdown("<br>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("📊 Inspect Extracted Acoustic Features & Spectrograms", use_container_width=True):
            st.switch_page("pages/3_Feature_Extraction.py")
    with col_nav2:
        if st.button("📑 View Detailed Detection Forensic Report", use_container_width=True):
            st.switch_page("pages/4_Detection_Results.py")
