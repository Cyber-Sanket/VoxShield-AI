"""
VoxShield AI - Visualizations Module
Modern cybersecurity-themed charts using Plotly and Matplotlib.
Ghost Protocol | AI & Cyber Security
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib

from config import THEME


def get_dark_plotly_layout(title: str = "", height: int = 380) -> Dict[str, Any]:
    """Base dark-cybersecurity layout template for Plotly charts."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Inter, sans-serif", size=16, color=THEME["primary"]),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor="#0A0F1D",
        margin=dict(l=50, r=30, t=50, b=45),
        height=height,
        font=dict(family="Inter, sans-serif", color=THEME["text_muted"], size=12),
        xaxis=dict(
            gridcolor=THEME["card_border"],
            zerolinecolor=THEME["card_border"],
            showgrid=True,
            linecolor=THEME["card_border"],
        ),
        yaxis=dict(
            gridcolor=THEME["card_border"],
            zerolinecolor=THEME["card_border"],
            showgrid=True,
            linecolor=THEME["card_border"],
        ),
        hoverlabel=dict(
            bgcolor=THEME["card_bg"],
            bordercolor=THEME["cyan"],
            font=dict(color=THEME["text_main"], family="Inter, sans-serif")
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor=THEME["card_border"],
            borderwidth=1,
            font=dict(color=THEME["text_main"])
        )
    )


def plot_waveform_comparison(
    orig_signal: np.ndarray,
    orig_sr: int,
    proc_signal: np.ndarray,
    proc_sr: int,
    title: str = "Audio Waveform: Original vs Preprocessed Signal"
) -> go.Figure:
    """
    Renders an interactive dual-panel Plotly comparison of the raw input audio
    versus the conditioned / noise-reduced signal.
    """
    t_orig = np.linspace(0, len(orig_signal) / orig_sr, len(orig_signal))
    t_proc = np.linspace(0, len(proc_signal) / proc_sr, len(proc_signal))

    # Downsample points for snappy browser rendering if over 40k samples
    step_orig = max(1, len(orig_signal) // 40000)
    step_proc = max(1, len(proc_signal) // 40000)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Raw Input Audio (Mono)", "Conditioned & Normalized Signal")
    )

    # Raw signal trace
    fig.add_trace(
        go.Scatter(
            x=t_orig[::step_orig],
            y=orig_signal[::step_orig],
            name="Raw Input",
            line=dict(color=THEME["blue"], width=1.2),
            hovertemplate="Time: %{x:.3f}s<br>Amp: %{y:.3f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Processed signal trace
    fig.add_trace(
        go.Scatter(
            x=t_proc[::step_proc],
            y=proc_signal[::step_proc],
            name="Processed",
            line=dict(color=THEME["primary"], width=1.2),
            hovertemplate="Time: %{x:.3f}s<br>Amp: %{y:.3f}<extra></extra>"
        ),
        row=2, col=1
    )

    layout = get_dark_plotly_layout(title, height=420)
    fig.update_layout(layout)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, range=[-1.05, 1.05])
    fig.update_yaxes(title_text="Amplitude", row=2, col=1, range=[-1.05, 1.05])
    
    # Custom styling for subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color=THEME["cyan"], family="Inter, sans-serif")

    return fig


def plot_mel_spectrogram(
    mel_spec_db: np.ndarray,
    times: np.ndarray,
    sr: int = 16000,
    title: str = "Log Mel-Spectrogram (Time-Frequency Energy Distribution)"
) -> go.Figure:
    """
    Renders an interactive 2D heatmap of the Mel-Spectrogram with decibel scale.
    """
    n_mels = mel_spec_db.shape[0]
    mel_bins = np.arange(n_mels)

    fig = go.Figure(
        data=go.Heatmap(
            z=mel_spec_db,
            x=times,
            y=mel_bins,
            colorscale="Inferno",
            colorbar=dict(
                title="Power (dB)",
                title_font=dict(color=THEME["text_muted"], size=11),
                tickfont=dict(color=THEME["text_muted"], size=10),
                len=0.85
            ),
            hovertemplate="Time: %{x:.2f}s<br>Mel Bin: %{y}<br>Power: %{z:.1f} dB<extra></extra>"
        )
    )

    layout = get_dark_plotly_layout(title, height=400)
    fig.update_layout(layout)
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Mel Frequency Bins (0 to 127)")

    return fig


def plot_mfcc_heatmap(
    mfcc: np.ndarray,
    times: np.ndarray,
    title: str = "Mel-Frequency Cepstral Coefficients (MFCC Matrix)"
) -> go.Figure:
    """
    Renders an interactive heatmap of the extracted MFCC coefficients across time.
    """
    n_mfcc = mfcc.shape[0]
    coeff_indices = [f"C{i+1}" for i in range(n_mfcc)]

    fig = go.Figure(
        data=go.Heatmap(
            z=mfcc,
            x=times,
            y=coeff_indices,
            colorscale="Viridis",
            colorbar=dict(
                title="Value",
                title_font=dict(color=THEME["text_muted"], size=11),
                tickfont=dict(color=THEME["text_muted"], size=10),
                len=0.85
            ),
            hovertemplate="Time: %{x:.2f}s<br>Coeff: %{y}<br>Value: %{z:.2f}<extra></extra>"
        )
    )

    layout = get_dark_plotly_layout(title, height=380)
    fig.update_layout(layout)
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Cepstral Coefficient")

    return fig


def plot_spectral_dynamics(
    times: np.ndarray,
    centroid: np.ndarray,
    bandwidth: np.ndarray,
    rolloff: np.ndarray,
    title: str = "Spectral Trajectory (Centroid, Bandwidth, Rolloff)"
) -> go.Figure:
    """
    Plots the three key frequency-domain dynamics together to inspect formant shifts and vocoder filtering.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=rolloff,
            name="Spectral Rolloff (85%)",
            line=dict(color=THEME["accent_purple"], width=1.5),
            hovertemplate="Time: %{x:.2f}s<br>Rolloff: %{y:.1f} Hz<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=centroid,
            name="Spectral Centroid",
            line=dict(color=THEME["cyan"], width=2.0),
            hovertemplate="Time: %{x:.2f}s<br>Centroid: %{y:.1f} Hz<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=bandwidth,
            name="Spectral Bandwidth",
            line=dict(color=THEME["primary"], width=1.5, dash="dot"),
            hovertemplate="Time: %{x:.2f}s<br>Bandwidth: %{y:.1f} Hz<extra></extra>"
        )
    )

    layout = get_dark_plotly_layout(title, height=400)
    fig.update_layout(layout)
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Frequency (Hz)")

    return fig


def plot_temporal_dynamics(
    times: np.ndarray,
    rms: np.ndarray,
    zcr: np.ndarray,
    title: str = "Temporal Energy Envelope (RMS) & Zero Crossing Rate (ZCR)"
) -> go.Figure:
    """
    Plots RMS energy alongside Zero Crossing Rate with dual axes to monitor micro-dynamics.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=times,
            y=rms,
            name="RMS Energy (Loudness)",
            line=dict(color=THEME["primary"], width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 163, 0.12)",
            hovertemplate="Time: %{x:.2f}s<br>RMS: %{y:.4f}<extra></extra>"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=zcr,
            name="Zero Crossing Rate",
            line=dict(color=THEME["cyan"], width=1.2, dash="dash"),
            hovertemplate="Time: %{x:.2f}s<br>ZCR: %{y:.4f}<extra></extra>"
        ),
        secondary_y=True
    )

    layout = get_dark_plotly_layout(title, height=380)
    fig.update_layout(layout)
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="RMS Amplitude", secondary_y=False, color=THEME["primary"])
    fig.update_yaxes(title_text="Zero Crossing Fraction", secondary_y=True, color=THEME["cyan"])

    return fig


def plot_feature_radar(
    summary_stats: Dict[str, Dict[str, float]],
    title: str = "Acoustic Fingerprint Summary (Normalized)"
) -> go.Figure:
    """
    Radar / spider chart summarizing key acoustic indicators normalized between 0 and 1.
    """
    categories = [
        "Centroid", "Bandwidth", "Rolloff",
        "ZCR", "RMS Energy", "Flatness"
    ]

    # Map keys
    keys = [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "zero_crossing_rate", "rms_energy", "spectral_flatness"
    ]

    # Reference normalizers based on typical speech distributions (at 16 kHz)
    normalizers = [4000.0, 3000.0, 6000.0, 0.3, 0.25, 0.15]

    values = []
    for k, norm in zip(keys, normalizers):
        val = summary_stats.get(k, {}).get("mean", 0.0)
        norm_val = min(1.0, max(0.0, val / norm))
        values.append(norm_val)

    # Close the radar loop
    radar_cats = categories + [categories[0]]
    radar_vals = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=radar_vals,
            theta=radar_cats,
            fill='toself',
            fillcolor="rgba(0, 229, 255, 0.20)",
            line=dict(color=THEME["cyan"], width=2),
            name="Acoustic Distribution"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickfont=dict(color=THEME["text_muted"], size=9),
                gridcolor=THEME["card_border"]
            ),
            angularaxis=dict(
                tickfont=dict(color=THEME["text_main"], size=11, family="Inter, sans-serif"),
                gridcolor=THEME["card_border"]
            ),
            bgcolor="#0A0F1D"
        ),
        paper_bgcolor=THEME["card_bg"],
        font=dict(family="Inter, sans-serif", color=THEME["text_muted"]),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="Inter, sans-serif", size=15, color=THEME["primary"]),
            x=0.01,
            y=0.96
        ),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False
    )

    return fig
