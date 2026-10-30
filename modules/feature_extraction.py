"""
VoxShield AI - Audio Feature Extraction Module
Comprehensive acoustic feature extraction for audio deepfake and synthetic voice forensics:
MFCCs, Mel-Spectrogram, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, and Flatness.
Ghost Protocol | AI & Cyber Security
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from config import (
    N_MFCC, N_FFT, HOP_LENGTH, N_MELS, TOP_DB, TARGET_SAMPLE_RATE
)


# Forensic interpretations explaining how each acoustic feature aids synthetic voice detection
FEATURE_EXPLANATIONS = {
    "mfcc": {
        "title": "Mel-Frequency Cepstral Coefficients (MFCC)",
        "role": "Captures the envelope of the vocal tract and spectral timbre.",
        "forensic_relevance": (
            "Human vocal tracts produce continuous, natural resonances (formants). "
            "Neural vocoders (such as HiFi-GAN or WaveGlow) often leave subtle mathematical artifacts "
            "and unnatural discontinuities across higher-order cepstral coefficients (MFCC 12-20)."
        )
    },
    "mel_spectrogram": {
        "title": "Mel-Spectrogram (Log Mel Power Spectrum)",
        "role": "Visualizes time-frequency energy distribution scaled to human pitch perception.",
        "forensic_relevance": (
            "Synthetic audio frequently displays visible checkerboard patterns, unnatural high-frequency cutoffs, "
            "harmonic smearing, or missing breath pauses in the 4 kHz - 8 kHz frequency bands."
        )
    },
    "spectral_centroid": {
        "title": "Spectral Centroid",
        "role": "Measures the center of mass ('brightness') of the sound spectrum.",
        "forensic_relevance": (
            "Deepfake voices often exhibit either abnormally depressed spectral centroids due to vocoder "
            "low-pass filtering, or erratic centroid spikes caused by synthetic high-frequency noise."
        )
    },
    "spectral_bandwidth": {
        "title": "Spectral Bandwidth",
        "role": "Measures the spread and dispersion of frequencies around the spectral centroid.",
        "forensic_relevance": (
            "Natural human speech displays wide, dynamic bandwidth variations during transitions between "
            "vowels and unvoiced consonants. Synthetic audio algorithms often generate overly rigid bandwidth ranges."
        )
    },
    "spectral_rolloff": {
        "title": "Spectral Rolloff (85% Energy Cutoff)",
        "role": "Frequency below which 85% of total signal energy resides.",
        "forensic_relevance": (
            "Many text-to-speech (TTS) and voice cloning architectures downsample internal representations, "
            "creating a telltale unnatural drop-off in high-frequency spectral rolloff compared to studio or phone recordings."
        )
    },
    "zero_crossing_rate": {
        "title": "Zero Crossing Rate (ZCR)",
        "role": "The rate at which the signal changes sign from positive to negative.",
        "forensic_relevance": (
            "Differentiates voiced and unvoiced speech segments. Synthetic audio models can struggle "
            "with natural fricatives ('s', 'f', 'sh'), leading to abnormal unvoiced ZCR distributions."
        )
    },
    "rms_energy": {
        "title": "Root Mean Square (RMS) Energy",
        "role": "Quantifies instantaneous loudness, dynamics, and temporal energy envelope.",
        "forensic_relevance": (
            "Human speakers introduce subtle micro-dynamics, breathing sounds, and natural decaying silences. "
            "Synthesized speech often shows overly uniform frame energy or robotic transitions between syllables."
        )
    },
    "spectral_flatness": {
        "title": "Spectral Flatness",
        "role": "Measures whether the spectrum is tone-like (low) or noise-like (high).",
        "forensic_relevance": (
            "Neural speech generators may suffer from buzziness or phase inconsistencies, "
            "altering the natural tonal vs. noise equilibrium of genuine speech."
        )
    }
}


@dataclass
class ExtractedFeatures:
    # 2D Spectrograms and Cepstral matrices
    mfcc: np.ndarray                  # Shape: (n_mfcc, n_frames)
    mfcc_delta: np.ndarray            # Shape: (n_mfcc, n_frames)
    mel_spectrogram_db: np.ndarray    # Shape: (n_mels, n_frames)
    
    # 1D Temporal Feature Series (Shape: (n_frames,))
    times: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    spectral_flatness: np.ndarray
    
    # Numerical Statistics Summary for DB and Reports
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    flat_feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


def compute_statistics(arr: np.ndarray) -> Dict[str, float]:
    """Computes mean, std, min, max, median of a 1D array."""
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }


def extract_audio_features(
    signal: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS
) -> ExtractedFeatures:
    """
    Extracts all required acoustic features from preprocessed audio signal.
    """
    import librosa

    # Ensure 1D float32
    y = signal.astype(np.float32)

    # 1. Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=TOP_DB)

    # 2. MFCCs and Delta
    mfcc = librosa.feature.mfcc(
        S=mel_spec_db, sr=sr, n_mfcc=n_mfcc
    )
    mfcc_delta = librosa.feature.delta(mfcc)

    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # 4. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # 5. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=n_fft, hop_length=hop_length
    )[0]

    # 7. RMS Energy
    rms = librosa.feature.rms(
        y=y, frame_length=n_fft, hop_length=hop_length
    )[0]

    # 8. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Time axis corresponding to frames
    n_frames = mel_spec_db.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # Harmonize lengths across 1D series
    min_len = min(
        len(times), len(spectral_centroid), len(spectral_bandwidth),
        len(spectral_rolloff), len(zcr), len(rms), len(flatness)
    )
    times = times[:min_len]
    spectral_centroid = spectral_centroid[:min_len]
    spectral_bandwidth = spectral_bandwidth[:min_len]
    spectral_rolloff = spectral_rolloff[:min_len]
    zcr = zcr[:min_len]
    rms = rms[:min_len]
    flatness = flatness[:min_len]
    mel_spec_db = mel_spec_db[:, :min_len]
    mfcc = mfcc[:, :min_len]
    mfcc_delta = mfcc_delta[:, :min_len]

    # Aggregate statistical summaries
    summary_stats = {
        "spectral_centroid": compute_statistics(spectral_centroid),
        "spectral_bandwidth": compute_statistics(spectral_bandwidth),
        "spectral_rolloff": compute_statistics(spectral_rolloff),
        "zero_crossing_rate": compute_statistics(zcr),
        "rms_energy": compute_statistics(rms),
        "spectral_flatness": compute_statistics(flatness),
        "mfcc_mean_0_4": compute_statistics(np.mean(mfcc[:4, :], axis=0)),
        "mfcc_mean_high": compute_statistics(np.mean(mfcc[12:, :], axis=0)),
    }

    # Construct a compact flat feature vector suitable for baseline ML models
    flat_vector_list = []
    # 20 MFCC means & stds
    flat_vector_list.extend(np.mean(mfcc, axis=1))
    flat_vector_list.extend(np.std(mfcc, axis=1))
    # 1D feature stats
    for series in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms, flatness]:
        flat_vector_list.extend([
            float(np.mean(series)),
            float(np.std(series)),
            float(np.min(series)),
            float(np.max(series))
        ])

    flat_vector = np.array(flat_vector_list, dtype=np.float32)

    return ExtractedFeatures(
        mfcc=mfcc,
        mfcc_delta=mfcc_delta,
        mel_spectrogram_db=mel_spec_db,
        times=times,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        zero_crossing_rate=zcr,
        rms_energy=rms,
        spectral_flatness=flatness,
        summary_stats=summary_stats,
        flat_feature_vector=flat_vector
    )
