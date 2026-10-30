"""
Unit tests for VoxShield AI - Feature Extraction Module
"""

import pytest
import numpy as np

from modules.feature_extraction import (
    extract_audio_features, compute_statistics,
    ExtractedFeatures, FEATURE_EXPLANATIONS
)


def test_feature_explanations_exist():
    """Verifies that all required feature explanations are documented."""
    required = [
        "mfcc", "mel_spectrogram", "spectral_centroid", "spectral_bandwidth",
        "spectral_rolloff", "zero_crossing_rate", "rms_energy", "spectral_flatness"
    ]
    for key in required:
        assert key in FEATURE_EXPLANATIONS
        assert "title" in FEATURE_EXPLANATIONS[key]
        assert "forensic_relevance" in FEATURE_EXPLANATIONS[key]


def test_compute_statistics():
    """Tests summary statistics calculation."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    stats = compute_statistics(arr)
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["median"] == 3.0


def test_extract_audio_features_shapes():
    """Verifies output shapes and feature vectors for extracted acoustic descriptors."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Harmonics
    signal = (0.5 * np.sin(2 * np.pi * 400 * t) + 0.25 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

    features = extract_audio_features(signal, sr=sr, n_mfcc=20, n_mels=128)

    assert isinstance(features, ExtractedFeatures)
    assert features.mfcc.shape[0] == 20
    assert features.mel_spectrogram_db.shape[0] == 128
    assert len(features.spectral_centroid) > 0
    assert len(features.spectral_bandwidth) == len(features.spectral_centroid)
    assert len(features.spectral_rolloff) == len(features.spectral_centroid)
    assert len(features.zero_crossing_rate) == len(features.spectral_centroid)
    assert len(features.rms_energy) == len(features.spectral_centroid)
    assert len(features.flat_feature_vector) > 40
    assert "spectral_centroid" in features.summary_stats
