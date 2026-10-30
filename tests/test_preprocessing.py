"""
Unit tests for VoxShield AI - Audio Preprocessing Pipeline
"""

import pytest
import numpy as np

from modules.preprocessing import (
    to_mono, resample_audio, normalize_audio,
    apply_spectral_subtraction, preprocess_audio
)
from modules.audio_loader import LoadedAudio, AudioMetadata


def test_to_mono_conversion():
    """Verifies stereo signal downmixing to 1D mono."""
    stereo = np.ones((16000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = 0.9
    mono = to_mono(stereo)

    assert mono.ndim == 1
    assert len(mono) == 16000
    assert pytest.approx(mono[0], abs=1e-4) == 0.7


def test_normalize_audio_peak():
    """Verifies peak scaling to target limit."""
    signal = np.array([-2.5, 0.0, 5.0, 1.2], dtype=np.float32)
    norm = normalize_audio(signal, target_peak=0.95)

    assert pytest.approx(np.max(np.abs(norm)), abs=1e-4) == 0.95
    assert np.all(np.abs(norm) <= 0.95)


def test_resample_audio():
    """Verifies audio resampling between different sampling rates."""
    sr_orig = 22050
    sr_target = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 300 * t).astype(np.float32)

    resampled = resample_audio(signal, sr_orig, sr_target)
    assert len(resampled) == int(sr_target * duration)


def test_full_preprocessing_pipeline():
    """Tests end-to-end preprocessing pipeline on mock LoadedAudio."""
    sr = 22050
    duration = 2.0
    signal = (np.random.rand(int(sr * duration), 2) * 2 - 1).astype(np.float32)

    meta = AudioMetadata(
        filename="mock.wav",
        file_size_mb=0.1,
        duration_seconds=duration,
        sample_rate=sr,
        channels=2,
        file_format="WAV",
        is_valid=True
    )
    loaded = LoadedAudio(raw_signal=signal, sample_rate=sr, metadata=meta, file_bytes=b"")

    prep = preprocess_audio(loaded, target_sr=16000, apply_noise_reduction=True)

    assert prep.sample_rate == 16000
    assert prep.processed_signal.ndim == 1
    assert len(prep.steps_applied) >= 3
    assert np.max(np.abs(prep.processed_signal)) <= 1.0
