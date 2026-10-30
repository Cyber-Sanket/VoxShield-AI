"""
Unit tests for VoxShield AI - Audio Loader Module
"""

import pytest
import io
import wave
import numpy as np
from pathlib import Path

from modules.audio_loader import load_audio, validate_file_header, AudioLoadingError
from config import SAMPLES_DIR


@pytest.fixture
def sample_wav_bytes():
    """Generates a valid 16-bit PCM WAV byte stream for testing."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    int16_data = (signal * 32767).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())
    return bio.getvalue()


def test_load_valid_wav(sample_wav_bytes):
    """Verifies that a well-formed WAV file loads with correct metadata."""
    loaded = load_audio(sample_wav_bytes, filename="test_audio.wav")
    assert loaded.raw_signal is not None
    assert loaded.sample_rate == 16000
    assert loaded.metadata.channels == 1
    assert loaded.metadata.file_format == "WAV"
    assert pytest.approx(loaded.metadata.duration_seconds, rel=1e-2) == 1.0


def test_reject_unsupported_extension():
    """Ensures unsupported formats raise AudioLoadingError."""
    fake_bytes = b"RIFF" + b"\x00" * 100
    with pytest.raises(AudioLoadingError, match="Unsupported file format"):
        load_audio(fake_bytes, filename="document.pdf")


def test_reject_truncated_or_empty_bytes():
    """Ensures truncated files under 44 bytes raise AudioLoadingError."""
    with pytest.raises(AudioLoadingError, match="empty or truncated"):
        load_audio(b"RIFFshort", filename="corrupted.wav")


def test_reject_invalid_wav_header():
    """Ensures file with .wav extension but invalid header bytes is rejected."""
    corrupted_bytes = b"NOT_A_WAV_HEADER" + b"\x00" * 100
    with pytest.raises(AudioLoadingError, match="missing valid RIFF header"):
        load_audio(corrupted_bytes, filename="fake.wav")


def test_reject_too_short_audio():
    """Ensures audio under 0.2s duration is rejected."""
    sr = 16000
    duration = 0.05
    int16_data = np.zeros(int(sr * duration), dtype=np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())
    
    with pytest.raises(AudioLoadingError, match="too short"):
        load_audio(bio.getvalue(), filename="too_short.wav")
