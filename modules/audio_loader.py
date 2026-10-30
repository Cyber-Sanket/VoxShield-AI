"""
VoxShield AI - Audio Loader Module
Robust audio file ingestion, format validation, metadata extraction, and in-memory safety.
Ghost Protocol | AI & Cyber Security
"""

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union, BinaryIO
import numpy as np

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_AUDIO_DURATION_SECONDS


@dataclass
class AudioMetadata:
    filename: str
    file_size_mb: float
    duration_seconds: float
    sample_rate: int
    channels: int
    file_format: str
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class LoadedAudio:
    raw_signal: np.ndarray          # Original multi-channel or mono audio array (float32)
    sample_rate: int                # Native sample rate
    metadata: AudioMetadata
    file_bytes: bytes               # Kept in memory for browser playback; never written to DB


class AudioLoadingError(Exception):
    """Raised when an uploaded audio file is invalid, corrupted, or unsupported."""
    pass


def validate_file_header(filename: str, file_bytes: bytes) -> str:
    """
    Validates file extension and basic magic bytes to prevent spoofed or corrupted uploads.
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioLoadingError(
            f"Unsupported file format '.{ext}'. Supported formats are: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}."
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise AudioLoadingError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB."
        )

    if len(file_bytes) < 44:  # Minimum valid header size for audio
        raise AudioLoadingError("File is empty or truncated. Header information could not be read.")

    # Validate signature
    if ext == "wav" and not (file_bytes.startswith(b"RIFF") or file_bytes.startswith(b"RIFX")):
        raise AudioLoadingError("Corrupted or invalid WAV file: missing valid RIFF header.")
    elif ext == "flac" and not file_bytes.startswith(b"fLaC"):
        raise AudioLoadingError("Corrupted or invalid FLAC file: missing standard FLAC header.")
    elif ext == "mp3":
        # Check standard ID3 or sync frame
        has_id3 = file_bytes.startswith(b"ID3")
        has_sync = file_bytes[0] == 0xFF and (file_bytes[1] & 0xE0) == 0xE0
        if not (has_id3 or has_sync):
            # Some MP3s might have slight leading junk, check within first 100 bytes
            found = False
            for i in range(min(128, len(file_bytes) - 1)):
                if file_bytes[i] == 0xFF and (file_bytes[i+1] & 0xE0) == 0xE0:
                    found = True
                    break
            if not found and not has_id3:
                raise AudioLoadingError("Corrupted or invalid MP3 file: missing valid MPEG audio frame sync.")

    return ext


def load_audio(
    source: Union[str, Path, BinaryIO, bytes],
    filename: Optional[str] = None
) -> LoadedAudio:
    """
    Loads and decodes audio from a file path, BytesIO, or raw bytes into a float32 numpy array.
    Extracts complete technical metadata without storing files to disk.
    """
    file_bytes: bytes
    name: str

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise AudioLoadingError(f"Audio file not found at path: {source}")
        name = filename or p.name
        with open(p, "rb") as f:
            file_bytes = f.read()
    elif isinstance(source, bytes):
        file_bytes = source
        name = filename or "uploaded_audio.wav"
    else:
        # Streamlit UploadedFile or BytesIO
        name = filename or getattr(source, "name", "uploaded_audio.wav")
        source.seek(0)
        file_bytes = source.read()

    ext = validate_file_header(name, file_bytes)
    size_mb = len(file_bytes) / (1024 * 1024)

    # Attempt decoding via soundfile (fast and memory-safe)
    signal: Optional[np.ndarray] = None
    sr: Optional[int] = None
    channels = 1

    try:
        import soundfile as sf
        with io.BytesIO(file_bytes) as bio:
            signal, sr = sf.read(bio, dtype="float32", always_2d=False)
            if signal.ndim > 1:
                channels = signal.shape[1]
            else:
                channels = 1
    except Exception as sf_err:
        # Fallback to librosa or scipy if soundfile encountered an odd container issue
        try:
            import librosa
            with io.BytesIO(file_bytes) as bio:
                y, sr = librosa.load(bio, sr=None, mono=False)
                if y.ndim > 1:
                    channels = y.shape[0]
                    # Transpose to (samples, channels) standard
                    signal = y.T.astype(np.float32)
                else:
                    signal = y.astype(np.float32)
                    channels = 1
        except Exception as lb_err:
            raise AudioLoadingError(
                f"Failed to decode audio file. The file may be damaged or encoded with an unsupported codec. Details: {sf_err}"
            )

    if signal is None or len(signal) == 0:
        raise AudioLoadingError("Audio file contains zero readable samples.")

    # Calculate duration
    num_samples = signal.shape[0] if signal.ndim > 1 else len(signal)
    duration = float(num_samples) / float(sr)

    if duration < 0.2:
        raise AudioLoadingError(f"Audio duration is too short ({duration:.2f}s). Minimum required is 0.2 seconds.")

    metadata = AudioMetadata(
        filename=name,
        file_size_mb=round(size_mb, 3),
        duration_seconds=round(duration, 3),
        sample_rate=int(sr),
        channels=channels,
        file_format=ext.upper(),
        is_valid=True
    )

    return LoadedAudio(
        raw_signal=signal,
        sample_rate=int(sr),
        metadata=metadata,
        file_bytes=file_bytes
    )
