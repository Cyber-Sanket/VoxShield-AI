"""
VoxShield AI - Audio Preprocessing Pipeline
Signal conditioning: Mono conversion, resampling, normalization, and forensic noise reduction.
Ghost Protocol | AI & Cyber Security
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import scipy.signal

from config import TARGET_SAMPLE_RATE, MAX_AUDIO_DURATION_SECONDS
from modules.audio_loader import LoadedAudio


@dataclass
class PreprocessedAudio:
    original_signal: np.ndarray       # Original mono signal
    original_sr: int                  # Original sample rate
    processed_signal: np.ndarray      # Fully preprocessed signal (mono, resampled, normalized)
    sample_rate: int                  # Final sample rate (typically 16 kHz)
    duration_seconds: float
    noise_reduced_signal: Optional[np.ndarray] = None
    steps_applied: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def to_mono(signal: np.ndarray) -> np.ndarray:
    """
    Converts multi-channel audio to mono by averaging across channel axes.
    """
    if signal.ndim == 1:
        return signal.astype(np.float32)
    elif signal.ndim == 2:
        # Check if channels are in axis 0 or axis 1
        if signal.shape[1] < signal.shape[0]:
            return np.mean(signal, axis=1, dtype=np.float32)
        else:
            return np.mean(signal, axis=0, dtype=np.float32)
    else:
        flat = signal.reshape(-1, signal.shape[-1])
        return np.mean(flat, axis=-1, dtype=np.float32)


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Resamples audio signal to target sample rate using scipy or librosa.
    """
    if orig_sr == target_sr:
        return signal.astype(np.float32)

    try:
        import librosa
        resampled = librosa.resample(y=signal, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best")
        return resampled.astype(np.float32)
    except Exception:
        # High quality polyphase resampling fallback
        import math
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = scipy.signal.resample_poly(signal, up, down)
        return resampled.astype(np.float32)


def normalize_audio(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak normalizes audio signal to [-target_peak, target_peak] to standardize dynamic range.
    """
    max_val = np.max(np.abs(signal))
    if max_val > 1e-6:
        return (signal / max_val * target_peak).astype(np.float32)
    return signal.astype(np.float32)


def apply_spectral_subtraction(signal: np.ndarray, sr: int, reduction_strength: float = 0.65) -> np.ndarray:
    """
    Demonstration noise-reduction module using spectral magnitude subtraction.
    Estimates stationary background noise from quietest frames and attenuates noise floor.
    """
    n_fft = 1024
    hop_length = 256
    
    # Compute Short-Time Fourier Transform
    try:
        import librosa
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    except Exception:
        _, _, stft = scipy.signal.stft(signal, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise floor from the lowest 10% energy frames
    frame_energies = np.sum(magnitude ** 2, axis=0)
    quiet_threshold = np.percentile(frame_energies, 15)
    noise_frames = magnitude[:, frame_energies <= quiet_threshold]

    if noise_frames.shape[1] > 0:
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.mean(magnitude, axis=1, keepdims=True) * 0.1

    # Spectral subtraction with spectral floor to avoid musical noise artifacts
    subtracted_mag = magnitude - (reduction_strength * noise_profile)
    floor = 0.05 * magnitude
    cleaned_mag = np.maximum(subtracted_mag, floor)

    # Reconstruct time-domain signal
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    try:
        import librosa
        cleaned_signal = librosa.istft(cleaned_stft, hop_length=hop_length, length=len(signal))
    except Exception:
        _, cleaned_signal = scipy.signal.istft(cleaned_stft, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        if len(cleaned_signal) > len(signal):
            cleaned_signal = cleaned_signal[:len(signal)]
        elif len(cleaned_signal) < len(signal):
            cleaned_signal = np.pad(cleaned_signal, (0, len(signal) - len(cleaned_signal)))

    return cleaned_signal.astype(np.float32)


def preprocess_audio(
    loaded_audio: LoadedAudio,
    target_sr: int = TARGET_SAMPLE_RATE,
    apply_noise_reduction: bool = True,
    max_duration: float = MAX_AUDIO_DURATION_SECONDS
) -> PreprocessedAudio:
    """
    Full preprocessing pipeline:
    1. Mono conversion
    2. Duration clamping (if exceeds maximum for demo performance)
    3. Resampling to target sample rate (default 16 kHz)
    4. Peak amplitude normalization
    5. Spectral noise reduction demonstration
    """
    steps: List[str] = []
    
    # Step 1: Mono conversion
    mono_orig = to_mono(loaded_audio.raw_signal)
    if loaded_audio.metadata.channels > 1:
        steps.append(f"Channel Downmix: Converted {loaded_audio.metadata.channels} channels to mono")
    else:
        steps.append("Channel Check: Verified single-channel mono input")

    # Step 2: Duration truncation if necessary
    orig_sr = loaded_audio.sample_rate
    max_samples = int(max_duration * orig_sr)
    if len(mono_orig) > max_samples:
        mono_orig = mono_orig[:max_samples]
        steps.append(f"Duration Trim: Clamped to first {max_duration:.1f}s for high-resolution processing")

    # Step 3: Resampling
    if orig_sr != target_sr:
        resampled = resample_audio(mono_orig, orig_sr, target_sr)
        steps.append(f"Audio Resampling: Converted sample rate from {orig_sr} Hz to {target_sr} Hz")
    else:
        resampled = mono_orig.copy()
        steps.append(f"Sample Rate: Retained standard {target_sr} Hz")

    # Step 4: Amplitude Normalization
    normalized = normalize_audio(resampled, target_peak=0.95)
    steps.append("Dynamic Range Normalization: Peak scaled to [-0.95, +0.95]")

    # Step 5: Noise Reduction Demonstration
    noise_reduced = None
    if apply_noise_reduction:
        try:
            noise_reduced = apply_spectral_subtraction(normalized, target_sr, reduction_strength=0.60)
            noise_reduced = normalize_audio(noise_reduced, target_peak=0.95)
            steps.append("Acoustic Cleaning: Spectral subtraction applied to isolate voice harmonics")
        except Exception:
            noise_reduced = normalized.copy()
            steps.append("Acoustic Cleaning: Passthrough (noise reduction skipped)")

    final_signal = noise_reduced if (apply_noise_reduction and noise_reduced is not None) else normalized
    final_duration = len(final_signal) / target_sr

    stats = {
        "original_sr": orig_sr,
        "processed_sr": target_sr,
        "original_samples": len(mono_orig),
        "processed_samples": len(final_signal),
        "peak_amplitude": float(np.max(np.abs(final_signal))),
        "rms_amplitude": float(np.sqrt(np.mean(final_signal ** 2))),
        "duration_seconds": round(final_duration, 3)
    }

    return PreprocessedAudio(
        original_signal=mono_orig,
        original_sr=orig_sr,
        processed_signal=final_signal,
        sample_rate=target_sr,
        duration_seconds=round(final_duration, 3),
        noise_reduced_signal=noise_reduced,
        steps_applied=steps,
        stats=stats
    )
