"""
VoxShield AI - Sample Audio Generator
Generates clean, synthetic audio benchmarks for out-of-the-box demonstration and unit testing.
Does not require external datasets.
Ghost Protocol | AI & Cyber Security
"""

import math
import struct
import wave
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "sample_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_natural_vocal_sample(filename: str = "sample_real_voice.wav", duration: float = 3.5, sr: int = 16000):
    """
    Synthesizes a realistic human vowel/voice phrase with natural pitch jitter,
    formant resonance (F1=600Hz, F2=1300Hz, F3=2500Hz), breath noise, and dynamic envelope.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Fundamental frequency (F0) with natural pitch drift (vibrato 5.5 Hz)
    f0 = 135.0 + 3.0 * np.sin(2 * np.pi * 5.2 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Glottal pulse train with rich harmonics
    glottal = (
        1.00 * np.sin(phase) +
        0.55 * np.sin(2 * phase) +
        0.35 * np.sin(3 * phase) +
        0.20 * np.sin(4 * phase) +
        0.12 * np.sin(5 * phase) +
        0.08 * np.sin(6 * phase)
    )

    # Formant filter resonances (simulating vocal tract vowels /a/ to /o/)
    formant1 = np.sin(2 * np.pi * 650.0 * t) * 0.4
    formant2 = np.sin(2 * np.pi * 1400.0 * t) * 0.25
    formant3 = np.sin(2 * np.pi * 2600.0 * t) * 0.15

    # Natural unvoiced breath noise
    noise = np.random.normal(0, 0.015, len(t))

    # Syllabic envelope (simulating speech cadence and pauses)
    cadence = 0.5 * (1 + np.sin(2 * np.pi * 1.8 * t))
    envelope = np.clip(cadence, 0.05, 1.0)
    
    # Smooth attack and release
    fade = int(0.15 * sr)
    envelope[:fade] *= np.linspace(0, 1, fade)
    envelope[-fade:] *= np.linspace(1, 0, fade)

    raw_voice = (glottal + formant1 + formant2 + formant3 + noise) * envelope
    # Peak normalize
    normalized = raw_voice / np.max(np.abs(raw_voice)) * 0.90

    out_path = OUTPUT_DIR / filename
    write_wav_file(str(out_path), normalized, sr)
    print(f"[+] Generated: {out_path} ({duration}s, {sr} Hz)")
    return out_path


def generate_synthetic_vocoder_sample(filename: str = "sample_synthetic_voice.wav", duration: float = 3.5, sr: int = 16000):
    """
    Synthesizes a voice with typical synthetic neural vocoder signatures:
    - Perfectly flat fundamental frequency (robotic / no micro-jitter)
    - Missing high-frequency harmonics above 3800 Hz
    - Slight phase buzziness and harmonic smearing
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Constant rigid pitch (unnatural robotic F0)
    f0 = 150.0
    phase = 2 * np.pi * f0 * t

    # Rigid square/sawtooth buzz characteristic of low-grade vocoder artifacts
    buzz = (
        1.0 * np.sin(phase) +
        0.7 * np.sin(2 * phase) +
        0.5 * np.sin(3 * phase) +
        0.4 * np.sin(4 * phase) +
        0.2 * np.sin(5 * phase)
    )

    # Artificial sharp frequency cutoff simulating 4 kHz low-pass vocoder limitation
    # Sharp formant without natural breath turbulence
    synth_formant = np.sin(2 * np.pi * 800.0 * t) * 0.5 + np.sin(2 * np.pi * 1800.0 * t) * 0.3
    
    # Envelope without natural breath decay
    cadence = 0.5 * (1 + np.cos(2 * np.pi * 2.0 * t))
    envelope = np.clip(cadence, 0.1, 1.0)

    # Add slight 50Hz electrical hum artifact
    hum = 0.03 * np.sin(2 * np.pi * 50.0 * t)

    raw_synth = (buzz + synth_formant + hum) * envelope
    normalized = raw_synth / np.max(np.abs(raw_synth)) * 0.90

    out_path = OUTPUT_DIR / filename
    write_wav_file(str(out_path), normalized, sr)
    print(f"[+] Generated: {out_path} ({duration}s, {sr} Hz)")
    return out_path


def write_wav_file(filepath: str, data: np.ndarray, sr: int):
    """Writes 16-bit PCM WAV file using standard python wave module (no dependencies)."""
    int16_data = (data * 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())


if __name__ == "__main__":
    generate_natural_vocal_sample()
    generate_synthetic_vocoder_sample()
