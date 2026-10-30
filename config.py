"""
VoxShield AI - Configuration Module
Ghost Protocol | AI & Cyber Security
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "voxshield.db"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
SAMPLES_DIR = ASSETS_DIR / "sample_audio"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# App Metadata
APP_NAME = "VoxShield AI"
APP_TAGLINE = "Detect Synthetic Voices. Protect Digital Trust."
TEAM_NAME = "Ghost Protocol"
DOMAIN = "AI & Cyber Security"
VERSION = "1.0.0"

# Audio Constraints & Defaults
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
MAX_FILE_SIZE_MB = 25
TARGET_SAMPLE_RATE = 16000  # Standard speech recognition & deepfake modeling sample rate
MAX_AUDIO_DURATION_SECONDS = 60.0  # Limit for responsive demo processing

# Feature Extraction Hyperparameters
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TOP_DB = 80.0

# Detection Mode Constants
MODE_MODEL = "Model-Based Detection"
MODE_DEMO = "Demo / Feature Analysis Mode"

# Classification Labels (Only valid when active model is present)
LABEL_REAL = "Real"
LABEL_FAKE = "Fake"
LABEL_INCONCLUSIVE = "Inconclusive"
LABEL_UNAVAILABLE = "Analysis Only - Prediction Unavailable"

# UI Theme Color Palette
THEME = {
    "bg_dark": "#0A0F1D",
    "card_bg": "#111827",
    "card_border": "#1E293B",
    "card_hover": "#1F293D",
    "primary": "#00FFA3",      # Neon Green
    "cyan": "#00E5FF",         # Neon Cyan
    "blue": "#3B82F6",         # Electric Blue
    "accent_purple": "#A855F7",
    "danger": "#EF4444",       # Warning / Fake
    "warning": "#F59E0B",
    "success": "#10B981",      # Verified / Real
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
}
