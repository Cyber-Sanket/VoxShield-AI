"""
Unit tests for VoxShield AI - Deepfake Detection Pipeline & Integrity Rules
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from modules.deepfake_detector import DeepfakeDetectionPipeline, ModelStatus
from modules.feature_extraction import extract_audio_features
from config import MODE_DEMO, LABEL_UNAVAILABLE, LABEL_REAL, LABEL_FAKE


def test_demo_mode_when_no_models_present():
    """
    STRICT INTEGRITY TEST:
    Verifies that when no trained model weights exist in models/,
    the engine operates strictly in Demo / Feature Analysis Mode,
    returning LABEL_UNAVAILABLE and confidence=None.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = DeepfakeDetectionPipeline(models_dir=Path(tmp_dir))
        assert not pipeline.model_status.is_loaded
        assert pipeline.model_status.model_name == "No Active Model"

        # Mock features
        sr = 16000
        signal = np.zeros(16000, dtype=np.float32)
        features = extract_audio_features(signal, sr=sr)

        res = pipeline.analyze(features, signal, sr)

        assert res.mode == MODE_DEMO
        assert res.prediction == LABEL_UNAVAILABLE
        assert res.confidence is None
        assert res.class_probabilities is None
        assert "Demo Analysis" in res.summary_message


def test_corrupted_model_file_handled_gracefully():
    """Verifies that an unreadable/corrupted model file in models/ does not crash the system."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupted_path = Path(tmp_dir) / "corrupt_model.joblib"
        corrupted_path.write_bytes(b"NOT_A_VALID_JOBLIB_PAYLOAD")

        pipeline = DeepfakeDetectionPipeline(models_dir=Path(tmp_dir))
        assert not pipeline.model_status.is_loaded
        assert "Failed" in pipeline.model_status.model_type

        # System continues operating in safe fallback mode without crashing
        sr = 16000
        signal = np.zeros(16000, dtype=np.float32)
        features = extract_audio_features(signal, sr=sr)
        res = pipeline.analyze(features, signal, sr)

        assert res.prediction == LABEL_UNAVAILABLE


def test_model_based_detection_when_model_present():
    """
    Verifies that when a valid trained model is in models/,
    the engine operates in Model-Based Detection mode, returning genuine prediction and confidence.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from config import MODE_MODEL

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock trained sklearn model
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        # Mock 64-dim feature vector
        X = np.random.randn(20, 64)
        y = np.array([0, 1] * 10)
        clf.fit(X, y)

        model_path = Path(tmp_dir) / "test_model.joblib"
        joblib.dump(clf, model_path)

        pipeline = DeepfakeDetectionPipeline(models_dir=Path(tmp_dir))
        assert pipeline.model_status.is_loaded
        assert pipeline.model_status.model_name == "test_model.joblib"

        # Mock feature vector with 64 elements
        sr = 16000
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        features = extract_audio_features(signal, sr=sr)

        res = pipeline.analyze(features, signal, sr)
        assert res.mode == MODE_MODEL
        assert res.prediction in [LABEL_REAL, LABEL_FAKE, "Inconclusive"]
        assert res.confidence is not None
        assert 0.0 <= res.confidence <= 1.0

