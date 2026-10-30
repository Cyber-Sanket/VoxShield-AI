"""
VoxShield AI - Deepfake Detection Engine
Modular detection architecture adhering to strict academic and cybersecurity integrity.
Separates genuine Model-Based Detection from Demo / Feature Analysis Mode.
Ghost Protocol | AI & Cyber Security
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from config import (
    MODELS_DIR, MODE_MODEL, MODE_DEMO,
    LABEL_REAL, LABEL_FAKE, LABEL_INCONCLUSIVE, LABEL_UNAVAILABLE
)
from modules.feature_extraction import ExtractedFeatures


@dataclass
class ModelStatus:
    is_loaded: bool
    model_name: str
    model_type: str
    model_path: Optional[str] = None
    device: str = "CPU"
    details: str = "No pre-trained weights active"


@dataclass
class DetectionResult:
    mode: str                                  # MODE_MODEL or MODE_DEMO
    prediction: str                            # Real, Fake, Inconclusive, or Unavailable
    confidence: Optional[float]                # 0.0 to 1.0 (ONLY when genuine model computes it)
    class_probabilities: Optional[Dict[str, float]]
    model_status: ModelStatus
    summary_message: str
    forensic_notes: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


class DeepfakeDetectionPipeline:
    """
    Manages detection model discovery, loading, and inference.
    If no trained model weights exist in models/, operates strictly in Demo / Feature Analysis Mode.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.active_model = None
        self.model_status = self._discover_and_load_model()

    def _discover_and_load_model(self) -> ModelStatus:
        """
        Inspects models/ directory for valid serialized model weights:
        Supports Scikit-Learn pipelines (.joblib), PyTorch models (.pt, .pth), or ONNX.
        """
        if not self.models_dir.exists():
            return ModelStatus(
                is_loaded=False,
                model_name="None (Unloaded)",
                model_type="None",
                details="Models directory does not exist yet."
            )

        # 1. Check for scikit-learn baseline model
        sklearn_files = list(self.models_dir.glob("*.joblib")) + list(self.models_dir.glob("*.pkl"))
        if sklearn_files:
            target_file = sklearn_files[0]
            try:
                import joblib
                loaded = joblib.load(target_file)
                self.active_model = ("sklearn", loaded)
                return ModelStatus(
                    is_loaded=True,
                    model_name=target_file.name,
                    model_type="Scikit-Learn Acoustic Feature Classifier",
                    model_path=str(target_file),
                    device="CPU",
                    details=f"Loaded trained classifier: {target_file.name}"
                )
            except Exception as e:
                return ModelStatus(
                    is_loaded=False,
                    model_name=target_file.name,
                    model_type="Failed Loading",
                    model_path=str(target_file),
                    details=f"Error reading model weights: {e}"
                )

        # 2. Check for PyTorch model weights (.pt or .pth)
        torch_files = list(self.models_dir.glob("*.pt")) + list(self.models_dir.glob("*.pth"))
        if torch_files:
            target_file = torch_files[0]
            try:
                import torch
                # Attempt loading state dict or torchscript model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                loaded = torch.load(target_file, map_location=device, weights_only=False)
                self.active_model = ("torch", loaded, device)
                return ModelStatus(
                    is_loaded=True,
                    model_name=target_file.name,
                    model_type="PyTorch Neural Network",
                    model_path=str(target_file),
                    device=device.upper(),
                    details=f"Loaded PyTorch weights: {target_file.name}"
                )
            except Exception as e:
                return ModelStatus(
                    is_loaded=False,
                    model_name=target_file.name,
                    model_type="Failed Loading",
                    model_path=str(target_file),
                    details=f"Error loading PyTorch weights: {e}"
                )

        # 3. Default state: No model file present
        return ModelStatus(
            is_loaded=False,
            model_name="No Active Model",
            model_type="None (Demo Analysis Mode Active)",
            model_path=None,
            device="CPU",
            details="No trained weights file found in models/ directory. Running in feature analysis mode."
        )

    def reload_models(self) -> ModelStatus:
        """Refreshes and checks the models directory for newly dropped weights."""
        self.model_status = self._discover_and_load_model()
        return self.model_status

    def analyze(
        self,
        features: ExtractedFeatures,
        signal: np.ndarray,
        sr: int
    ) -> DetectionResult:
        """
        Executes analysis according to availability of genuine model weights:
        Mode A: Model-Based Detection (if trained weights are loaded)
        Mode B: Demo / Feature Analysis Mode (if no model weights are available)
        """
        # Strict integrity check: do we have a loaded trained model?
        if self.model_status.is_loaded and self.active_model is not None:
            return self._run_model_detection(features, signal, sr)
        else:
            return self._run_demo_analysis(features)

    def _run_model_detection(
        self,
        features: ExtractedFeatures,
        signal: np.ndarray,
        sr: int
    ) -> DetectionResult:
        """Mode A: Genuine classification through loaded trained model."""
        model_flavor = self.active_model[0]

        try:
            if model_flavor == "sklearn":
                classifier = self.active_model[1]
                # Prepare flat feature vector
                x = features.flat_feature_vector.reshape(1, -1)
                
                # Check for predict_proba
                if hasattr(classifier, "predict_proba"):
                    probs = classifier.predict_proba(x)[0]
                    classes = getattr(classifier, "classes_", [0, 1])
                    # Map 0 to Real, 1 to Fake or label names
                    prob_dict = {}
                    for cls_label, prob in zip(classes, probs):
                        name = LABEL_REAL if str(cls_label).lower() in ["0", "real"] else LABEL_FAKE
                        prob_dict[name] = float(prob)
                    
                    fake_prob = prob_dict.get(LABEL_FAKE, 0.5)
                    real_prob = prob_dict.get(LABEL_REAL, 0.5)

                    if abs(fake_prob - real_prob) < 0.10:
                        pred_label = LABEL_INCONCLUSIVE
                        conf = float(max(fake_prob, real_prob))
                    elif fake_prob > real_prob:
                        pred_label = LABEL_FAKE
                        conf = float(fake_prob)
                    else:
                        pred_label = LABEL_REAL
                        conf = float(real_prob)
                else:
                    pred = classifier.predict(x)[0]
                    pred_label = LABEL_FAKE if str(pred).lower() in ["1", "fake"] else LABEL_REAL
                    conf = 1.0
                    prob_dict = {pred_label: 1.0}

                return DetectionResult(
                    mode=MODE_MODEL,
                    prediction=pred_label,
                    confidence=round(conf, 4),
                    class_probabilities=prob_dict,
                    model_status=self.model_status,
                    summary_message=f"Model '{self.model_status.model_name}' classified this audio as {pred_label} with {conf*100:.1f}% confidence.",
                    forensic_notes=[
                        f"Features processed: {len(features.flat_feature_vector)} acoustic descriptors",
                        "Decision boundary calculated from trained scikit-learn classifier"
                    ],
                    limitations=[
                        "Performance depends entirely on the domain and diversity of the model's training dataset.",
                        "Adversarial compression, re-encoding, and background noise may reduce classification fidelity."
                    ]
                )

            elif model_flavor == "torch":
                # PyTorch forward pass
                model = self.active_model[1]
                device = self.active_model[2]
                import torch

                model.eval()
                with torch.no_grad():
                    # Check if model expects 1D feature vector or raw waveform
                    tensor_x = torch.from_numpy(features.flat_feature_vector).float().unsqueeze(0).to(device)
                    out = model(tensor_x)
                    probs = torch.softmax(out, dim=-1).cpu().numpy()[0]
                    fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    real_prob = 1.0 - fake_prob

                    if abs(fake_prob - real_prob) < 0.10:
                        pred_label = LABEL_INCONCLUSIVE
                        conf = max(fake_prob, real_prob)
                    elif fake_prob > real_prob:
                        pred_label = LABEL_FAKE
                        conf = fake_prob
                    else:
                        pred_label = LABEL_REAL
                        conf = real_prob

                    return DetectionResult(
                        mode=MODE_MODEL,
                        prediction=pred_label,
                        confidence=round(conf, 4),
                        class_probabilities={LABEL_REAL: round(real_prob, 4), LABEL_FAKE: round(fake_prob, 4)},
                        model_status=self.model_status,
                        summary_message=f"Neural model '{self.model_status.model_name}' produced a {pred_label} classification with {conf*100:.1f}% confidence.",
                        forensic_notes=["Inference executed via PyTorch neural network runtime."],
                        limitations=["PyTorch model evaluation requires independent cross-dataset testing for generalization."]
                    )

        except Exception as err:
            # Fallback to Demo mode if inference failed due to dimension mismatch or error
            return DetectionResult(
                mode=MODE_MODEL,
                prediction=LABEL_INCONCLUSIVE,
                confidence=None,
                class_probabilities=None,
                model_status=self.model_status,
                summary_message=f"Model evaluation encountered an execution error: {err}. Falling back to inconclusive.",
                forensic_notes=[f"Model execution error: {err}"],
                limitations=["Ensure model input dimensionality matches the acoustic feature vector dimension."]
            )

        # Fallback
        return self._run_demo_analysis(features)

    def _run_demo_analysis(self, features: ExtractedFeatures) -> DetectionResult:
        """
        Mode B: Academic Demo & Acoustic Feature Analysis Mode.
        CRITICAL INTEGRITY ENFORCEMENT:
        No fake random predictions or fabricated confidence numbers are returned.
        Clear labeling that no trained model is active.
        """
        # Analyze observed acoustic markers for educational / forensic inspection
        notes = []
        mean_centroid = features.summary_stats.get("spectral_centroid", {}).get("mean", 0.0)
        mean_rolloff = features.summary_stats.get("spectral_rolloff", {}).get("mean", 0.0)
        mean_zcr = features.summary_stats.get("zero_crossing_rate", {}).get("mean", 0.0)
        mean_flatness = features.summary_stats.get("spectral_flatness", {}).get("mean", 0.0)

        notes.append(f"Spectral Centroid average: {mean_centroid:.1f} Hz (typical adult speech: 1000 - 3500 Hz)")
        notes.append(f"Spectral Rolloff (85% energy): {mean_rolloff:.1f} Hz")
        notes.append(f"Zero Crossing Rate mean: {mean_zcr:.4f}")
        notes.append(f"Spectral Flatness index: {mean_flatness:.4f} (indicates harmonic tonality vs. noise)")
        notes.append("Acoustic parameters successfully extracted across all time frames.")

        limitations = [
            "Academic Integrity Note: No trained deepfake detection model weights are currently loaded in the 'models/' directory.",
            "Acoustic features can highlight vocoder signatures, but reliable automated classification requires a model trained on verified datasets (e.g. ASVspoel, In-the-Wild, WaveFake).",
            "To activate Model-Based Detection, place a trained model (.joblib or .pt) into the 'models/' folder.",
            "Raw feature values alone without calibrated statistical boundaries should not be used as definitive proof of authenticity."
        ]

        return DetectionResult(
            mode=MODE_DEMO,
            prediction=LABEL_UNAVAILABLE,
            confidence=None,
            class_probabilities=None,
            model_status=self.model_status,
            summary_message=(
                "Demo Analysis - Not a Verified Deepfake Prediction. "
                "The signal processing and feature extraction pipeline completed successfully. "
                "Automated classification is disabled because no pre-trained deepfake model is active."
            ),
            forensic_notes=notes,
            limitations=limitations
        )
