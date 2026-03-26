"""Model Registry — loads all Keras models, scalers, and encoders at startup."""

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds all loaded ML artifacts for inference."""

    def __init__(self):
        self.model1 = None
        self.model1_scaler = None

        self.model2 = None
        self.model2_scaler = None
        self.model2_category_encoder = None
        self.model2_diagnosis_encoder = None

        self.model3 = None
        self.model3_scaler = None

        self.model4 = None
        self.model4_scaler = None
        self.model4_available: bool = False

    def load(self, models_dir: str = "models") -> "ModelRegistry":
        import joblib
        import tensorflow as tf  # noqa: F401 — triggers Keras backend

        base = Path(models_dir)

        # --- Model 1: Per-Claim Autoencoder ---
        self.model1 = tf.keras.models.load_model(base / "cbc_model1_claim_autoencoder.keras")
        self.model1_scaler = joblib.load(base / "cbc_model1_scaler.pkl")
        logger.info("Model 1 loaded")

        # --- Model 2: Hierarchical Classifier ---
        self.model2 = tf.keras.models.load_model(base / "cbc_model2_hierarchical_classifier.keras")
        self.model2_scaler = joblib.load(base / "cbc_model2_scaler.pkl")
        self.model2_category_encoder = joblib.load(base / "cbc_model2_category_encoder.pkl")
        self.model2_diagnosis_encoder = joblib.load(base / "cbc_model2_diagnosis_encoder.pkl")
        logger.info("Model 2 loaded")

        # --- Model 3: Patient Temporal LSTM ---
        self.model3 = tf.keras.models.load_model(base / "cbc_model3_patient_temporal_ae.keras")
        self.model3_scaler = joblib.load(base / "cbc_model3_patient_scaler.pkl")
        logger.info("Model 3 loaded")

        # --- Model 4: Facility Temporal LSTM Autoencoder ---
        m4_path = base / "model4_facility_temporal_ae.keras"
        m4_scaler_path = base / "model4_facility_scaler.pkl"
        if m4_path.exists() and m4_scaler_path.exists():
            self.model4 = tf.keras.models.load_model(m4_path)
            self.model4_scaler = joblib.load(m4_scaler_path)
            self.model4_available = True
            logger.info("Model 4 loaded")
        else:
            self.model4_available = False
            logger.info(f"Model 4 not found at {m4_path} — skipping (model4_available=False)")

        return self


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    from app.core.config import settings
    registry = ModelRegistry()
    registry.load(settings.MODELS_DIR)
    return registry
