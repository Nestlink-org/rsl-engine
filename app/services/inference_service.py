"""Inference service — runs Models 1, 2, and 3 for a single claim."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Thresholds from model metadata
MODEL1_THRESHOLD = 5.004205464072272e-05
MODEL3_THRESHOLD = 0.2951826353643561
MODEL4_THRESHOLD = 1.2420042311621498

# Feature orders from model metadata
MODEL1_FEATURES = ["age", "sex_encoded", "HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT"]
MODEL2_FEATURES = ["HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT"]
MODEL3_FEATURES = ["age", "sex_encoded", "HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT", "length_of_stay"]
MODEL4_FEATURES = [
    "claim_volume", "avg_age", "age_std", "pct_male",
    "HGB_mean", "HGB_std", "HCT_mean", "HCT_std",
    "MCV_mean", "MCV_std", "MCHC_mean", "MCHC_std",
    "NEU_mean", "LYM_mean", "EOS_mean", "BAS_mean", "MON_mean", "PLT_mean",
    "avg_los",
]
MODEL4_SEQ_LEN = 8

SEQUENCE_LENGTH = 5


def _severity(score: float) -> str:
    if score > 0.8:
        return "high"
    if score > 0.5:
        return "medium"
    return "low"


@dataclass
class Model1Result:
    anomaly_score: float
    is_anomaly: bool
    severity: str
    flag_reason: str
    mse: float


@dataclass
class Model2Result:
    anomaly_score: float
    is_anomaly: bool
    severity: str
    flag_reason: str
    predicted_category: str
    predicted_diagnosis: str
    category_confidence: float
    diagnosis_confidence: float


@dataclass
class Model3Result:
    trajectory_anomaly_score: float
    is_trajectory_anomaly: bool
    per_visit_errors: List[float]
    most_anomalous_visit_index: int
    insufficient_history: bool


@dataclass
class Model4Result:
    facility_anomaly_score: float
    is_facility_anomaly: bool
    severity: str
    flag_reason: str
    per_week_errors: List[float]
    most_anomalous_week_index: int
    insufficient_history: bool


def run_model1(features: Dict[str, Any], registry: ModelRegistry) -> Model1Result:
    """Per-claim autoencoder anomaly detection."""
    x = np.array([[features[f] for f in MODEL1_FEATURES]], dtype=np.float32)
    x_scaled = registry.model1_scaler.transform(x)
    x_recon = registry.model1.predict(x_scaled, verbose=0)
    mse = float(np.mean((x_scaled - x_recon) ** 2))
    score = min(1.0, mse / MODEL1_THRESHOLD)
    is_anomaly = score >= 0.5
    sev = _severity(score)
    reason = f"CBC pattern anomaly detected (score={score:.3f})" if is_anomaly else "CBC pattern within normal range"
    return Model1Result(anomaly_score=score, is_anomaly=is_anomaly, severity=sev, flag_reason=reason, mse=mse)


def run_model2(features: Dict[str, Any], claimed_diagnosis: str, registry: ModelRegistry) -> Model2Result:
    """Hierarchical disease classifier — detects diagnosis/category mismatches."""
    x = np.array([[features[f] for f in MODEL2_FEATURES]], dtype=np.float32)
    x_scaled = registry.model2_scaler.transform(x)
    preds = registry.model2.predict(x_scaled, verbose=0)

    # preds is a list: [category_probs, diagnosis_probs]
    cat_probs, diag_probs = preds[0][0], preds[1][0]
    cat_idx = int(np.argmax(cat_probs))
    diag_idx = int(np.argmax(diag_probs))

    predicted_category = registry.model2_category_encoder.classes_[cat_idx]
    predicted_diagnosis = registry.model2_diagnosis_encoder.classes_[diag_idx]
    cat_conf = float(cat_probs[cat_idx])
    diag_conf = float(diag_probs[diag_idx])

    # Mismatch detection
    claimed_upper = claimed_diagnosis.strip().upper()
    predicted_upper = predicted_diagnosis.strip().upper()
    mismatch = claimed_upper != predicted_upper

    if mismatch:
        score = min(1.0, (1.0 - diag_conf) + 0.3)
        is_anomaly = True
        reason = (
            f"Diagnosis mismatch: claimed '{claimed_diagnosis}' "
            f"but model predicts '{predicted_diagnosis}' ({diag_conf:.1%} confidence)"
        )
    else:
        score = 1.0 - diag_conf
        is_anomaly = score > 0.5
        reason = f"Diagnosis consistent with CBC pattern ({diag_conf:.1%} confidence)"

    return Model2Result(
        anomaly_score=min(1.0, score),
        is_anomaly=is_anomaly,
        severity=_severity(min(1.0, score)),
        flag_reason=reason,
        predicted_category=predicted_category,
        predicted_diagnosis=predicted_diagnosis,
        category_confidence=cat_conf,
        diagnosis_confidence=diag_conf,
    )


def run_model3(
    sequence: List[Dict[str, Any]],
    registry: ModelRegistry,
    insufficient_history: bool,
) -> Model3Result:
    """Patient temporal LSTM autoencoder — trajectory anomaly detection."""
    # Pad or truncate to SEQUENCE_LENGTH
    if len(sequence) < SEQUENCE_LENGTH:
        pad = [{f: 0.0 for f in MODEL3_FEATURES}] * (SEQUENCE_LENGTH - len(sequence))
        sequence = pad + sequence
    else:
        sequence = sequence[-SEQUENCE_LENGTH:]

    x = np.array([[visit[f] for f in MODEL3_FEATURES] for visit in sequence], dtype=np.float32)
    # Scale each timestep independently
    x_flat = x.reshape(-1, len(MODEL3_FEATURES))
    x_scaled_flat = registry.model3_scaler.transform(x_flat)
    x_scaled = x_scaled_flat.reshape(1, SEQUENCE_LENGTH, len(MODEL3_FEATURES))

    x_recon = registry.model3.predict(x_scaled, verbose=0)
    per_visit_mse = np.mean((x_scaled[0] - x_recon[0]) ** 2, axis=1).tolist()
    overall_mse = float(np.mean(per_visit_mse))
    score = min(1.0, overall_mse / MODEL3_THRESHOLD)
    is_anomaly = score >= 0.5
    most_anomalous = int(np.argmax(per_visit_mse))

    return Model3Result(
        trajectory_anomaly_score=score,
        is_trajectory_anomaly=is_anomaly,
        per_visit_errors=per_visit_mse,
        most_anomalous_visit_index=most_anomalous,
        insufficient_history=insufficient_history,
    )


def run_model4(
    weekly_sequence: List[Dict[str, Any]],
    registry: ModelRegistry,
) -> Model4Result:
    """Facility temporal LSTM autoencoder — detects anomalous facility-level weekly patterns."""
    insufficient = len(weekly_sequence) < MODEL4_SEQ_LEN

    # Pad or truncate to MODEL4_SEQ_LEN
    if len(weekly_sequence) < MODEL4_SEQ_LEN:
        pad = [{f: 0.0 for f in MODEL4_FEATURES}] * (MODEL4_SEQ_LEN - len(weekly_sequence))
        weekly_sequence = pad + weekly_sequence
    else:
        weekly_sequence = weekly_sequence[-MODEL4_SEQ_LEN:]

    x = np.array([[w.get(f, 0.0) for f in MODEL4_FEATURES] for w in weekly_sequence], dtype=np.float32)
    x_flat = x.reshape(-1, len(MODEL4_FEATURES))
    import pandas as pd
    x_flat_df = pd.DataFrame(x_flat, columns=MODEL4_FEATURES)
    x_scaled_flat = registry.model4_scaler.transform(x_flat_df)
    # Replace NaN from zero-variance features with 0.0 (mean-centered, no scale)
    x_scaled_flat = np.nan_to_num(x_scaled_flat, nan=0.0)
    x_scaled = x_scaled_flat.reshape(1, MODEL4_SEQ_LEN, len(MODEL4_FEATURES))

    x_recon = registry.model4.predict(x_scaled, verbose=0)
    per_week_mse = np.mean((x_scaled[0] - x_recon[0]) ** 2, axis=1).tolist()
    overall_mse = float(np.mean(per_week_mse))
    score = min(1.0, overall_mse / MODEL4_THRESHOLD)
    is_anomaly = score >= 0.5
    most_anomalous = int(np.argmax(per_week_mse))
    sev = _severity(score)
    reason = (
        f"Facility behavior anomaly detected (score={score:.3f})"
        if is_anomaly else "Facility behavior within normal range"
    )
    return Model4Result(
        facility_anomaly_score=score,
        is_facility_anomaly=is_anomaly,
        severity=sev,
        flag_reason=reason,
        per_week_errors=per_week_mse,
        most_anomalous_week_index=most_anomalous,
        insufficient_history=insufficient,
    )


async def run_inference(
    claim_data: Dict[str, Any],
    patient_history: List[Dict[str, Any]],
    registry: ModelRegistry,
    facility_weekly_sequence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run models 1, 2, 3 concurrently. Runs model 4 if registry.model4_available
    and facility_weekly_sequence is provided.
    Returns dict with model1, model2, model3, and optionally model4 results.
    """
    t0 = time.time()
    loop = asyncio.get_event_loop()
    insufficient_history = len(patient_history) < 5

    # Build sequence: history + current visit
    current_visit = {f: claim_data.get(f, 0.0) for f in MODEL3_FEATURES}
    sequence = patient_history[-4:] + [current_visit]  # up to 5 visits

    m1, m2, m3 = await asyncio.gather(
        loop.run_in_executor(None, run_model1, claim_data, registry),
        loop.run_in_executor(None, run_model2, claim_data, claim_data.get("claimed_diagnosis", ""), registry),
        loop.run_in_executor(None, run_model3, sequence, registry, insufficient_history),
    )

    result: Dict[str, Any] = {"model1": m1, "model2": m2, "model3": m3}

    if registry.model4_available and facility_weekly_sequence is not None:
        m4 = await loop.run_in_executor(None, run_model4, facility_weekly_sequence, registry)
        result["model4"] = m4
        logger.info(
            f"[inference] claim={claim_data.get('claim_id')} "
            f"m1={m1.anomaly_score:.3f}({'A' if m1.is_anomaly else 'N'}) "
            f"m2={m2.anomaly_score:.3f}({'A' if m2.is_anomaly else 'N'}) "
            f"m3={m3.trajectory_anomaly_score:.3f}({'A' if m3.is_trajectory_anomaly else 'N'}) "
            f"m4={m4.facility_anomaly_score:.3f}({'A' if m4.is_facility_anomaly else 'N'}) "
            f"elapsed={time.time()-t0:.3f}s"
        )
    else:
        logger.info(
            f"[inference] claim={claim_data.get('claim_id')} "
            f"m1={m1.anomaly_score:.3f}({'A' if m1.is_anomaly else 'N'}) "
            f"m2={m2.anomaly_score:.3f}({'A' if m2.is_anomaly else 'N'}) "
            f"m3={m3.trajectory_anomaly_score:.3f}({'A' if m3.is_trajectory_anomaly else 'N'}) "
            f"elapsed={time.time()-t0:.3f}s"
        )

    return result
