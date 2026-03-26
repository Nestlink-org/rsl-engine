"""CBC model tools for the RSL agent pipeline.

Four tools, one per model. Each tool:
  - Accepts a structured claim dict (already validated by model_validator)
  - Runs the corresponding model via inference_service
  - Returns a typed result dict
  - Logs timing and outcome

Tools are plain functions (not LangChain @tool decorated) so they can be called
directly from the orchestrator node without async overhead. The orchestrator
decides which tools to call based on _model_validation in the claim dict.

Model input contracts (from metadata + test scripts):
  Model 1 — Claim Autoencoder:
    features: age, sex_encoded, HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT (12)
    threshold: 5.004205464072272e-05
    output: anomaly_score, is_anomaly, severity, flag_reason, mse, top_features

  Model 2 — Hierarchical Disease Classifier:
    features: HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT (10)
    categories: obstetric, respiratory, trauma
    diagnoses: APH PPH, ASTHMA, PNEUMONIA, PUERPERAL SEPSIS, TBI
    output: predicted_category, predicted_diagnosis, confidences, mismatch flag

  Model 3 — Patient Temporal LSTM Autoencoder:
    features per visit: age, sex_encoded, HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT, length_of_stay (13)
    sequence_length: 5
    threshold: 0.2951826353643561
    output: trajectory_anomaly_score, is_trajectory_anomaly, per_visit_errors, most_anomalous_visit

  Model 4 — Facility Temporal LSTM Autoencoder:
    features per week: claim_volume, avg_age, age_std, pct_male, HGB_mean, HGB_std,
                       HCT_mean, HCT_std, MCV_mean, MCV_std, MCHC_mean, MCHC_std,
                       NEU_mean, LYM_mean, EOS_mean, BAS_mean, MON_mean, PLT_mean, avg_los (19)
    sequence_length: 8
    threshold: 1.2420042311621498
    output: facility_anomaly_score, is_facility_anomaly, per_week_errors, most_anomalous_week
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Feature lists (match training exactly) ───────────────────────────────────

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

MODEL1_THRESHOLD = 5.004205464072272e-05
MODEL3_THRESHOLD = 0.2951826353643561
MODEL4_THRESHOLD = 1.2420042311621498
MODEL3_SEQ_LEN = 5
MODEL4_SEQ_LEN = 8


def _severity(score: float) -> str:
    if score > 0.8:
        return "high"
    if score > 0.5:
        return "medium"
    return "low"


# ── Tool 1: CBC Claim Autoencoder ─────────────────────────────────────────────

def run_cbc_model1(claim: Dict[str, Any], registry) -> Dict[str, Any]:
    """
    Run Model 1 (CBC Claim Autoencoder) on a single claim.

    Input: claim dict with age, sex_encoded + 10 CBC lab values
    Output: anomaly_score, is_anomaly, severity, flag_reason, mse, top_features
    """
    import numpy as np
    t0 = time.time()
    claim_id = claim.get("claim_id", "unknown")
    logger.info(f"[cbc_tool1] START claim_id={claim_id}")

    try:
        x = np.array([[float(claim[f]) for f in MODEL1_FEATURES]], dtype=np.float32)
        x_scaled = registry.model1_scaler.transform(x)
        x_recon = registry.model1.predict(x_scaled, verbose=0)

        mse = float(np.mean((x_scaled - x_recon) ** 2))
        score = min(1.0, mse / MODEL1_THRESHOLD)
        is_anomaly = score >= 0.5
        sev = _severity(score)

        # Top contributing features (per-feature squared error)
        feature_errors = np.square(x_scaled[0] - x_recon[0])
        top_idx = feature_errors.argsort()[-5:][::-1]
        top_features = [
            {"feature": MODEL1_FEATURES[i], "error": float(feature_errors[i])}
            for i in top_idx
        ]

        reason = (
            f"CBC pattern anomaly (score={score:.3f}, top: {top_features[0]['feature']})"
            if is_anomaly else f"CBC pattern normal (score={score:.3f})"
        )

        result = {
            "model_id": 1,
            "model_name": "CBC Claim Autoencoder",
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "severity": sev,
            "flag_reason": reason,
            "mse": round(mse, 8),
            "top_features": top_features,
            "eligible": True,
        }
        logger.info(
            f"[cbc_tool1] DONE claim_id={claim_id} score={score:.3f} "
            f"anomaly={is_anomaly} elapsed={time.time()-t0:.3f}s"
        )
        return result

    except Exception as e:
        logger.error(f"[cbc_tool1] FAILED claim_id={claim_id}: {e}", exc_info=True)
        return {"model_id": 1, "eligible": True, "error": str(e), "is_anomaly": False, "anomaly_score": 0.0}


# ── Tool 2: CBC Disease Classifier ───────────────────────────────────────────

def run_cbc_model2(claim: Dict[str, Any], registry) -> Dict[str, Any]:
    """
    Run Model 2 (CBC Hierarchical Disease Classifier) on a single claim.

    Input: claim dict with 10 CBC lab values
    Output: predicted_category, predicted_diagnosis, confidences, mismatch detection
    """
    import numpy as np
    t0 = time.time()
    claim_id = claim.get("claim_id", "unknown")
    claimed_diagnosis = str(claim.get("diagnosis") or claim.get("claimed_diagnosis") or "").strip()
    logger.info(f"[cbc_tool2] START claim_id={claim_id} claimed_diagnosis={claimed_diagnosis!r}")

    try:
        x = np.array([[float(claim[f]) for f in MODEL2_FEATURES]], dtype=np.float32)
        x_scaled = registry.model2_scaler.transform(x)
        preds = registry.model2.predict(x_scaled, verbose=0)

        cat_probs, diag_probs = preds[0][0], preds[1][0]
        cat_idx = int(cat_probs.argmax())
        diag_idx = int(diag_probs.argmax())

        predicted_category = registry.model2_category_encoder.classes_[cat_idx]
        predicted_diagnosis = registry.model2_diagnosis_encoder.classes_[diag_idx]
        cat_conf = float(cat_probs[cat_idx])
        diag_conf = float(diag_probs[diag_idx])

        # Mismatch: claimed diagnosis vs model prediction
        mismatch = (
            claimed_diagnosis.upper() != predicted_diagnosis.upper()
            if claimed_diagnosis else False
        )
        if mismatch:
            score = min(1.0, (1.0 - diag_conf) + 0.3)
            is_anomaly = True
            reason = (
                f"Diagnosis mismatch: claimed '{claimed_diagnosis}' "
                f"but CBC pattern predicts '{predicted_diagnosis}' ({diag_conf:.1%})"
            )
        else:
            score = 1.0 - diag_conf
            is_anomaly = score > 0.5
            reason = f"Diagnosis consistent with CBC pattern ({diag_conf:.1%} confidence)"

        result = {
            "model_id": 2,
            "model_name": "CBC Disease Classifier",
            "anomaly_score": round(min(1.0, score), 4),
            "is_anomaly": is_anomaly,
            "severity": _severity(min(1.0, score)),
            "flag_reason": reason,
            "predicted_category": predicted_category,
            "predicted_diagnosis": predicted_diagnosis,
            "category_confidence": round(cat_conf, 4),
            "diagnosis_confidence": round(diag_conf, 4),
            "claimed_diagnosis": claimed_diagnosis,
            "mismatch": mismatch,
            "eligible": True,
        }
        logger.info(
            f"[cbc_tool2] DONE claim_id={claim_id} pred={predicted_diagnosis} "
            f"conf={diag_conf:.3f} mismatch={mismatch} elapsed={time.time()-t0:.3f}s"
        )
        return result

    except Exception as e:
        logger.error(f"[cbc_tool2] FAILED claim_id={claim_id}: {e}", exc_info=True)
        return {"model_id": 2, "eligible": True, "error": str(e), "is_anomaly": False, "anomaly_score": 0.0}


# ── Tool 3: Patient Temporal LSTM ─────────────────────────────────────────────

def run_cbc_model3(
    claim: Dict[str, Any],
    patient_history: List[Dict[str, Any]],
    registry,
) -> Dict[str, Any]:
    """
    Run Model 3 (Patient Temporal LSTM Autoencoder).

    Input: current claim + up to 4 prior visits (patient_history)
    Sequence: [history[-4:]] + [current_visit] → padded to seq_len=5
    Output: trajectory_anomaly_score, is_trajectory_anomaly, per_visit_errors
    """
    import numpy as np
    t0 = time.time()
    claim_id = claim.get("claim_id", "unknown")
    insufficient = len(patient_history) < 4
    logger.info(
        f"[cbc_tool3] START claim_id={claim_id} "
        f"history_visits={len(patient_history)} insufficient={insufficient}"
    )

    try:
        current_visit = {f: float(claim.get(f, 0.0)) for f in MODEL3_FEATURES}
        sequence = (patient_history[-4:] if len(patient_history) >= 4 else patient_history) + [current_visit]

        # Pad to MODEL3_SEQ_LEN
        if len(sequence) < MODEL3_SEQ_LEN:
            pad = [{f: 0.0 for f in MODEL3_FEATURES}] * (MODEL3_SEQ_LEN - len(sequence))
            sequence = pad + sequence
        else:
            sequence = sequence[-MODEL3_SEQ_LEN:]

        x = np.array([[v[f] for f in MODEL3_FEATURES] for v in sequence], dtype=np.float32)
        x_flat = x.reshape(-1, len(MODEL3_FEATURES))
        x_scaled_flat = registry.model3_scaler.transform(x_flat)
        x_scaled = x_scaled_flat.reshape(1, MODEL3_SEQ_LEN, len(MODEL3_FEATURES))

        x_recon = registry.model3.predict(x_scaled, verbose=0)
        per_visit_mse = np.mean((x_scaled[0] - x_recon[0]) ** 2, axis=1).tolist()
        overall_mse = float(np.mean(per_visit_mse))
        score = min(1.0, overall_mse / MODEL3_THRESHOLD)
        is_anomaly = score >= 0.5
        most_anomalous = int(np.argmax(per_visit_mse))

        result = {
            "model_id": 3,
            "model_name": "Patient Temporal LSTM",
            "trajectory_anomaly_score": round(score, 4),
            "is_trajectory_anomaly": is_anomaly,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "severity": _severity(score),
            "flag_reason": (
                f"Trajectory anomaly at visit {most_anomalous+1} (score={score:.3f})"
                if is_anomaly else f"Trajectory normal (score={score:.3f})"
            ),
            "per_visit_errors": [round(e, 6) for e in per_visit_mse],
            "most_anomalous_visit_index": most_anomalous,
            "insufficient_history": insufficient,
            "eligible": True,
        }
        logger.info(
            f"[cbc_tool3] DONE claim_id={claim_id} score={score:.3f} "
            f"anomaly={is_anomaly} elapsed={time.time()-t0:.3f}s"
        )
        return result

    except Exception as e:
        logger.error(f"[cbc_tool3] FAILED claim_id={claim_id}: {e}", exc_info=True)
        return {
            "model_id": 3, "eligible": True, "error": str(e),
            "is_anomaly": False, "anomaly_score": 0.0,
            "trajectory_anomaly_score": 0.0, "is_trajectory_anomaly": False,
            "insufficient_history": True,
        }


# ── Tool 4: Facility Temporal LSTM ────────────────────────────────────────────

def run_cbc_model4(
    weekly_sequence: List[Dict[str, Any]],
    facility_id: str,
    registry,
) -> Dict[str, Any]:
    """
    Run Model 4 (Facility Temporal LSTM Autoencoder).

    Input: list of weekly aggregate dicts for a facility (up to 8 weeks)
    Output: facility_anomaly_score, is_facility_anomaly, per_week_errors
    """
    import numpy as np
    import pandas as pd
    t0 = time.time()
    insufficient = len(weekly_sequence) < MODEL4_SEQ_LEN
    logger.info(
        f"[cbc_tool4] START facility_id={facility_id} "
        f"weeks={len(weekly_sequence)} insufficient={insufficient}"
    )

    try:
        # Pad or truncate
        if len(weekly_sequence) < MODEL4_SEQ_LEN:
            pad = [{f: 0.0 for f in MODEL4_FEATURES}] * (MODEL4_SEQ_LEN - len(weekly_sequence))
            weekly_sequence = pad + weekly_sequence
        else:
            weekly_sequence = weekly_sequence[-MODEL4_SEQ_LEN:]

        x = np.array([[w.get(f, 0.0) for f in MODEL4_FEATURES] for w in weekly_sequence], dtype=np.float32)
        x_flat = x.reshape(-1, len(MODEL4_FEATURES))
        x_df = pd.DataFrame(x_flat, columns=MODEL4_FEATURES)
        x_scaled_flat = registry.model4_scaler.transform(x_df)
        x_scaled_flat = np.nan_to_num(x_scaled_flat, nan=0.0)
        x_scaled = x_scaled_flat.reshape(1, MODEL4_SEQ_LEN, len(MODEL4_FEATURES))

        x_recon = registry.model4.predict(x_scaled, verbose=0)
        per_week_mse = np.mean((x_scaled[0] - x_recon[0]) ** 2, axis=1).tolist()
        overall_mse = float(np.mean(per_week_mse))
        score = min(1.0, overall_mse / MODEL4_THRESHOLD)
        is_anomaly = score >= 0.5
        most_anomalous = int(np.argmax(per_week_mse))
        sev = _severity(score)

        result = {
            "model_id": 4,
            "model_name": "Facility Temporal LSTM",
            "facility_anomaly_score": round(score, 4),
            "is_facility_anomaly": is_anomaly,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "severity": sev,
            "flag_reason": (
                f"Facility anomaly at week {most_anomalous+1} (score={score:.3f})"
                if is_anomaly else f"Facility behavior normal (score={score:.3f})"
            ),
            "per_week_errors": [round(e, 6) for e in per_week_mse],
            "most_anomalous_week_index": most_anomalous,
            "insufficient_history": insufficient,
            "eligible": True,
        }
        logger.info(
            f"[cbc_tool4] DONE facility_id={facility_id} score={score:.3f} "
            f"anomaly={is_anomaly} elapsed={time.time()-t0:.3f}s"
        )
        return result

    except Exception as e:
        logger.error(f"[cbc_tool4] FAILED facility_id={facility_id}: {e}", exc_info=True)
        return {
            "model_id": 4, "eligible": True, "error": str(e),
            "is_anomaly": False, "anomaly_score": 0.0,
            "facility_anomaly_score": 0.0, "is_facility_anomaly": False,
            "insufficient_history": True,
        }
