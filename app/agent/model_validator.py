"""Model input validator — checks which models can be triggered for a given claim.

Each model has a strict set of required input features. A model is only triggered
if ALL its required features are present and non-null in the structured claim.

CBC Models:
  Model 1 — Claim Autoencoder:       age, sex_encoded + 10 CBC labs
  Model 2 — Disease Classifier:      10 CBC labs only
  Model 3 — Patient Temporal LSTM:   age, sex_encoded + 10 CBC labs + length_of_stay
  Model 4 — Facility Temporal LSTM:  weekly aggregate features (facility-level, not per-claim)

HBA1C Models (future):
  Model 1 — Claim Autoencoder:       age, sex_encoded, HBA1C, CREATININE, UREA
  Model 2 — Disease Classifier:      HBA1C, CREATININE, UREA
  Model 3 — Patient Temporal LSTM:   age, sex_encoded, HBA1C, CREATININE, UREA
  Model 4 — Facility Temporal LSTM:  weekly aggregate features
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── CBC model feature requirements ───────────────────────────────────────────

CBC_LAB_FEATURES = ["HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT"]

CBC_MODEL_FEATURES: Dict[str, List[str]] = {
    "cbc_model1": ["age", "sex_encoded"] + CBC_LAB_FEATURES,
    "cbc_model2": CBC_LAB_FEATURES,
    "cbc_model3": ["age", "sex_encoded"] + CBC_LAB_FEATURES + ["length_of_stay"],
    # model4 is facility-level — validated separately via facility weekly sequence
}

# ── HBA1C model feature requirements ─────────────────────────────────────────

HBA1C_LAB_FEATURES = ["HBA1C", "CREATININE", "UREA"]

HBA1C_MODEL_FEATURES: Dict[str, List[str]] = {
    "hba1c_model1": ["age", "sex_encoded"] + HBA1C_LAB_FEATURES,
    "hba1c_model2": HBA1C_LAB_FEATURES,
    "hba1c_model3": ["age", "sex_encoded"] + HBA1C_LAB_FEATURES,
}

# ── Claim type detection ──────────────────────────────────────────────────────

def detect_claim_type(claim: Dict[str, Any]) -> str:
    """
    Detect whether a claim is CBC or HBA1C based on which lab fields are present.
    Returns 'cbc', 'hba1c', or 'unknown'.
    """
    has_cbc = any(
        claim.get(f) is not None and str(claim.get(f, "")).strip() not in ("", "null", "None")
        for f in CBC_LAB_FEATURES
    )
    has_hba1c = any(
        claim.get(f) is not None and str(claim.get(f, "")).strip() not in ("", "null", "None")
        for f in HBA1C_LAB_FEATURES
    )
    if has_cbc:
        return "cbc"
    if has_hba1c:
        return "hba1c"
    return "unknown"


# ── Per-model validation ──────────────────────────────────────────────────────

def _has_feature(claim: Dict[str, Any], feature: str) -> bool:
    """Return True if feature is present, non-null, and numeric."""
    val = claim.get(feature)
    if val is None:
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def validate_model_inputs(
    claim: Dict[str, Any],
    claim_type: str,
) -> Dict[str, Any]:
    """
    Check which models can be triggered for this claim.

    Returns a dict:
    {
        "claim_type": "cbc" | "hba1c" | "unknown",
        "models": {
            "model1": {"eligible": bool, "missing": [str]},
            "model2": {"eligible": bool, "missing": [str]},
            "model3": {"eligible": bool, "missing": [str]},
            "model4": {"eligible": False, "note": "facility-level, validated separately"},
        },
        "any_eligible": bool,
    }
    """
    if claim_type == "cbc":
        feature_map = CBC_MODEL_FEATURES
    elif claim_type == "hba1c":
        feature_map = HBA1C_MODEL_FEATURES
    else:
        return {
            "claim_type": "unknown",
            "models": {},
            "any_eligible": False,
        }

    models_status: Dict[str, Any] = {}
    any_eligible = False

    for model_key, required_features in feature_map.items():
        # strip prefix for output key: "cbc_model1" → "model1"
        out_key = model_key.split("_", 1)[1]
        missing = [f for f in required_features if not _has_feature(claim, f)]
        eligible = len(missing) == 0
        if eligible:
            any_eligible = True
        models_status[out_key] = {
            "eligible": eligible,
            "missing": missing,
            "required": required_features,
        }

    # Model 4 is always facility-level
    models_status["model4"] = {
        "eligible": False,
        "note": "facility-level model — validated via facility weekly sequence, not per-claim",
    }

    result = {
        "claim_type": claim_type,
        "models": models_status,
        "any_eligible": any_eligible,
    }

    logger.info(
        f"[model_validator] claim_id={claim.get('claim_id')} type={claim_type} "
        f"eligible_models={[k for k,v in models_status.items() if v.get('eligible')]}"
    )
    return result


def summarise_validation(validation: Dict[str, Any]) -> str:
    """Return a human-readable summary of model eligibility."""
    if not validation.get("any_eligible"):
        return (
            f"Claim type: {validation['claim_type']}. "
            "No models can be triggered — insufficient feature coverage."
        )
    eligible = [k for k, v in validation["models"].items() if v.get("eligible")]
    ineligible = {
        k: v["missing"]
        for k, v in validation["models"].items()
        if not v.get("eligible") and v.get("missing")
    }
    lines = [f"Claim type: {validation['claim_type']}"]
    lines.append(f"Eligible models: {', '.join(eligible)}")
    if ineligible:
        for model, missing in ineligible.items():
            lines.append(f"  {model} skipped — missing: {', '.join(missing)}")
    return "\n".join(lines)
