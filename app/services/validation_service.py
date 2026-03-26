"""Claim validation — checks required fields, ranges, dates, duplicates, and sex encoding."""

from datetime import date
from typing import Optional, Set, Tuple

REQUIRED_CBC_FIELDS = [
    "claim_id", "patient_id", "facility_id", "admission_date", "discharge_date",
    "claimed_diagnosis", "age", "sex", "HGB", "HCT", "MCV", "MCHC",
    "NEU", "LYM", "EOS", "BAS", "MON", "PLT", "length_of_stay",
]

CBC_RANGES = {
    "HGB":  (1.0, 25.0),
    "HCT":  (5.0, 70.0),
    "MCV":  (50.0, 150.0),
    "MCHC": (20.0, 40.0),
    "NEU":  (0.0, 100.0),
    "LYM":  (0.0, 100.0),
    "EOS":  (0.0, 50.0),
    "BAS":  (0.0, 10.0),
    "MON":  (0.0, 30.0),
    "PLT":  (10.0, 1500.0),
    "age":  (0.0, 120.0),
}

VALID_SEX_VALUES = {"Male", "Female", "1", "0"}


def validate_claim(
    claim_dict: dict,
    existing_claim_ids: Set[str],
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Validate a single claim dict.

    Returns:
        (is_valid, reason, enriched_dict)
        enriched_dict includes sex_encoded and parsed dates when valid.
    """
    # 1. Required fields present
    missing = [f for f in REQUIRED_CBC_FIELDS if f not in claim_dict or claim_dict[f] is None or str(claim_dict[f]).strip() == ""]
    if missing:
        return False, f"missing_fields:{','.join(missing)}", None

    # 2. Duplicate claim_id
    if str(claim_dict["claim_id"]) in existing_claim_ids:
        return False, "duplicate_claim_id", None

    # 3. Sex value
    sex_val = str(claim_dict["sex"]).strip()
    if sex_val not in VALID_SEX_VALUES:
        return False, "invalid_sex_value", None
    sex_encoded = 1 if sex_val in ("Male", "1") else 0

    # 4. Numeric ranges
    for field, (lo, hi) in CBC_RANGES.items():
        try:
            val = float(claim_dict[field])
        except (ValueError, TypeError):
            return False, f"non_numeric:{field}", None
        if not (lo <= val <= hi):
            return False, f"out_of_range:{field}={val}", None

    # 5. Date range
    try:
        adm = claim_dict["admission_date"]
        dis = claim_dict["discharge_date"]
        if not isinstance(adm, date):
            from datetime import datetime
            adm = datetime.strptime(str(adm), "%Y-%m-%d").date()
        if not isinstance(dis, date):
            from datetime import datetime
            dis = datetime.strptime(str(dis), "%Y-%m-%d").date()
    except Exception:
        return False, "invalid_date_format", None

    if adm > dis:
        return False, "invalid_date_range", None

    enriched = {**claim_dict, "sex_encoded": sex_encoded, "admission_date": adm, "discharge_date": dis}
    return True, None, enriched
