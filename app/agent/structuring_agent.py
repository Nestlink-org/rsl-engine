"""Structuring Agent — Phase 1 of the RSL multi-agent pipeline.

Responsibilities:
  1. Accept input from three sources:
       a. OCR full_text (string) — from the OCR engine
       b. CSV file path — structured tabular data
       c. Excel file path — structured tabular data
  2. Use gpt-5.4-nano to extract and normalise claim fields from OCR text.
  3. For CSV/Excel: map column names to canonical field names, no LLM needed.
  4. Output a list of StructuredClaim dicts — one per claim row/document.
  5. For each claim, run model_validator to determine which models can be triggered.
  6. Enforce batch size limit (AGENT_BATCH_SIZE=5). Excess claims queued in Redis.
  7. Log every step with structured logging.

CBC canonical fields:
  claim_id, patient_id, age, sex, sex_encoded, facility_id, facility_name,
  facility_type, facility_level, disease_category, diagnosis, diagnosis_code,
  procedure, admission_date, discharge_date, date, timestamp_processed,
  is_fraud, HGB, HCT, MCV, MCHC, NEU, LYM, EOS, BAS, MON, PLT, length_of_stay

HBA1C canonical fields (same metadata + HBA1C, CREATININE, UREA instead of CBC labs)
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.agent.config import get_llm
from app.agent.model_validator import detect_claim_type, summarise_validation, validate_model_inputs
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Canonical field definitions ───────────────────────────────────────────────

# Metadata fields shared by both CBC and HBA1C claims
METADATA_FIELDS = [
    "claim_id", "patient_id", "age", "sex", "facility_id", "facility_name",
    "facility_type", "facility_level", "disease_category", "claimed_diagnosis",
    "diagnosis_code", "procedure", "admission_date", "discharge_date",
    "date", "timestamp_processed", "is_fraud",
]

CBC_LAB_FIELDS = ["HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT"]
HBA1C_LAB_FIELDS = ["HBA1C", "CREATININE", "UREA"]

CBC_FIELDS = METADATA_FIELDS + CBC_LAB_FIELDS + ["length_of_stay"]
HBA1C_FIELDS = METADATA_FIELDS + HBA1C_LAB_FIELDS

# Column name aliases — maps common variations to canonical names
COLUMN_ALIASES: Dict[str, str] = {
    # metadata
    "claimid": "claim_id", "claim id": "claim_id",
    "patientid": "patient_id", "patient id": "patient_id", "patient": "patient_id",
    "facilityid": "facility_id", "facility id": "facility_id",
    "facilityname": "facility_name", "facility name": "facility_name",
    "facilitytype": "facility_type", "facility type": "facility_type",
    "facilitylevel": "facility_level", "facility level": "facility_level",
    "diseasecategory": "disease_category", "disease category": "disease_category",
    # diagnosis — map all variants to claimed_diagnosis (what validation expects)
    "diagnosis": "claimed_diagnosis",
    "claimed_diagnosis": "claimed_diagnosis",
    "diagnosiscode": "diagnosis_code", "diagnosis code": "diagnosis_code",
    "icd": "diagnosis_code", "icd_code": "diagnosis_code",
    "admissiondate": "admission_date", "admission date": "admission_date", "admit_date": "admission_date",
    "dischargedate": "discharge_date", "discharge date": "discharge_date",
    "timestampprocessed": "timestamp_processed",
    "isfraud": "is_fraud", "fraud": "is_fraud",
    "lengthofstay": "length_of_stay", "length of stay": "length_of_stay", "los": "length_of_stay",
    # CBC labs
    "haemoglobin": "HGB", "hemoglobin": "HGB", "hgb": "HGB", "hb": "HGB",
    "haematocrit": "HCT", "hematocrit": "HCT", "hct": "HCT", "pcv": "HCT",
    "mcv": "MCV", "mchc": "MCHC", "mch": "MCHC",
    "neutrophils": "NEU", "neu": "NEU", "neut": "NEU",
    "lymphocytes": "LYM", "lym": "LYM", "lymph": "LYM",
    "eosinophils": "EOS", "eos": "EOS",
    "basophils": "BAS", "bas": "BAS",
    "monocytes": "MON", "mon": "MON", "mono": "MON",
    "platelets": "PLT", "plt": "PLT", "platelet": "PLT", "platelet count": "PLT",
    # HBA1C labs
    "hba1c": "HBA1C", "hba1c_value": "HBA1C", "glycated haemoglobin": "HBA1C",
    "creatinine": "CREATININE", "creat": "CREATININE",
    "urea": "UREA", "bun": "UREA",
}

# ── LLM extraction prompt ─────────────────────────────────────────────────────

OCR_EXTRACTION_PROMPT = """You are a medical insurance claim data extraction specialist.

Extract ALL medical claims from the OCR text below. Each claim must be returned as a JSON object.

For CBC (Complete Blood Count) claims, use these fields:
{cbc_fields}

For HBA1C / Diabetes claims, use these fields:
{hba1c_fields}

Rules:
- Return a JSON array. Each element is one claim.
- If a field cannot be found in the text, set it to null — do NOT guess.
- sex must be exactly "Male" or "Female" or null.
- Dates must be in YYYY-MM-DD format or null.
- All numeric lab values must be numbers (not strings).
- claim_id: if not found, generate a UUID string.
- length_of_stay: calculate from admission_date and discharge_date if both present, else null.
- Do NOT include markdown, explanation, or any text outside the JSON array.

OCR Text:
{text}
"""


# ── Column normalisation ──────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to canonical names using COLUMN_ALIASES."""
    rename_map = {}
    for col in df.columns:
        normalised = col.strip().lower().replace("-", "_").replace(" ", "_")
        # try exact alias match first
        canonical = COLUMN_ALIASES.get(normalised) or COLUMN_ALIASES.get(col.strip().lower())
        if canonical:
            rename_map[col] = canonical
        else:
            # keep as-is but strip whitespace
            rename_map[col] = col.strip()
    return df.rename(columns=rename_map)


def _row_to_claim(row: pd.Series) -> Dict[str, Any]:
    """Convert a DataFrame row to a claim dict, filling missing fields with None."""
    claim: Dict[str, Any] = {}
    all_fields = set(CBC_FIELDS + HBA1C_FIELDS)
    for field in all_fields:
        val = row.get(field)
        if pd.isna(val) if not isinstance(val, (list, dict)) else False:
            claim[field] = None
        else:
            claim[field] = val
    # also carry any extra columns
    for col in row.index:
        if col not in claim:
            claim[col] = row[col]
    return claim


# ── Sex encoding ──────────────────────────────────────────────────────────────

def _encode_sex(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Add sex_encoded field (1=Male, 0=Female) if sex is present."""
    sex = str(claim.get("sex", "")).strip()
    if sex in ("Male", "1"):
        claim["sex_encoded"] = 1
    elif sex in ("Female", "0"):
        claim["sex_encoded"] = 0
    else:
        claim["sex_encoded"] = None
    return claim


# ── Length of stay calculation ────────────────────────────────────────────────

def _calc_los(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate length_of_stay from dates if not already present."""
    if claim.get("length_of_stay") is not None:
        return claim
    try:
        adm = claim.get("admission_date")
        dis = claim.get("discharge_date")
        if adm and dis:
            if not isinstance(adm, datetime):
                adm = datetime.strptime(str(adm)[:10], "%Y-%m-%d")
            if not isinstance(dis, datetime):
                dis = datetime.strptime(str(dis)[:10], "%Y-%m-%d")
            claim["length_of_stay"] = max(1.0, float((dis - adm).days))
    except Exception:
        pass
    return claim


# ── Redis batch queue ─────────────────────────────────────────────────────────

def _get_redis():
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"[structuring_agent] Redis unavailable: {e}")
        return None


def queue_overflow_claims(job_id: str, claims: List[Dict[str, Any]]) -> int:
    """
    Push overflow claims (beyond AGENT_BATCH_SIZE) to Redis list for later processing.
    Returns number of claims queued.
    """
    r = _get_redis()
    if r is None or not claims:
        return 0
    key = f"rsl:batch_queue:{job_id}"
    pipe = r.pipeline()
    for claim in claims:
        pipe.rpush(key, json.dumps(claim, default=str))
    pipe.expire(key, 86400)  # 24h TTL
    pipe.execute()
    logger.info(f"[structuring_agent] queued {len(claims)} overflow claims key={key}")
    return len(claims)


def dequeue_batch(job_id: str, batch_size: int = None) -> List[Dict[str, Any]]:
    """Pop up to batch_size claims from the Redis queue for this job."""
    batch_size = batch_size or settings.AGENT_BATCH_SIZE
    r = _get_redis()
    if r is None:
        return []
    key = f"rsl:batch_queue:{job_id}"
    pipe = r.pipeline()
    for _ in range(batch_size):
        pipe.lpop(key)
    results = pipe.execute()
    claims = [json.loads(r) for r in results if r is not None]
    logger.info(f"[structuring_agent] dequeued {len(claims)} claims from key={key}")
    return claims


def get_queue_length(job_id: str) -> int:
    """Return number of claims still queued in Redis for this job."""
    r = _get_redis()
    if r is None:
        return 0
    return r.llen(f"rsl:batch_queue:{job_id}") or 0


# ── Main structuring functions ────────────────────────────────────────────────

def structure_from_ocr(full_text: str, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use gpt-5.4-nano to extract structured claims from OCR full_text.

    Returns list of claim dicts with model_validation attached.
    """
    t0 = time.time()
    job_id = job_id or str(uuid.uuid4())
    logger.info(f"[structuring_agent] OCR structuring START job={job_id} text_len={len(full_text)}")

    llm = get_llm(temperature=0)
    prompt = OCR_EXTRACTION_PROMPT.format(
        cbc_fields=", ".join(CBC_FIELDS),
        hba1c_fields=", ".join(HBA1C_FIELDS),
        text=full_text,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        raw_claims = json.loads(content)
        if isinstance(raw_claims, dict):
            raw_claims = [raw_claims]

    except json.JSONDecodeError as e:
        logger.error(f"[structuring_agent] LLM returned invalid JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"[structuring_agent] LLM call failed: {e}", exc_info=True)
        return []

    claims = []
    for raw in raw_claims:
        claim = _encode_sex(raw)
        claim = _calc_los(claim)
        if not claim.get("claim_id"):
            claim["claim_id"] = str(uuid.uuid4())
        if not claim.get("timestamp_processed"):
            claim["timestamp_processed"] = datetime.utcnow().isoformat()
        claim_type = detect_claim_type(claim)
        claim["_claim_type"] = claim_type
        claim["_model_validation"] = validate_model_inputs(claim, claim_type)
        claim["_validation_summary"] = summarise_validation(claim["_model_validation"])
        claims.append(claim)

    logger.info(
        f"[structuring_agent] OCR structuring DONE job={job_id} "
        f"claims={len(claims)} elapsed={time.time()-t0:.3f}s"
    )
    return claims


def structure_from_file(
    file_path: str,
    job_id: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load CSV or Excel file, normalise columns, and structure claims.

    Enforces AGENT_BATCH_SIZE — first N claims returned, remainder queued in Redis.

    Returns:
        (batch_claims, queued_count)
        batch_claims: up to batch_size structured claims ready for model input
        queued_count: number of overflow claims pushed to Redis
    """
    t0 = time.time()
    job_id = job_id or str(uuid.uuid4())
    batch_size = batch_size or settings.AGENT_BATCH_SIZE
    logger.info(f"[structuring_agent] file structuring START job={job_id} file={file_path}")

    # Load file
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    except Exception as e:
        logger.error(f"[structuring_agent] file load failed: {e}", exc_info=True)
        return [], 0

    logger.info(f"[structuring_agent] loaded {len(df)} rows columns={list(df.columns)}")

    # Normalise columns
    df = _normalise_columns(df)
    logger.info(f"[structuring_agent] normalised columns={list(df.columns)}")

    # Convert rows to claim dicts
    all_claims: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        claim = _row_to_claim(row)
        claim = _encode_sex(claim)
        claim = _calc_los(claim)
        if not claim.get("claim_id"):
            claim["claim_id"] = str(uuid.uuid4())
        if not claim.get("timestamp_processed"):
            claim["timestamp_processed"] = datetime.utcnow().isoformat()
        claim_type = detect_claim_type(claim)
        claim["_claim_type"] = claim_type
        claim["_model_validation"] = validate_model_inputs(claim, claim_type)
        claim["_validation_summary"] = summarise_validation(claim["_model_validation"])
        all_claims.append(claim)

    # Split into batch + overflow
    batch = all_claims[:batch_size]
    overflow = all_claims[batch_size:]
    queued = queue_overflow_claims(job_id, overflow) if overflow else 0

    elapsed = time.time() - t0
    logger.info(
        f"[structuring_agent] file structuring DONE job={job_id} "
        f"total={len(all_claims)} batch={len(batch)} queued={queued} elapsed={elapsed:.3f}s"
    )
    return batch, queued


def get_structuring_summary(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a summary of structured claims — types, model eligibility counts.
    Useful for logging and API responses.
    """
    cbc_count = sum(1 for c in claims if c.get("_claim_type") == "cbc")
    hba1c_count = sum(1 for c in claims if c.get("_claim_type") == "hba1c")
    unknown_count = sum(1 for c in claims if c.get("_claim_type") == "unknown")

    model_eligible: Dict[str, int] = {}
    for claim in claims:
        mv = claim.get("_model_validation", {})
        for model_key, status in mv.get("models", {}).items():
            if status.get("eligible"):
                model_eligible[model_key] = model_eligible.get(model_key, 0) + 1

    return {
        "total_claims": len(claims),
        "cbc_claims": cbc_count,
        "hba1c_claims": hba1c_count,
        "unknown_claims": unknown_count,
        "model_eligible_counts": model_eligible,
    }
