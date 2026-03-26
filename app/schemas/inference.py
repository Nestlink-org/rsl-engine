"""Pydantic schemas for the agent inference endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Model validation ──────────────────────────────────────────────────────────

class ModelEligibility(BaseModel):
    eligible: bool
    missing: List[str] = []
    required: List[str] = []
    note: Optional[str] = None


class ModelValidationResult(BaseModel):
    claim_type: str = Field(description="'cbc', 'hba1c', or 'unknown'")
    models: Dict[str, ModelEligibility]
    any_eligible: bool


# ── Structured claim (preview, no DB write) ───────────────────────────────────

class StructuredClaimPreview(BaseModel):
    claim_id: Optional[str] = None
    patient_id: Optional[str] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    sex_encoded: Optional[int] = None
    facility_id: Optional[str] = None
    facility_name: Optional[str] = None
    facility_type: Optional[str] = None
    facility_level: Optional[str] = None
    disease_category: Optional[str] = None
    diagnosis: Optional[str] = None
    claimed_diagnosis: Optional[str] = None  # canonical name after column mapping
    diagnosis_code: Optional[str] = None
    procedure: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    date: Optional[str] = None
    timestamp_processed: Optional[str] = None
    is_fraud: Optional[Any] = None
    # CBC labs
    HGB: Optional[float] = None
    HCT: Optional[float] = None
    MCV: Optional[float] = None
    MCHC: Optional[float] = None
    NEU: Optional[float] = None
    LYM: Optional[float] = None
    EOS: Optional[float] = None
    BAS: Optional[float] = None
    MON: Optional[float] = None
    PLT: Optional[float] = None
    length_of_stay: Optional[float] = None
    # HBA1C labs
    HBA1C: Optional[float] = None
    CREATININE: Optional[float] = None
    UREA: Optional[float] = None
    # Agent metadata (internal keys stripped of leading underscore)
    claim_type: Optional[str] = None
    validation_summary: Optional[str] = None
    model_validation: Optional[ModelValidationResult] = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class StructurePreviewResponse(BaseModel):
    """Response for dry-run structuring — no DB write, no inference.

    NOTE: preview_id is NOT a job_id. It cannot be used with /jobs/ or /agent/queue/.
    To process claims and track via jobs, use POST /api/v1/upload instead.
    """
    preview_id: str = Field(description="Temporary ID for this preview session only — not a job ID")
    source: str = Field(description="'ocr_text', 'csv', or 'excel'")
    total_claims: int = Field(description="Total claims found in file")
    claims_shown: int = Field(description="Claims returned in this response (all claims for preview)")
    cbc_claims: int
    hba1c_claims: int
    unknown_claims: int
    model_eligible_counts: Dict[str, int]
    claims: List[StructuredClaimPreview]
    processing_time: float
    note: str = Field(
        default="This is a dry-run preview. No data was saved. Use POST /api/v1/upload to process and persist claims."
    )


# ── Model result ──────────────────────────────────────────────────────────────

class ModelResultOut(BaseModel):
    model_id: int
    model_name: str
    anomaly_score: float
    is_anomaly: bool
    severity: str
    flag_reason: str
    # Model 1 extras
    mse: Optional[float] = None
    top_features: Optional[List[Dict[str, Any]]] = None
    # Model 2 extras
    predicted_category: Optional[str] = None
    predicted_diagnosis: Optional[str] = None
    category_confidence: Optional[float] = None
    diagnosis_confidence: Optional[float] = None
    mismatch: Optional[bool] = None
    # Model 3 extras
    trajectory_anomaly_score: Optional[float] = None
    per_visit_errors: Optional[List[float]] = None
    most_anomalous_visit_index: Optional[int] = None
    insufficient_history: Optional[bool] = None
    # Model 4 extras
    facility_anomaly_score: Optional[float] = None
    per_week_errors: Optional[List[float]] = None
    most_anomalous_week_index: Optional[int] = None


class ClaimInferenceResult(BaseModel):
    claim_id: str
    status: str = Field(description="'processed', 'skipped', or 'failed'")
    any_anomaly: Optional[bool] = None
    max_anomaly_score: Optional[float] = None
    models_run: List[str] = []
    model_results: Dict[str, Any] = {}
    reason: Optional[str] = None


# ── Pipeline response ─────────────────────────────────────────────────────────

class PipelineResponse(BaseModel):
    """Response for full pipeline runs (file → structure → infer → DB)."""
    job_id: str
    status: str
    filename: str
    file_type: str
    total_processed: int
    total_failed: int
    queued_count: int
    models_triggered: Dict[str, int]
    results: List[ClaimInferenceResult]
    failed_details: List[Dict[str, Any]]
    processing_time: float
    poll_url: str = Field(description="URL to poll for job status")


# ── Batch queue status ────────────────────────────────────────────────────────

class BatchQueueStatus(BaseModel):
    job_id: str
    queued_claims: int
    batch_size: int
    message: str
