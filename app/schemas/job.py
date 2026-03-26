from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    total_claims: int
    processed_claims: int
    failed_claims: int
    error_detail: Optional[str] = None


class FraudFlagOut(BaseModel):
    model_id: int
    anomaly_score: float
    is_anomaly: bool
    severity: str
    flag_reason: str
    predicted_category: Optional[str] = None
    predicted_diagnosis: Optional[str] = None
    category_confidence: Optional[float] = None
    diagnosis_confidence: Optional[float] = None
    insufficient_history: Optional[bool] = None


class ClaimResultItem(BaseModel):
    claim_id: str
    patient_id: str
    facility_id: str
    admission_date: date
    discharge_date: date
    claimed_diagnosis: str
    fraud_flags: List[FraudFlagOut]


class JobResultsResponse(BaseModel):
    job_id: str
    status: str
    page: int
    page_size: int
    total_claims: int
    claims: List[ClaimResultItem]
