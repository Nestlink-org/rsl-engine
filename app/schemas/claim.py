from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel

from app.schemas.job import FraudFlagOut


class CBCDataOut(BaseModel):
    age: float
    sex_encoded: int
    HGB: float
    HCT: float
    MCV: float
    MCHC: float
    NEU: float
    LYM: float
    EOS: float
    BAS: float
    MON: float
    PLT: float
    length_of_stay: float


class ClaimDetailResponse(BaseModel):
    claim_id: str
    job_id: str
    patient_id: str
    facility_id: str
    admission_date: date
    discharge_date: date
    claimed_diagnosis: str
    created_at: datetime
    cbc_data: Optional[CBCDataOut] = None
    fraud_flags: List[FraudFlagOut] = []
