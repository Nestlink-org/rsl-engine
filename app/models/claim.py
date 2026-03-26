from datetime import date, datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Claim(SQLModel, table=True):
    __tablename__ = "claim"

    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: str = Field(unique=True, index=True)
    job_id: str = Field(index=True)
    user_id: str = Field(index=True)
    patient_id: str = Field(index=True)
    facility_id: str = Field(index=True)
    admission_date: date
    discharge_date: date
    claimed_diagnosis: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CBCData(SQLModel, table=True):
    __tablename__ = "cbcdata"

    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: int = Field(foreign_key="claim.id", index=True)
    age: float
    sex_encoded: int                  # 0=Female, 1=Male
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
