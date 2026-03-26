from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class FraudFlag(SQLModel, table=True):
    __tablename__ = "fraudflag"

    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: int = Field(foreign_key="claim.id", index=True)
    model_id: int                                          # 1, 2, 3, or 4
    anomaly_score: float                                   # 0.0–1.0
    is_anomaly: bool
    severity: str                                          # low | medium | high
    flag_reason: str
    # Model 2 fields
    predicted_category: Optional[str] = Field(default=None)   # obstetric|respiratory|trauma
    predicted_diagnosis: Optional[str] = Field(default=None)
    category_confidence: Optional[float] = Field(default=None)
    diagnosis_confidence: Optional[float] = Field(default=None)
    # Model 3 fields
    insufficient_history: Optional[bool] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
