from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class PatientTrajectory(SQLModel, table=True):
    __tablename__ = "patienttrajectory"

    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: str = Field(unique=True, index=True)
    # JSON-encoded list[dict] of up to 5 visit feature dicts, oldest→newest
    visit_sequence: str = Field(default="[]")
    trajectory_anomaly_score: Optional[float] = Field(default=None)
    is_trajectory_anomaly: Optional[bool] = Field(default=None)
    # JSON-encoded list[float] of per-visit reconstruction errors
    per_visit_errors: Optional[str] = Field(default=None)
    most_anomalous_visit_index: Optional[int] = Field(default=None)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
