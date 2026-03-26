from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class VisitRecord(BaseModel):
    visit_index: int
    features: Dict[str, Any]
    reconstruction_error: Optional[float] = None


class PatientTrajectoryResponse(BaseModel):
    patient_id: str
    total_visits: int
    trajectory_anomaly_score: Optional[float] = None
    is_trajectory_anomaly: Optional[bool] = None
    most_anomalous_visit_index: Optional[int] = None
    insufficient_history: bool
    visits: List[VisitRecord]
    last_updated: datetime
