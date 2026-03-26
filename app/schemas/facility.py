from datetime import date
from typing import List, Optional
from pydantic import BaseModel


class WeeklyMetricOut(BaseModel):
    week_start_date: date
    claim_volume: int
    avg_anomaly_score: float
    flagged_claims: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int


class FacilityRiskResponse(BaseModel):
    facility_id: str
    weeks: int
    total_claims: int
    flagged_claims: int
    flag_rate: float
    avg_anomaly_score: float
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    weekly_metrics: List[WeeklyMetricOut]
    model4_available: bool = False
    model4_anomaly_score: Optional[float] = None
    model4_is_anomaly: Optional[bool] = None
    model4_flag_reason: Optional[str] = None
