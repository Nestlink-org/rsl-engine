from datetime import date
from typing import Optional
from sqlmodel import SQLModel, Field


class FacilityWeeklyMetric(SQLModel, table=True):
    __tablename__ = "facilityweeklymetric"

    id: Optional[int] = Field(default=None, primary_key=True)
    facility_id: str = Field(index=True)
    week_start_date: date = Field(index=True)
    claim_volume: int = Field(default=0)
    avg_anomaly_score: float = Field(default=0.0)
    flagged_claims: int = Field(default=0)
    high_severity_count: int = Field(default=0)
    medium_severity_count: int = Field(default=0)
    low_severity_count: int = Field(default=0)
