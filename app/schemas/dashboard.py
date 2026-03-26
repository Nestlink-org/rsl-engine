from typing import List, Optional
from pydantic import BaseModel


class TrendPoint(BaseModel):
    period: str
    total_claims: int
    flagged_claims: int
    avg_anomaly_score: float


class TopFacility(BaseModel):
    facility_id: str
    flagged_claims: int
    avg_anomaly_score: float
    high_severity_count: int = 0
    flag_rate: float = 0.0


class TopPatient(BaseModel):
    patient_id: str
    trajectory_anomaly_score: float
    is_trajectory_anomaly: bool
    total_visits: int
    most_anomalous_visit_index: Optional[int] = None


class DiseaseBreakdown(BaseModel):
    diagnosis: str
    category: str
    claim_count: int
    flagged_count: int
    avg_confidence: float
    flag_rate: float


class SeverityDistribution(BaseModel):
    high: int
    medium: int
    low: int
    high_pct: float
    medium_pct: float
    low_pct: float


class ModelPerformance(BaseModel):
    model_id: int
    model_name: str
    total_evaluated: int
    flagged: int
    flag_rate: float
    avg_anomaly_score: float


class DashboardMetricsResponse(BaseModel):
    period: str
    total_claims: int
    flagged_claims: int
    flag_rate: float
    avg_anomaly_score: float
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    severity_distribution: SeverityDistribution
    trend: List[TrendPoint]
    top_facilities: List[TopFacility]
    top_anomalous_patients: List[TopPatient]
    disease_breakdown: List[DiseaseBreakdown]
    model_performance: List[ModelPerformance]
