from datetime import date
from pydantic import BaseModel, model_validator


class ROIReportRequest(BaseModel):
    start_date: date
    end_date: date
    avg_claim_value_kes: float = 50000.0
    recovery_rate: float = 0.7          # fraction of flagged fraud actually recovered

    @model_validator(mode="after")
    def validate_date_range(self) -> "ROIReportRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
        return self


class ROIReportResponse(BaseModel):
    start_date: date
    end_date: date
    total_claims: int
    flagged_claims: int
    flag_rate: float
    avg_claim_value_kes: float
    estimated_fraud_amount_kes: float
    potential_savings_kes: float      # estimated_fraud_amount * 0.7 (recovery rate)
    roi_percentage: float
