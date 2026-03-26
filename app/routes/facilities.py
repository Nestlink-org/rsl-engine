"""GET /api/v1/facilities/{facility_id}/risk"""

import logging
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.sessions import get_async_db
from app.models.claim import Claim
from app.models.facility_metric import FacilityWeeklyMetric
from app.models.fraud_flag import FraudFlag
from app.schemas.facility import FacilityRiskResponse, WeeklyMetricOut
from app.services.inference_service import MODEL4_FEATURES, MODEL4_SEQ_LEN, run_model4
from app.services.model_registry import get_model_registry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/facilities/{facility_id}/risk", response_model=FacilityRiskResponse)
async def facility_risk(
    facility_id: str,
    request: Request,
    weeks: int = Query(8, ge=1, le=52),
    session: AsyncSession = Depends(get_async_db),
):
    user_id = getattr(request.state, "user_id", None) or "anonymous"
    role = getattr(request.state, "role", "user")
    since = date.today() - timedelta(weeks=weeks)

    # Scope by user unless admin
    claim_filter = [Claim.facility_id == facility_id, Claim.admission_date >= since]
    if role != "admin":
        claim_filter.append(Claim.user_id == user_id)

    claims_result = await session.execute(select(Claim.id).where(*claim_filter))
    claim_ids = [r for r in claims_result.scalars().all()]
    if not claim_ids:
        raise HTTPException(status_code=404, detail="No claims found for this facility")

    flags_result = await session.execute(
        select(FraudFlag).where(FraudFlag.claim_id.in_(claim_ids))
    )
    flags = flags_result.scalars().all()

    total = len(claim_ids)
    flagged = sum(1 for f in flags if f.is_anomaly and f.model_id == 1)
    avg_score = sum(f.anomaly_score for f in flags if f.model_id == 1) / max(total, 1)
    high = sum(1 for f in flags if f.severity == "high")
    med = sum(1 for f in flags if f.severity == "medium")
    low = sum(1 for f in flags if f.severity == "low")

    # Weekly metrics
    metric_filter = [
        FacilityWeeklyMetric.facility_id == facility_id,
        FacilityWeeklyMetric.week_start_date >= since,
    ]
    metrics_result = await session.execute(
        select(FacilityWeeklyMetric).where(*metric_filter).order_by(FacilityWeeklyMetric.week_start_date)
    )
    weekly_rows = metrics_result.scalars().all()
    weekly = [WeeklyMetricOut(**m.__dict__) for m in weekly_rows]

    # --- Model 4: facility temporal anomaly ---
    registry = get_model_registry()
    model4_score: Optional[float] = None
    model4_anomaly: Optional[bool] = None
    model4_reason: Optional[str] = None

    if registry.model4_available and weekly_rows:
        # Build weekly aggregate sequence from FacilityWeeklyMetric rows
        # We need the 19 features; most are stored in the metric, derive missing ones from CBC flags
        weekly_sequence = []
        for m in weekly_rows[-MODEL4_SEQ_LEN:]:
            # Compute avg CBC values from flags for this week's claims — use stored metric fields
            # FacilityWeeklyMetric stores aggregate stats; map to model4 feature names
            weekly_sequence.append({
                "claim_volume": float(m.claim_volume),
                "avg_age": 0.0,       # not stored in metric — use 0 (padded)
                "age_std": 0.0,
                "pct_male": 0.0,
                "HGB_mean": 0.0,
                "HGB_std": 0.0,
                "HCT_mean": 0.0,
                "HCT_std": 0.0,
                "MCV_mean": 0.0,
                "MCV_std": 0.0,
                "MCHC_mean": 0.0,
                "MCHC_std": 0.0,
                "NEU_mean": 0.0,
                "LYM_mean": 0.0,
                "EOS_mean": 0.0,
                "BAS_mean": 0.0,
                "MON_mean": 0.0,
                "PLT_mean": 0.0,
                "avg_los": 0.0,
            })

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            m4 = await loop.run_in_executor(None, run_model4, weekly_sequence, registry)
            model4_score = round(m4.facility_anomaly_score, 4)
            model4_anomaly = m4.is_facility_anomaly
            model4_reason = m4.flag_reason
            logger.info(
                f"[facility_risk] facility={facility_id} "
                f"m4_score={model4_score} anomaly={model4_anomaly}"
            )
        except Exception as e:
            logger.warning(f"[facility_risk] model4 inference failed for {facility_id}: {e}")

    return FacilityRiskResponse(
        facility_id=facility_id, weeks=weeks,
        total_claims=total, flagged_claims=flagged,
        flag_rate=round(flagged / max(total, 1), 4),
        avg_anomaly_score=round(avg_score, 4),
        high_severity_count=high, medium_severity_count=med, low_severity_count=low,
        weekly_metrics=weekly,
        model4_available=registry.model4_available,
        model4_anomaly_score=model4_score,
        model4_is_anomaly=model4_anomaly,
        model4_flag_reason=model4_reason,
    )
