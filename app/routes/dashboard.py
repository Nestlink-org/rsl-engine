"""GET /api/v1/dashboard/metrics — comprehensive analytics dashboard."""

import logging
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import List

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.sessions import get_async_db
from app.models.claim import Claim
from app.models.facility_metric import FacilityWeeklyMetric
from app.models.fraud_flag import FraudFlag
from app.models.patient_trajectory import PatientTrajectory
from app.schemas.dashboard import (
    DashboardMetricsResponse,
    DiseaseBreakdown,
    ModelPerformance,
    SeverityDistribution,
    TopFacility,
    TopPatient,
    TrendPoint,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_NAMES = {
    1: "CBC Claim Autoencoder",
    2: "CBC Disease Classifier",
    3: "Patient Temporal LSTM",
    4: "Facility Temporal LSTM",
}


def _period_key(d: date, period: str) -> str:
    if period == "daily":
        return d.strftime("%Y-%m-%d")
    if period == "monthly":
        return d.strftime("%Y-%m")
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


def _period_cutoff(period: str) -> date:
    today = date.today()
    if period == "daily":
        return today - timedelta(days=11)
    if period == "monthly":
        month = today.month - 11
        year = today.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        return date(year, month, 1)
    return today - timedelta(weeks=11)


@router.get(
    "/dashboard/metrics",
    response_model=DashboardMetricsResponse,
    summary="Comprehensive analytics dashboard",
    description="""
Returns full analytics for the fraud detection system:

- **Trend data** — claim volume and flag rates over time (daily/weekly/monthly)
- **Top facilities** — ranked by flagged claims and avg anomaly score
- **Top anomalous patients** — ranked by trajectory anomaly score (Model 3)
- **Disease breakdown** — per-diagnosis claim counts, flag rates, and classifier confidence
- **Model performance** — per-model evaluation counts, flag rates, avg scores
- **Severity distribution** — high/medium/low breakdown with percentages

Admins see all data; regular users see only their own claims.
    """,
    tags=["Analytics"],
)
async def dashboard_metrics(
    request: Request,
    period: str = Query("weekly", pattern="^(daily|weekly|monthly)$"),
    session: AsyncSession = Depends(get_async_db),
):
    t0 = time.time()
    user_id = getattr(request.state, "user_id", None) or "anonymous"
    role = getattr(request.state, "role", "user")
    cutoff = _period_cutoff(period)

    claim_filter = [Claim.admission_date >= cutoff]
    if role != "admin":
        claim_filter.append(Claim.user_id == user_id)

    # ── Claims ────────────────────────────────────────────────────────────────
    claims_result = await session.execute(
        select(Claim.id, Claim.facility_id, Claim.admission_date, Claim.claimed_diagnosis)
        .where(*claim_filter)
    )
    claims_rows = claims_result.all()

    empty = DashboardMetricsResponse(
        period=period, total_claims=0, flagged_claims=0, flag_rate=0.0,
        avg_anomaly_score=0.0, high_severity_count=0, medium_severity_count=0,
        low_severity_count=0,
        severity_distribution=SeverityDistribution(high=0, medium=0, low=0, high_pct=0, medium_pct=0, low_pct=0),
        trend=[], top_facilities=[], top_anomalous_patients=[],
        disease_breakdown=[], model_performance=[],
    )
    if not claims_rows:
        return empty

    claim_ids = [r[0] for r in claims_rows]
    claim_date_map = {r[0]: r[2] for r in claims_rows}
    claim_diag_map = {r[0]: r[3] for r in claims_rows}

    # ── All fraud flags ───────────────────────────────────────────────────────
    flags_result = await session.execute(
        select(FraudFlag).where(FraudFlag.claim_id.in_(claim_ids))
    )
    all_flags = flags_result.scalars().all()

    # Group by model
    flags_by_model: dict = defaultdict(list)
    for f in all_flags:
        flags_by_model[f.model_id].append(f)

    # Model 1 flags for primary metrics
    m1_flags = flags_by_model.get(1, [])
    flag_by_claim_m1 = {f.claim_id: f for f in m1_flags}

    total = len(claim_ids)
    flagged = sum(1 for f in m1_flags if f.is_anomaly)
    avg_score = sum(f.anomaly_score for f in m1_flags) / max(len(m1_flags), 1)
    high = sum(1 for f in all_flags if f.severity == "high")
    med = sum(1 for f in all_flags if f.severity == "medium")
    low_sev = sum(1 for f in all_flags if f.severity == "low")
    total_flags = max(high + med + low_sev, 1)

    severity_dist = SeverityDistribution(
        high=high, medium=med, low=low_sev,
        high_pct=round(high / total_flags * 100, 1),
        medium_pct=round(med / total_flags * 100, 1),
        low_pct=round(low_sev / total_flags * 100, 1),
    )

    # ── Trend ─────────────────────────────────────────────────────────────────
    bucket_total: dict = defaultdict(int)
    bucket_flagged: dict = defaultdict(int)
    bucket_score_sum: dict = defaultdict(float)
    bucket_score_count: dict = defaultdict(int)

    for cid in claim_ids:
        adm = claim_date_map.get(cid)
        if not adm:
            continue
        key = _period_key(adm, period)
        bucket_total[key] += 1
        flag = flag_by_claim_m1.get(cid)
        if flag:
            if flag.is_anomaly:
                bucket_flagged[key] += 1
            bucket_score_sum[key] += flag.anomaly_score
            bucket_score_count[key] += 1

    trend = [
        TrendPoint(
            period=key,
            total_claims=bucket_total[key],
            flagged_claims=bucket_flagged.get(key, 0),
            avg_anomaly_score=round(
                bucket_score_sum.get(key, 0.0) / max(bucket_score_count.get(key, 1), 1), 4
            ),
        )
        for key in sorted(bucket_total.keys())
    ]

    # ── Top facilities ────────────────────────────────────────────────────────
    top_metrics_result = await session.execute(
        select(
            FacilityWeeklyMetric.facility_id,
            func.sum(FacilityWeeklyMetric.flagged_claims).label("total_flagged"),
            func.avg(FacilityWeeklyMetric.avg_anomaly_score).label("avg_score"),
            func.sum(FacilityWeeklyMetric.high_severity_count).label("high_sev"),
            func.sum(FacilityWeeklyMetric.claim_volume).label("vol"),
        )
        .where(FacilityWeeklyMetric.week_start_date >= cutoff)
        .group_by(FacilityWeeklyMetric.facility_id)
        .order_by(func.sum(FacilityWeeklyMetric.flagged_claims).desc())
        .limit(10)
    )
    top_facilities = [
        TopFacility(
            facility_id=r[0],
            flagged_claims=int(r[1] or 0),
            avg_anomaly_score=round(float(r[2] or 0), 4),
            high_severity_count=int(r[3] or 0),
            flag_rate=round(int(r[1] or 0) / max(int(r[4] or 1), 1), 4),
        )
        for r in top_metrics_result.all()
    ]

    # ── Top anomalous patients (Model 3) ──────────────────────────────────────
    patients_result = await session.execute(
        select(PatientTrajectory)
        .where(PatientTrajectory.is_trajectory_anomaly == True)
        .order_by(PatientTrajectory.trajectory_anomaly_score.desc())
        .limit(10)
    )
    top_patients = [
        TopPatient(
            patient_id=p.patient_id,
            trajectory_anomaly_score=round(p.trajectory_anomaly_score or 0.0, 4),
            is_trajectory_anomaly=bool(p.is_trajectory_anomaly),
            total_visits=len(__import__("json").loads(p.visit_sequence or "[]")),
            most_anomalous_visit_index=p.most_anomalous_visit_index,
        )
        for p in patients_result.scalars().all()
    ]

    # ── Disease breakdown (Model 2) ───────────────────────────────────────────
    m2_flags = flags_by_model.get(2, [])
    diag_stats: dict = defaultdict(lambda: {"count": 0, "flagged": 0, "conf_sum": 0.0, "cat": ""})
    for f in m2_flags:
        diag = f.predicted_diagnosis or "Unknown"
        cat = f.predicted_category or "Unknown"
        cid = f.claim_id
        diag_stats[diag]["count"] += 1
        diag_stats[diag]["cat"] = cat
        if f.is_anomaly:
            diag_stats[diag]["flagged"] += 1
        diag_stats[diag]["conf_sum"] += float(f.diagnosis_confidence or 0)

    disease_breakdown = sorted(
        [
            DiseaseBreakdown(
                diagnosis=diag,
                category=stats["cat"],
                claim_count=stats["count"],
                flagged_count=stats["flagged"],
                avg_confidence=round(stats["conf_sum"] / max(stats["count"], 1), 4),
                flag_rate=round(stats["flagged"] / max(stats["count"], 1), 4),
            )
            for diag, stats in diag_stats.items()
        ],
        key=lambda x: x.flagged_count,
        reverse=True,
    )

    # ── Model performance ─────────────────────────────────────────────────────
    model_performance = []
    for mid, mname in MODEL_NAMES.items():
        mflags = flags_by_model.get(mid, [])
        if not mflags:
            continue
        mflagged = sum(1 for f in mflags if f.is_anomaly)
        mavg = sum(f.anomaly_score for f in mflags) / len(mflags)
        model_performance.append(ModelPerformance(
            model_id=mid,
            model_name=mname,
            total_evaluated=len(mflags),
            flagged=mflagged,
            flag_rate=round(mflagged / max(len(mflags), 1), 4),
            avg_anomaly_score=round(mavg, 4),
        ))

    logger.info(
        f"[dashboard] done {time.time()-t0:.3f}s total={total} flagged={flagged} "
        f"diseases={len(disease_breakdown)} top_patients={len(top_patients)}"
    )

    return DashboardMetricsResponse(
        period=period, total_claims=total, flagged_claims=flagged,
        flag_rate=round(flagged / max(total, 1), 4),
        avg_anomaly_score=round(avg_score, 4),
        high_severity_count=high, medium_severity_count=med, low_severity_count=low_sev,
        severity_distribution=severity_dist,
        trend=trend,
        top_facilities=top_facilities,
        top_anomalous_patients=top_patients,
        disease_breakdown=disease_breakdown,
        model_performance=model_performance,
    )
