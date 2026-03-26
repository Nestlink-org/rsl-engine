"""DB persistence helpers for the agent pipeline.

Handles writing Claim, CBCData, FraudFlag, PatientTrajectory, FacilityWeeklyMetric
from structured + inferred claim data. Works for both CSV and OCR sources.
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


async def persist_claim_results(
    session: AsyncSession,
    claim: Dict[str, Any],
    model_results: Dict[str, Any],
    job_id: str,
    user_id: str,
    existing_ids: set,
) -> Dict[str, Any]:
    """
    Persist one claim and its model results to the database.

    Args:
        session: async DB session
        claim: validated + enriched claim dict (from structuring + validation nodes)
        model_results: dict keyed by "model1".."model4" with tool output dicts
        job_id: parent job UUID
        user_id: submitting user
        existing_ids: set of already-persisted claim_ids (updated in-place)

    Returns:
        {"claim_id": str, "status": "persisted"|"skipped"|"failed", "reason"?: str}
    """
    from sqlalchemy import select
    from app.models.claim import CBCData, Claim
    from app.models.fraud_flag import FraudFlag
    from app.models.patient_trajectory import PatientTrajectory
    from app.models.facility_metric import FacilityWeeklyMetric

    claim_id_str = str(claim.get("claim_id", ""))
    t0 = time.time()

    if claim_id_str in existing_ids:
        logger.warning(f"[persistence] SKIP duplicate claim_id={claim_id_str}")
        return {"claim_id": claim_id_str, "status": "skipped", "reason": "duplicate"}

    try:
        # ── Claim ──────────────────────────────────────────────────────────
        adm = claim.get("admission_date")
        dis = claim.get("discharge_date")
        if isinstance(adm, str):
            adm = datetime.strptime(adm[:10], "%Y-%m-%d").date()
        if isinstance(dis, str):
            dis = datetime.strptime(dis[:10], "%Y-%m-%d").date()

        db_claim = Claim(
            claim_id=claim_id_str,
            job_id=job_id,
            user_id=user_id,
            patient_id=str(claim.get("patient_id", "")),
            facility_id=str(claim.get("facility_id", "")),
            admission_date=adm,
            discharge_date=dis,
            claimed_diagnosis=str(
                claim.get("diagnosis") or claim.get("claimed_diagnosis") or ""
            ),
        )
        session.add(db_claim)
        await session.flush()  # get db_claim.id

        # ── CBCData ────────────────────────────────────────────────────────
        mv = claim.get("_model_validation", {}).get("models", {})
        # Only write CBCData if at least model1 or model3 was eligible (they need CBC labs)
        if mv.get("model1", {}).get("eligible") or mv.get("model3", {}).get("eligible"):
            session.add(CBCData(
                claim_id=db_claim.id,
                age=float(claim.get("age") or 0),
                sex_encoded=int(claim.get("sex_encoded") or 0),
                HGB=float(claim.get("HGB") or 0),
                HCT=float(claim.get("HCT") or 0),
                MCV=float(claim.get("MCV") or 0),
                MCHC=float(claim.get("MCHC") or 0),
                NEU=float(claim.get("NEU") or 0),
                LYM=float(claim.get("LYM") or 0),
                EOS=float(claim.get("EOS") or 0),
                BAS=float(claim.get("BAS") or 0),
                MON=float(claim.get("MON") or 0),
                PLT=float(claim.get("PLT") or 0),
                length_of_stay=float(claim.get("length_of_stay") or 1),
            ))

        # ── FraudFlags ─────────────────────────────────────────────────────
        for model_key, result in model_results.items():
            if not result or result.get("error"):
                continue
            model_id = result.get("model_id")
            if model_id is None:
                continue

            flag = FraudFlag(
                claim_id=db_claim.id,
                model_id=model_id,
                anomaly_score=float(result.get("anomaly_score", 0)),
                is_anomaly=bool(result.get("is_anomaly", False)),
                severity=result.get("severity", "low"),
                flag_reason=result.get("flag_reason", ""),
                insufficient_history=result.get("insufficient_history", False),
            )
            # Model 2 extras
            if model_id == 2:
                flag.predicted_category = result.get("predicted_category")
                flag.predicted_diagnosis = result.get("predicted_diagnosis")
                flag.category_confidence = result.get("category_confidence")
                flag.diagnosis_confidence = result.get("diagnosis_confidence")
            session.add(flag)

        # ── PatientTrajectory (Model 3) ────────────────────────────────────
        m3 = model_results.get("model3")
        if m3 and not m3.get("error"):
            patient_id = str(claim.get("patient_id", ""))
            result_pt = await session.execute(
                select(PatientTrajectory).where(PatientTrajectory.patient_id == patient_id)
            )
            traj = result_pt.scalar_one_or_none()
            if traj is None:
                traj = PatientTrajectory(patient_id=patient_id)
                session.add(traj)

            visits = json.loads(traj.visit_sequence or "[]")
            from app.agent.tools.cbc_tools import MODEL3_FEATURES
            visits.append({f: float(claim.get(f, 0.0)) for f in MODEL3_FEATURES})
            if len(visits) > 5:
                visits = visits[-5:]

            traj.visit_sequence = json.dumps(visits)
            traj.trajectory_anomaly_score = m3.get("trajectory_anomaly_score", 0.0)
            traj.is_trajectory_anomaly = m3.get("is_trajectory_anomaly", False)
            traj.per_visit_errors = json.dumps(m3.get("per_visit_errors", []))
            traj.most_anomalous_visit_index = m3.get("most_anomalous_visit_index", 0)
            traj.last_updated = datetime.utcnow()

        # ── FacilityWeeklyMetric ───────────────────────────────────────────
        if adm:
            week_start = _week_start(adm)
            facility_id = str(claim.get("facility_id", ""))

            all_scores = [
                r.get("anomaly_score", 0.0)
                for r in model_results.values()
                if r and not r.get("error")
            ]
            max_score = max(all_scores) if all_scores else 0.0
            any_anomaly = any(
                r.get("is_anomaly", False)
                for r in model_results.values()
                if r and not r.get("error")
            )
            worst_sev = "high" if max_score > 0.8 else ("medium" if max_score > 0.5 else "low")

            result_fm = await session.execute(
                select(FacilityWeeklyMetric).where(
                    FacilityWeeklyMetric.facility_id == facility_id,
                    FacilityWeeklyMetric.week_start_date == week_start,
                )
            )
            metric = result_fm.scalar_one_or_none()
            if metric is None:
                metric = FacilityWeeklyMetric(
                    facility_id=facility_id, week_start_date=week_start
                )
                session.add(metric)

            metric.claim_volume += 1
            prev_total = metric.avg_anomaly_score * (metric.claim_volume - 1)
            metric.avg_anomaly_score = (prev_total + max_score) / metric.claim_volume
            if any_anomaly:
                metric.flagged_claims += 1
                if worst_sev == "high":
                    metric.high_severity_count += 1
                elif worst_sev == "medium":
                    metric.medium_severity_count += 1
                else:
                    metric.low_severity_count += 1

        existing_ids.add(claim_id_str)
        logger.info(
            f"[persistence] DONE claim_id={claim_id_str} "
            f"models_run={list(model_results.keys())} elapsed={time.time()-t0:.3f}s"
        )
        return {"claim_id": claim_id_str, "status": "persisted"}

    except Exception as e:
        logger.error(f"[persistence] FAILED claim_id={claim_id_str}: {e}", exc_info=True)
        return {"claim_id": claim_id_str, "status": "failed", "reason": str(e)}
