"""Orchestrator node — Phase 2 of the RSL multi-agent pipeline.

Responsibilities:
  1. Iterate over validated_claims from Phase 1
  2. For each claim, check _model_validation to decide which CBC tools to run
  3. Run eligible tools (model1, model2, model3; model4 is facility-level)
  4. Build facility weekly sequences from the current batch for model4
  5. Persist all results to DB via persistence.py
  6. Return inference_results summary

Model 4 strategy:
  - Collect all claims per facility in the current batch
  - Build weekly aggregate from the batch (best effort — may be < 8 weeks)
  - Run model4 once per facility, store result against each claim from that facility

Redis usage:
  - Patient history fetched from DB (PatientTrajectory table)
  - Facility weekly history fetched from DB (FacilityWeeklyMetric table)
  - Overflow claims already queued in Redis by structuring_agent (Phase 1)
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


def _week_start(d) -> date:
    if isinstance(d, str):
        d = datetime.strptime(d[:10], "%Y-%m-%d").date()
    return d - timedelta(days=d.weekday())


async def _get_patient_history(session: AsyncSession, patient_id: str) -> List[Dict]:
    """Fetch patient visit history from PatientTrajectory table."""
    from sqlalchemy import select
    from app.models.patient_trajectory import PatientTrajectory
    from app.agent.tools.cbc_tools import MODEL3_FEATURES
    try:
        result = await session.execute(
            select(PatientTrajectory).where(PatientTrajectory.patient_id == patient_id)
        )
        traj = result.scalar_one_or_none()
        if traj and traj.visit_sequence:
            return json.loads(traj.visit_sequence)
    except Exception as e:
        logger.warning(f"[orchestrator] patient history fetch failed pid={patient_id}: {e}")
    return []


async def _get_facility_weekly_sequence(
    session: AsyncSession, facility_id: str, seq_len: int = 8
) -> List[Dict]:
    """Fetch last seq_len weeks of facility metrics from DB."""
    from sqlalchemy import select
    from app.models.facility_metric import FacilityWeeklyMetric
    from app.agent.tools.cbc_tools import MODEL4_FEATURES
    try:
        cutoff = date.today() - timedelta(weeks=seq_len)
        result = await session.execute(
            select(FacilityWeeklyMetric)
            .where(
                FacilityWeeklyMetric.facility_id == facility_id,
                FacilityWeeklyMetric.week_start_date >= cutoff,
            )
            .order_by(FacilityWeeklyMetric.week_start_date)
        )
        metrics = result.scalars().all()
        if not metrics:
            return []
        # Convert to feature dicts
        sequence = []
        for m in metrics:
            sequence.append({
                "claim_volume": float(m.claim_volume),
                "avg_age": 0.0,   # not stored per-week in current schema — use 0
                "age_std": 0.0,
                "pct_male": 0.0,
                "HGB_mean": 0.0, "HGB_std": 0.0,
                "HCT_mean": 0.0, "HCT_std": 0.0,
                "MCV_mean": 0.0, "MCV_std": 0.0,
                "MCHC_mean": 0.0, "MCHC_std": 0.0,
                "NEU_mean": 0.0, "LYM_mean": 0.0,
                "EOS_mean": 0.0, "BAS_mean": 0.0,
                "MON_mean": 0.0, "PLT_mean": 0.0,
                "avg_los": 0.0,
            })
        return sequence
    except Exception as e:
        logger.warning(f"[orchestrator] facility sequence fetch failed fid={facility_id}: {e}")
    return []


def _build_facility_weekly_from_batch(
    claims: List[Dict[str, Any]], facility_id: str
) -> List[Dict]:
    """
    Build a weekly aggregate sequence from the current batch claims for a facility.
    Groups claims by week, computes mean/std of CBC labs per week.
    """
    import numpy as np
    from app.agent.tools.cbc_tools import MODEL4_FEATURES

    facility_claims = [c for c in claims if str(c.get("facility_id", "")) == facility_id]
    if not facility_claims:
        return []

    # Group by week
    weeks: Dict[date, List[Dict]] = defaultdict(list)
    for c in facility_claims:
        adm = c.get("admission_date")
        if adm:
            ws = _week_start(adm)
            weeks[ws].append(c)

    sequence = []
    for ws in sorted(weeks.keys()):
        wc = weeks[ws]
        lab_fields = ["HGB", "HCT", "MCV", "MCHC", "NEU", "LYM", "EOS", "BAS", "MON", "PLT"]
        ages = [float(c.get("age") or 0) for c in wc]
        sexes = [float(c.get("sex_encoded") or 0) for c in wc]
        los_vals = [float(c.get("length_of_stay") or 1) for c in wc]

        week_dict: Dict[str, float] = {
            "claim_volume": float(len(wc)),
            "avg_age": float(np.mean(ages)) if ages else 0.0,
            "age_std": float(np.std(ages)) if len(ages) > 1 else 0.0,
            "pct_male": float(np.mean(sexes)) if sexes else 0.0,
            "avg_los": float(np.mean(los_vals)) if los_vals else 0.0,
        }
        for lab in lab_fields:
            vals = [float(c.get(lab) or 0) for c in wc if c.get(lab) is not None]
            week_dict[f"{lab}_mean"] = float(np.mean(vals)) if vals else 0.0
            week_dict[f"{lab}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

        sequence.append(week_dict)

    return sequence


async def run_orchestrator(
    validated_claims: List[Dict[str, Any]],
    job_id: str,
    user_id: str,
    session: AsyncSession,
    registry,
) -> Dict[str, Any]:
    """
    Main orchestrator: run eligible CBC models per claim, persist results.

    Returns summary dict with per-claim results and aggregate stats.
    """
    from sqlalchemy import select
    from app.models.claim import Claim
    from app.agent.tools.cbc_tools import (
        run_cbc_model1, run_cbc_model2, run_cbc_model3, run_cbc_model4
    )
    from app.agent.tools.persistence import persist_claim_results

    t0 = time.time()
    logger.info(
        f"[orchestrator] START job={job_id} claims={len(validated_claims)}"
    )

    # Fetch existing claim_ids to prevent duplicates
    existing_result = await session.execute(select(Claim.claim_id))
    existing_ids = set(existing_result.scalars().all())

    # Pre-build facility weekly sequences from this batch (for model4)
    facility_ids = list({str(c.get("facility_id", "")) for c in validated_claims})
    facility_sequences: Dict[str, List[Dict]] = {}
    for fid in facility_ids:
        # Combine DB history + current batch
        db_seq = await _get_facility_weekly_sequence(session, fid)
        batch_seq = _build_facility_weekly_from_batch(validated_claims, fid)
        combined = (db_seq + batch_seq)[-8:]  # keep last 8 weeks
        facility_sequences[fid] = combined

    all_results = []
    failed_results = []
    models_triggered: Dict[str, int] = defaultdict(int)

    loop = asyncio.get_event_loop()

    for claim in validated_claims:
        claim_id = str(claim.get("claim_id", "unknown"))
        mv = claim.get("_model_validation", {})
        models_status = mv.get("models", {})
        claim_type = mv.get("claim_type", "unknown")

        logger.info(
            f"[orchestrator] processing claim_id={claim_id} type={claim_type} "
            f"eligible={[k for k,v in models_status.items() if v.get('eligible')]}"
        )

        # Skip non-CBC claims in this phase
        if claim_type != "cbc":
            logger.info(f"[orchestrator] SKIP claim_id={claim_id} type={claim_type} (not CBC)")
            failed_results.append({
                "claim_id": claim_id,
                "reason": f"claim_type={claim_type} — only CBC supported in this phase",
            })
            continue

        model_results: Dict[str, Any] = {}
        patient_id = str(claim.get("patient_id", ""))

        # ── Model 1 ──────────────────────────────────────────────────────
        if models_status.get("model1", {}).get("eligible"):
            m1 = await loop.run_in_executor(None, run_cbc_model1, claim, registry)
            model_results["model1"] = m1
            models_triggered["model1"] += 1

        # ── Model 2 ──────────────────────────────────────────────────────
        if models_status.get("model2", {}).get("eligible"):
            m2 = await loop.run_in_executor(None, run_cbc_model2, claim, registry)
            model_results["model2"] = m2
            models_triggered["model2"] += 1

        # ── Model 3 ──────────────────────────────────────────────────────
        if models_status.get("model3", {}).get("eligible"):
            patient_history = await _get_patient_history(session, patient_id)
            m3 = await loop.run_in_executor(
                None, run_cbc_model3, claim, patient_history, registry
            )
            model_results["model3"] = m3
            models_triggered["model3"] += 1

        # ── Model 4 (facility-level) ──────────────────────────────────────
        if registry.model4_available:
            fid = str(claim.get("facility_id", ""))
            fseq = facility_sequences.get(fid, [])
            if fseq:
                m4 = await loop.run_in_executor(
                    None, run_cbc_model4, fseq, fid, registry
                )
                model_results["model4"] = m4
                models_triggered["model4"] += 1

        if not model_results:
            logger.warning(f"[orchestrator] no models ran for claim_id={claim_id}")
            failed_results.append({
                "claim_id": claim_id,
                "reason": "no eligible models produced results",
            })
            continue

        # ── Persist ───────────────────────────────────────────────────────
        persist_result = await persist_claim_results(
            session=session,
            claim=claim,
            model_results=model_results,
            job_id=job_id,
            user_id=user_id,
            existing_ids=existing_ids,
        )

        if persist_result["status"] == "persisted":
            # Build result summary for response
            any_anomaly = any(
                r.get("is_anomaly", False)
                for r in model_results.values() if r and not r.get("error")
            )
            max_score = max(
                (r.get("anomaly_score", 0.0) for r in model_results.values() if r and not r.get("error")),
                default=0.0,
            )
            all_results.append({
                "claim_id": claim_id,
                "status": "processed",
                "any_anomaly": any_anomaly,
                "max_anomaly_score": round(max_score, 4),
                "models_run": list(model_results.keys()),
                "model_results": {
                    k: {
                        "anomaly_score": v.get("anomaly_score"),
                        "is_anomaly": v.get("is_anomaly"),
                        "severity": v.get("severity"),
                        "flag_reason": v.get("flag_reason"),
                    }
                    for k, v in model_results.items() if v and not v.get("error")
                },
            })
        elif persist_result["status"] == "skipped":
            # Duplicate — count as processed (not failed), include in results
            all_results.append({
                "claim_id": claim_id,
                "status": "skipped",
                "reason": persist_result.get("reason", "duplicate"),
                "any_anomaly": None,
                "max_anomaly_score": None,
                "models_run": list(model_results.keys()),
                "model_results": {},
            })
        else:
            failed_results.append(persist_result)

    await session.commit()

    elapsed = time.time() - t0
    summary = {
        "job_id": job_id,
        "total_claims": len(validated_claims),
        "processed": len(all_results),
        "failed": len(failed_results),
        "models_triggered": dict(models_triggered),
        "elapsed": round(elapsed, 3),
        "results": all_results,
        "failed_details": failed_results,
    }
    logger.info(
        f"[orchestrator] DONE job={job_id} processed={len(all_results)} "
        f"failed={len(failed_results)} models={dict(models_triggered)} elapsed={elapsed:.3f}s"
    )
    return summary
