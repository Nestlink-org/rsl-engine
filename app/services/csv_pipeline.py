"""CSV/XLSX pipeline — parse, validate, infer, and persist claims."""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.claim import CBCData, Claim
from app.models.facility_metric import FacilityWeeklyMetric
from app.models.fraud_flag import FraudFlag
from app.models.patient_trajectory import PatientTrajectory
from app.services.inference_service import MODEL3_FEATURES, run_inference
from app.services.model_registry import ModelRegistry
from app.services.validation_service import validate_claim

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "claim_id", "patient_id", "facility_id", "admission_date", "discharge_date",
    "claimed_diagnosis", "age", "sex", "HGB", "HCT", "MCV", "MCHC",
    "NEU", "LYM", "EOS", "BAS", "MON", "PLT", "length_of_stay",
]


def parse_csv_file(file_path: str) -> pd.DataFrame:
    """Parse CSV/XLSX file and validate required columns."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def _week_start(d: date) -> date:
    """Return the Monday of the week containing d."""
    return d - timedelta(days=d.weekday())


async def _upsert_patient_trajectory(
    session: AsyncSession,
    patient_id: str,
    visit_features: Dict[str, Any],
    model3_result,
) -> None:
    from sqlalchemy import select
    result = await session.execute(select(PatientTrajectory).where(PatientTrajectory.patient_id == patient_id))
    traj = result.scalar_one_or_none()

    if traj is None:
        traj = PatientTrajectory(patient_id=patient_id)
        session.add(traj)

    visits = json.loads(traj.visit_sequence or "[]")
    visits.append({f: visit_features.get(f, 0.0) for f in MODEL3_FEATURES})
    if len(visits) > 5:
        visits = visits[-5:]  # keep most recent 5

    traj.visit_sequence = json.dumps(visits)
    traj.trajectory_anomaly_score = model3_result.trajectory_anomaly_score
    traj.is_trajectory_anomaly = model3_result.is_trajectory_anomaly
    traj.per_visit_errors = json.dumps(model3_result.per_visit_errors)
    traj.most_anomalous_visit_index = model3_result.most_anomalous_visit_index
    traj.last_updated = datetime.utcnow()


async def _upsert_facility_metric(
    session: AsyncSession,
    facility_id: str,
    week_start: date,
    anomaly_score: float,
    is_anomaly: bool,
    severity: str,
) -> None:
    from sqlalchemy import select
    result = await session.execute(
        select(FacilityWeeklyMetric).where(
            FacilityWeeklyMetric.facility_id == facility_id,
            FacilityWeeklyMetric.week_start_date == week_start,
        )
    )
    metric = result.scalar_one_or_none()
    if metric is None:
        metric = FacilityWeeklyMetric(facility_id=facility_id, week_start_date=week_start)
        session.add(metric)

    metric.claim_volume += 1
    # Running average
    prev_total = metric.avg_anomaly_score * (metric.claim_volume - 1)
    metric.avg_anomaly_score = (prev_total + anomaly_score) / metric.claim_volume

    if is_anomaly:
        metric.flagged_claims += 1
        if severity == "high":
            metric.high_severity_count += 1
        elif severity == "medium":
            metric.medium_severity_count += 1
        else:
            metric.low_severity_count += 1


async def run_csv_pipeline(
    file_path: str,
    job_id: str,
    user_id: str,
    session: AsyncSession,
    registry: ModelRegistry,
) -> Tuple[int, int, List[Dict]]:
    """
    Parse → validate → infer → persist.
    Returns (processed_count, failed_count, failed_details).
    """
    from sqlalchemy import select

    t0 = time.time()
    logger.info(f"[csv_pipeline] START job={job_id} file={file_path}")

    df = parse_csv_file(file_path)
    logger.info(f"[csv_pipeline] parsed rows={len(df)} elapsed={time.time()-t0:.3f}s")

    # Fetch existing claim_ids to detect duplicates
    existing_result = await session.execute(select(Claim.claim_id))
    existing_ids = set(existing_result.scalars().all())

    processed, failed = 0, 0
    failed_details: List[Dict] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        claim_id_str = str(row_dict.get("claim_id", ""))

        is_valid, reason, enriched = validate_claim(row_dict, existing_ids)
        if not is_valid:
            failed += 1
            failed_details.append({"claim_id": claim_id_str, "reason": reason})
            continue

        try:
            t_row = time.time()
            # Fetch patient history for Model 3
            hist_result = await session.execute(
                select(PatientTrajectory).where(PatientTrajectory.patient_id == str(enriched["patient_id"]))
            )
            traj = hist_result.scalar_one_or_none()
            patient_history = json.loads(traj.visit_sequence) if traj else []

            results = await run_inference(enriched, patient_history, registry)
            m1, m2, m3 = results["model1"], results["model2"], results["model3"]

            # Persist Claim
            claim = Claim(
                claim_id=claim_id_str,
                job_id=job_id,
                user_id=user_id,
                patient_id=str(enriched["patient_id"]),
                facility_id=str(enriched["facility_id"]),
                admission_date=enriched["admission_date"],
                discharge_date=enriched["discharge_date"],
                claimed_diagnosis=str(enriched["claimed_diagnosis"]),
            )
            session.add(claim)
            await session.flush()  # get claim.id

            # Persist CBCData
            cbc = CBCData(
                claim_id=claim.id,
                age=float(enriched["age"]),
                sex_encoded=enriched["sex_encoded"],
                HGB=float(enriched["HGB"]),
                HCT=float(enriched["HCT"]),
                MCV=float(enriched["MCV"]),
                MCHC=float(enriched["MCHC"]),
                NEU=float(enriched["NEU"]),
                LYM=float(enriched["LYM"]),
                EOS=float(enriched["EOS"]),
                BAS=float(enriched["BAS"]),
                MON=float(enriched["MON"]),
                PLT=float(enriched["PLT"]),
                length_of_stay=float(enriched["length_of_stay"]),
            )
            session.add(cbc)

            # Persist FraudFlags
            session.add(FraudFlag(
                claim_id=claim.id, model_id=1,
                anomaly_score=m1.anomaly_score, is_anomaly=m1.is_anomaly,
                severity=m1.severity, flag_reason=m1.flag_reason,
            ))
            session.add(FraudFlag(
                claim_id=claim.id, model_id=2,
                anomaly_score=m2.anomaly_score, is_anomaly=m2.is_anomaly,
                severity=m2.severity, flag_reason=m2.flag_reason,
                predicted_category=m2.predicted_category,
                predicted_diagnosis=m2.predicted_diagnosis,
                category_confidence=m2.category_confidence,
                diagnosis_confidence=m2.diagnosis_confidence,
            ))
            session.add(FraudFlag(
                claim_id=claim.id, model_id=3,
                anomaly_score=m3.trajectory_anomaly_score,
                is_anomaly=m3.is_trajectory_anomaly,
                severity="low" if not m3.is_trajectory_anomaly else (
                    "high" if m3.trajectory_anomaly_score > 0.8 else "medium"
                ),
                flag_reason=(
                    f"Trajectory anomaly (score={m3.trajectory_anomaly_score:.3f})"
                    if m3.is_trajectory_anomaly else "Trajectory within normal range"
                ),
                insufficient_history=m3.insufficient_history,
            ))

            # Update PatientTrajectory
            await _upsert_patient_trajectory(session, str(enriched["patient_id"]), enriched, m3)

            # Update FacilityWeeklyMetric
            week_start = _week_start(enriched["admission_date"])
            max_score = max(m1.anomaly_score, m2.anomaly_score, m3.trajectory_anomaly_score)
            any_anomaly = m1.is_anomaly or m2.is_anomaly or m3.is_trajectory_anomaly
            worst_sev = "high" if max_score > 0.8 else ("medium" if max_score > 0.5 else "low")
            await _upsert_facility_metric(
                session, str(enriched["facility_id"]), week_start, max_score, any_anomaly, worst_sev
            )

            existing_ids.add(claim_id_str)
            processed += 1
            logger.debug(
                f"[csv_pipeline] claim={claim_id_str} processed elapsed={time.time()-t_row:.3f}s"
            )

        except Exception as e:
            logger.error(f"Row processing error for claim {claim_id_str}: {e}")
            failed += 1
            failed_details.append({"claim_id": claim_id_str, "reason": str(e)})

    await session.commit()
    elapsed = time.time() - t0
    logger.info(
        f"[csv_pipeline] DONE job={job_id} processed={processed} failed={failed} "
        f"total={processed+failed} elapsed={elapsed:.3f}s"
    )
    return processed, failed, failed_details
