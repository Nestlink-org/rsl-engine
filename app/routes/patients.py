"""GET /api/v1/patients/{patient_id}/trajectory"""

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.sessions import get_async_db
from app.models.patient_trajectory import PatientTrajectory
from app.schemas.patient import PatientTrajectoryResponse, VisitRecord

router = APIRouter()


@router.get("/patients/{patient_id}/trajectory", response_model=PatientTrajectoryResponse)
async def patient_trajectory(
    patient_id: str,
    request: Request,
    session: AsyncSession = Depends(get_async_db),
):
    user_id = getattr(request.state, "user_id", None) or "anonymous"

    result = await session.execute(
        select(PatientTrajectory).where(PatientTrajectory.patient_id == patient_id)
    )
    traj = result.scalar_one_or_none()
    if not traj:
        raise HTTPException(status_code=404, detail="Patient trajectory not found")

    visits_raw = json.loads(traj.visit_sequence or "[]")
    per_visit_errors = json.loads(traj.per_visit_errors or "[]")
    total_visits = len(visits_raw)

    if total_visits < 2:
        raise HTTPException(status_code=422, detail="Insufficient visit history (minimum 2 visits required)")

    visits: List[VisitRecord] = [
        VisitRecord(
            visit_index=i,
            features=v,
            reconstruction_error=per_visit_errors[i] if i < len(per_visit_errors) else None,
        )
        for i, v in enumerate(visits_raw[-5:])
    ]

    return PatientTrajectoryResponse(
        patient_id=patient_id,
        total_visits=total_visits,
        trajectory_anomaly_score=traj.trajectory_anomaly_score,
        is_trajectory_anomaly=traj.is_trajectory_anomaly,
        most_anomalous_visit_index=traj.most_anomalous_visit_index,
        insufficient_history=total_visits < 5,
        visits=visits,
        last_updated=traj.last_updated,
    )
