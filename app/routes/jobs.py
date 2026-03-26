"""GET /api/v1/jobs/{job_id}/status and /results"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.sessions import get_async_db
from app.models.claim import Claim
from app.models.fraud_flag import FraudFlag
from app.schemas.job import ClaimResultItem, FraudFlagOut, JobResultsResponse, JobStatusResponse
from app.services.job_service import get_job

router = APIRouter()


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def job_status(job_id: str, request: Request, session: AsyncSession = Depends(get_async_db)):
    user_id = getattr(request.state, "user_id", None) or "anonymous"
    job = await get_job(session, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id, status=job.status,
        created_at=job.created_at, updated_at=job.updated_at,
        total_claims=job.total_claims, processed_claims=job.processed_claims,
        failed_claims=job.failed_claims, error_detail=job.error_detail,
    )


@router.get("/jobs/{job_id}/results", response_model=JobResultsResponse)
async def job_results(
    job_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    session: AsyncSession = Depends(get_async_db),
):
    user_id = getattr(request.state, "user_id", None) or "anonymous"
    job = await get_job(session, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("pending", "processing"):
        raise HTTPException(status_code=202, detail=f"Job is {job.status}")

    offset = (page - 1) * page_size
    claims_result = await session.execute(
        select(Claim).where(Claim.job_id == job_id, Claim.user_id == user_id)
        .offset(offset).limit(page_size)
    )
    claims = claims_result.scalars().all()

    items: List[ClaimResultItem] = []
    for claim in claims:
        flags_result = await session.execute(
            select(FraudFlag).where(FraudFlag.claim_id == claim.id)
        )
        flags = [FraudFlagOut(**f.__dict__) for f in flags_result.scalars().all()]
        items.append(ClaimResultItem(
            claim_id=claim.claim_id, patient_id=claim.patient_id,
            facility_id=claim.facility_id,
            admission_date=claim.admission_date,
            discharge_date=claim.discharge_date,
            claimed_diagnosis=claim.claimed_diagnosis,
            fraud_flags=flags,
        ))

    return JobResultsResponse(
        job_id=job_id, status=job.status, page=page, page_size=page_size,
        total_claims=job.total_claims, claims=items,
    )
