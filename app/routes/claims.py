"""GET /api/v1/claims/{claim_id}"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.sessions import get_async_db
from app.models.claim import CBCData, Claim
from app.models.fraud_flag import FraudFlag
from app.schemas.claim import CBCDataOut, ClaimDetailResponse
from app.schemas.job import FraudFlagOut
from app.services.audit_service import log_audit_event

router = APIRouter()


@router.get("/claims/{claim_id}", response_model=ClaimDetailResponse)
async def get_claim(claim_id: str, request: Request, session: AsyncSession = Depends(get_async_db)):
    user_id = getattr(request.state, "user_id", None) or "anonymous"

    result = await session.execute(
        select(Claim).where(Claim.claim_id == claim_id, Claim.user_id == user_id)
    )
    claim = result.scalar_one_or_none()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    cbc_result = await session.execute(select(CBCData).where(CBCData.claim_id == claim.id))
    cbc = cbc_result.scalar_one_or_none()

    flags_result = await session.execute(select(FraudFlag).where(FraudFlag.claim_id == claim.id))
    flags = [FraudFlagOut(**f.__dict__) for f in flags_result.scalars().all()]

    await log_audit_event(session, user_id, "view_claim", "claim", claim_id)

    return ClaimDetailResponse(
        claim_id=claim.claim_id, job_id=claim.job_id,
        patient_id=claim.patient_id, facility_id=claim.facility_id,
        admission_date=claim.admission_date, discharge_date=claim.discharge_date,
        claimed_diagnosis=claim.claimed_diagnosis, created_at=claim.created_at,
        cbc_data=CBCDataOut(**cbc.__dict__) if cbc else None,
        fraud_flags=flags,
    )
