"""Job service — create, fetch, and update Job records with Redis mirroring."""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.models.job import Job

logger = logging.getLogger(__name__)

VALID_STATUSES = {"pending", "processing", "completed", "failed", "partial", "skipped"}

_redis_client: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


async def create_job(
    session: AsyncSession,
    filename: str,
    file_type: str,
    user_id: str,
) -> Job:
    job = Job(
        job_id=str(uuid.uuid4()),
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        status="pending",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Mirror to Redis
    try:
        r = get_redis()
        await r.set(
            f"job:{job.job_id}",
            json.dumps({
                "job_id": job.job_id,
                "status": job.status,
                "total_claims": job.total_claims,
                "processed_claims": job.processed_claims,
                "failed_claims": job.failed_claims,
            }),
            ex=86400,  # 24h TTL
        )
    except Exception as e:
        logger.warning(f"Redis mirror failed for job {job.job_id}: {e}")

    return job


async def get_job(
    session: AsyncSession,
    job_id: str,
    user_id: str,
) -> Optional[Job]:
    result = await session.execute(
        select(Job).where(Job.job_id == job_id, Job.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def update_job_status(
    session: AsyncSession,
    job_id: str,
    status: str,
    total: Optional[int] = None,
    processed: Optional[int] = None,
    failed: Optional[int] = None,
    error_detail: Optional[str] = None,
) -> None:
    assert status in VALID_STATUSES, f"Invalid status: {status}"

    result = await session.execute(select(Job).where(Job.job_id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        return

    job.status = status
    job.updated_at = datetime.utcnow()
    if total is not None:
        job.total_claims = total
    if processed is not None:
        job.processed_claims = processed
    if failed is not None:
        job.failed_claims = failed
    if error_detail is not None:
        job.error_detail = error_detail

    await session.commit()

    # Mirror to Redis
    try:
        r = get_redis()
        await r.set(
            f"job:{job_id}",
            json.dumps({
                "job_id": job_id,
                "status": status,
                "total_claims": job.total_claims,
                "processed_claims": job.processed_claims,
                "failed_claims": job.failed_claims,
                "error_detail": job.error_detail,
            }),
            ex=86400,
        )
    except Exception as e:
        logger.warning(f"Redis mirror update failed for job {job_id}: {e}")
