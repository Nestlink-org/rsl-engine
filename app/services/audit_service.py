"""Audit service — persists AuditLog records and publishes to Kafka."""

import json
import logging
from typing import Any, Dict, Optional

from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


async def log_audit_event(
    session: AsyncSession,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist an AuditLog record using the provided session (no commit — caller commits).
    Kafka publish is fire-and-forget via asyncio.create_task so it never blocks the caller.
    """
    import asyncio

    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        extra_data=json.dumps(metadata) if metadata else None,
    )
    session.add(log)
    # Flush to DB within the caller's transaction — do NOT commit here.
    # The caller (route handler) owns the transaction lifecycle.
    await session.flush()

    # Kafka publish is non-blocking — failures are logged, never raised
    async def _publish():
        try:
            from app.services.kafka_service import get_kafka_producer
            producer = get_kafka_producer()
            await producer.publish("rsl.audit.events", {
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "metadata": metadata or {},
            })
        except Exception as e:
            logger.warning(f"Audit Kafka publish failed: {e}")

    asyncio.create_task(_publish())
