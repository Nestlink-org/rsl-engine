"""Job worker — routes uploaded files to the appropriate processing pipeline.

Routing logic:
  - .csv / .xlsx / .xls  → CSV pipeline (pandas → validate → infer)
  - .pdf                  → OCR pipeline (PaddleOCR → LLM → validate → infer)
  - .jpg / .jpeg / .png / .bmp / .tiff → OCR pipeline (same as PDF, single-page)

Model registry is loaded in a thread executor to avoid blocking the event loop
during the heavy TensorFlow/Keras model loading.
"""

import asyncio
import logging
import time

from app.db.sessions import async_session
from app.services.csv_pipeline import run_csv_pipeline
from app.services.job_service import update_job_status
from app.services.model_registry import get_model_registry

logger = logging.getLogger(__name__)

# File types that go through the CSV pipeline
CSV_FILE_TYPES = {"csv", "xlsx", "xls"}

# File types that go through the OCR pipeline (PDF + all image formats)
OCR_FILE_TYPES = {"pdf", "jpg", "jpeg", "png", "bmp", "tiff"}


async def process_job(
    job_id: str,
    file_path: str,
    file_type: str,
    user_id: str,
) -> None:
    """
    Background task: set status → load models → run pipeline → update final status → publish events.

    Args:
        job_id: UUID string of the job
        file_path: path to the saved upload file
        file_type: file extension without dot (e.g. "csv", "pdf", "jpeg")
        user_id: user who submitted the job
    """
    async with async_session() as session:
        await update_job_status(session, job_id, "processing")

        try:
            t0 = time.time()
            logger.info(f"[job_worker] START job={job_id} file_type={file_type} user={user_id}")

            # Load models in a thread executor — TF/Keras loading is CPU-bound and slow
            loop = asyncio.get_event_loop()
            registry = await loop.run_in_executor(None, get_model_registry)
            logger.info(f"[job_worker] models loaded elapsed={time.time()-t0:.3f}s")

            if file_type in CSV_FILE_TYPES:
                # CSV/Excel now goes through the full agent pipeline
                # (structuring → validation → orchestrator → DB)
                from app.agent.graph import run_pipeline
                result = await run_pipeline(
                    file_path=file_path,
                    file_type=file_type,
                    job_id=job_id,
                    user_id=user_id,
                    session=session,
                    registry=registry,
                )
                processed = len(result["all"])
                failed = len(result["failed"])

            elif file_type in OCR_FILE_TYPES:
                from app.agent.graph import run_pipeline
                result = await run_pipeline(
                    file_path=file_path,
                    file_type=file_type,
                    job_id=job_id,
                    user_id=user_id,
                    session=session,
                    registry=registry,
                )
                processed = len(result["all"])
                failed = len(result["failed"])

            else:
                raise ValueError(
                    f"Unsupported file type: '{file_type}'. "
                    f"Expected one of: {sorted(CSV_FILE_TYPES | OCR_FILE_TYPES)}"
                )

            total = processed + failed
            if failed == 0:
                final_status = "completed"
            elif processed == 0:
                final_status = "failed"
            else:
                final_status = "partial"

            await update_job_status(
                session, job_id, final_status,
                total=total, processed=processed, failed=failed,
            )

            logger.info(
                f"[job_worker] DONE job={job_id} status={final_status} "
                f"processed={processed} failed={failed} elapsed={time.time()-t0:.3f}s"
            )

            # Publish results and audit events to Kafka
            try:
                from app.services.kafka_service import get_kafka_producer
                producer = get_kafka_producer()
                await producer.publish("rsl.fraud.results", {
                    "job_id": job_id,
                    "user_id": user_id,
                    "status": final_status,
                    "processed_claims": processed,
                    "failed_claims": failed,
                    "total_claims": total,
                })
                await producer.publish("rsl.audit.events", {
                    "event": "job_completed",
                    "job_id": job_id,
                    "user_id": user_id,
                    "status": final_status,
                    "total": total,
                    "processed": processed,
                    "failed": failed,
                })
            except Exception as ke:
                logger.warning(f"Kafka publish failed for job {job_id}: {ke}")

        except Exception as e:
            logger.error(f"[job_worker] FAILED job={job_id}: {e}", exc_info=True)
            await update_job_status(session, job_id, "failed", error_detail=str(e))
