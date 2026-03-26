"""POST /api/v1/upload — unified file upload and fraud detection processing.

Single endpoint for all file types:
  - CSV / Excel (.csv, .xlsx, .xls)  → agent pipeline (structuring → validate → infer → DB)
  - PDF (.pdf)                        → OCR → LLM structuring → validate → infer → DB
  - Images (.jpg, .jpeg, .png, .bmp, .tiff) → OCR → LLM structuring → validate → infer → DB

Query params:
  ?sync=false (default) — returns HTTP 202 immediately, processing in background
  ?sync=true            — waits for completion, returns full results inline (≤5 claims)
"""

import asyncio
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.db.sessions import get_async_db
from app.schemas.inference import ClaimInferenceResult, PipelineResponse
from app.schemas.upload import UploadResponse
from app.services.audit_service import log_audit_event
from app.services.job_service import create_job, update_job_status

logger = logging.getLogger(__name__)
router = APIRouter()

CSV_EXTENSIONS = {".csv", ".xlsx", ".xls"}
OCR_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
ALLOWED_EXTENSIONS = CSV_EXTENSIONS | OCR_EXTENSIONS


@router.post(
    "/upload",
    summary="Upload and process a claim file",
    description="""
**Unified upload endpoint** — accepts all supported file formats.

Supported formats:
- **Structured**: `.csv`, `.xlsx`, `.xls` — direct column mapping, no OCR needed
- **Scanned documents**: `.pdf` — PaddleOCR text extraction → gpt-5.4-nano structuring
- **Scanned images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` — same as PDF

Processing pipeline for every file:
1. Structure claim fields (pandas for CSV/Excel, OCR+LLM for PDF/image)
2. Validate fields, ranges, dates
3. Run eligible CBC fraud detection models (1–4)
4. Persist results to database

**Async mode** (`sync=false`, default):
- Returns HTTP 202 immediately with `job_id`
- Poll `GET /api/v1/jobs/{job_id}/status` for progress
- Retrieve results at `GET /api/v1/jobs/{job_id}/results`

**Sync mode** (`sync=true`):
- Waits for completion, returns full results inline
- Recommended only for small files (≤5 claims)
- Larger files: first batch processed, overflow queued in Redis
    """,
    tags=["Upload"],
)
async def upload_file(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Claim file: CSV/Excel (structured) or PDF/image (OCR pipeline)",
    ),
    sync: bool = Query(
        False,
        description="If true, wait for processing and return results inline. Use for small files only.",
    ),
    session: AsyncSession = Depends(get_async_db),
):
    user_id = getattr(request.state, "user_id", None) or "anonymous"
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Structured: {sorted(CSV_EXTENSIONS)} | "
                f"Scanned (OCR): {sorted(OCR_EXTENSIONS)}"
            ),
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit")

    file_type = ext.lstrip(".")
    job = await create_job(session, filename, file_type, user_id)

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(settings.UPLOAD_DIR, f"{job.job_id}{ext}")
    with open(save_path, "wb") as f:
        f.write(content)

    await log_audit_event(
        session, user_id, "upload_file", "job", job.job_id,
        {"filename": filename, "file_type": file_type, "size_bytes": len(content), "sync": sync},
    )

    # ── Async mode (default) ──────────────────────────────────────────────────
    if not sync:
        from app.workers.job_worker import process_job
        asyncio.create_task(process_job(job.job_id, save_path, file_type, user_id))
        return UploadResponse(
            job_id=job.job_id,
            status=job.status,
            filename=filename,
            message="Processing started in background.",
            status_url=f"/api/v1/jobs/{job.job_id}/status",
            results_url=f"/api/v1/jobs/{job.job_id}/results",
        )

    # ── Sync mode ─────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        await update_job_status(session, job.job_id, "processing")

        from app.agent.graph import run_pipeline
        from app.services.model_registry import get_model_registry

        loop = asyncio.get_event_loop()
        registry = await loop.run_in_executor(None, get_model_registry)

        result = await run_pipeline(
            file_path=save_path,
            file_type=file_type,
            job_id=job.job_id,
            user_id=user_id,
            session=session,
            registry=registry,
        )

        processed = len(result["all"])
        failed = len(result["failed"])
        queued_count = result.get("queued_count", 0)
        orch = result.get("orchestrator_summary", {})

        # Determine status: skipped = all duplicates, partial = some processed
        skipped = sum(1 for r in result["all"] if r.get("status") == "skipped")
        truly_processed = processed - skipped
        if failed == 0 and truly_processed == 0 and skipped > 0:
            status = "skipped"
        elif failed == 0:
            status = "completed" if queued_count == 0 else "partial"
        elif truly_processed == 0 and skipped == 0:
            status = "failed"
        else:
            status = "partial"

        # total includes queued claims so job record reflects full file size
        await update_job_status(
            session, job.job_id, status,
            total=processed + failed + queued_count,
            processed=truly_processed,
            failed=failed,
        )

        return PipelineResponse(
            job_id=job.job_id,
            status=status,
            filename=filename,
            file_type=file_type,
            total_processed=processed,
            total_failed=failed,
            queued_count=queued_count,
            models_triggered=orch.get("models_triggered", {}),
            results=[ClaimInferenceResult(**r) for r in result["all"]],
            failed_details=result["failed"],
            processing_time=round(time.time() - t0, 3),
            poll_url=f"/api/v1/jobs/{job.job_id}/status",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[upload/sync] FAILED job={job.job_id}: {e}", exc_info=True)
        await update_job_status(session, job.job_id, "failed", error_detail=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass
