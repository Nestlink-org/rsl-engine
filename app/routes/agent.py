"""Agent utility endpoints — structuring preview and batch queue management.

These complement the main /api/v1/upload endpoint:
  POST /agent/structure/preview   — dry-run: extract + validate claims, NO inference, NO DB write
  GET  /agent/queue/{job_id}      — check how many claims are still queued in Redis
  POST /agent/queue/{job_id}/next — process next batch from Redis queue
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.db.sessions import get_async_db
from app.schemas.inference import (
    BatchQueueStatus,
    ClaimInferenceResult,
    PipelineResponse,
    StructurePreviewResponse,
    StructuredClaimPreview,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent — Utilities"])

STRUCTURED_EXTS = {".csv", ".xlsx", ".xls"}
UNSTRUCTURED_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
ALL_EXTS = STRUCTURED_EXTS | UNSTRUCTURED_EXTS


def _save_upload(content: bytes, ext: str) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, f"preview_{uuid.uuid4()}{ext}")
    with open(path, "wb") as f:
        f.write(content)
    return path


# ── POST /agent/structure/preview ─────────────────────────────────────────────

@router.post(
    "/structure/preview",
    response_model=StructurePreviewResponse,
    summary="Dry-run: preview claim extraction without inference or DB write",
    description="""
Upload any supported file and preview what the agent extracts — **no models run, nothing saved to DB**.

Returns ALL claims found in the file (no batch limit) with:
- Extracted canonical claim fields
- Which CBC models are eligible per claim
- Validation summary per claim

**This is a dry-run only.** The `preview_id` in the response is NOT a job ID and cannot be
used with `/jobs/` or `/agent/queue/` endpoints.

To actually process claims (run models + save to DB), use `POST /api/v1/upload`.
    """,
)
async def structure_preview(
    request: Request,
    file: UploadFile = File(..., description="CSV/Excel or PDF/image claim file"),
):
    t0 = time.time()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALL_EXTS:
        raise HTTPException(status_code=422, detail=f"Unsupported file type '{ext}'")

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit")

    file_path = _save_upload(content, ext)
    file_type = ext.lstrip(".")
    preview_id = str(uuid.uuid4())

    try:
        from app.agent.structuring_agent import (
            get_structuring_summary,
            structure_from_file,
            structure_from_ocr,
        )

        if file_type in {"csv", "xlsx", "xls"}:
            # Preview shows ALL claims — pass a very large batch_size so nothing is queued
            claims, _ = structure_from_file(file_path, job_id=preview_id, batch_size=10000)
            source = "csv" if file_type == "csv" else "excel"
        else:
            from app.services.ocr_service import get_ocr_service
            ocr = get_ocr_service()
            blocks = ocr.process_pdf(file_path) if file_type == "pdf" else ocr.process_image(file_path, page_number=1)
            full_text = "\n".join(b.text for b in blocks)
            claims = structure_from_ocr(full_text=full_text, job_id=preview_id)
            source = "ocr_text"

        summary = get_structuring_summary(claims)

        preview_claims = []
        for c in claims:
            flat = {(k[1:] if k.startswith("_") else k): v for k, v in c.items()}
            try:
                preview_claims.append(StructuredClaimPreview(**flat))
            except Exception:
                preview_claims.append(StructuredClaimPreview.model_construct(**flat))

        return StructurePreviewResponse(
            preview_id=preview_id,
            source=source,
            total_claims=summary["total_claims"],
            claims_shown=len(preview_claims),
            cbc_claims=summary["cbc_claims"],
            hba1c_claims=summary["hba1c_claims"],
            unknown_claims=summary["unknown_claims"],
            model_eligible_counts=summary["model_eligible_counts"],
            claims=preview_claims,
            processing_time=round(time.time() - t0, 3),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[agent/preview] FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass


# ── GET /agent/queue/{job_id} ─────────────────────────────────────────────────

@router.get(
    "/queue/{job_id}",
    response_model=BatchQueueStatus,
    summary="Check Redis batch queue depth for a job",
    description="""
After uploading a large file (>5 claims) via `POST /api/v1/upload`, overflow claims
are queued in Redis under the returned `job_id`.

This endpoint shows how many claims are still waiting to be processed.

**Correct flow for large files:**
1. `POST /api/v1/upload` → get `job_id`
2. `GET /agent/queue/{job_id}` → check how many claims are queued
3. `POST /agent/queue/{job_id}/next` → process next batch (repeat until queue is empty)
4. `GET /jobs/{job_id}/status` → check overall job progress

**Note:** The `preview_id` from `POST /agent/structure/preview` is NOT a valid job_id here.
    """,
)
async def get_queue_status(job_id: str):
    from app.agent.structuring_agent import get_queue_length
    queued = get_queue_length(job_id)
    return BatchQueueStatus(
        job_id=job_id,
        queued_claims=queued,
        batch_size=settings.AGENT_BATCH_SIZE,
        message=(
            f"{queued} claims queued. POST /agent/queue/{job_id}/next to process next batch."
            if queued > 0 else "Queue empty — all claims processed."
        ),
    )


# ── POST /agent/queue/{job_id}/next ──────────────────────────────────────────

@router.post(
    "/queue/{job_id}/next",
    response_model=PipelineResponse,
    summary="Process next batch from Redis queue",
    description="""
Dequeues the next `AGENT_BATCH_SIZE` (default 5) claims from Redis for this job
and runs the full inference + DB persistence pipeline synchronously.

**This only works with `job_id` values returned by `POST /api/v1/upload`.**
The `preview_id` from the preview endpoint cannot be used here.

Call repeatedly until `GET /agent/queue/{job_id}` returns `queued_claims: 0`.

Example for a 15-row CSV (batch_size=5):
- Upload → job_id=abc, processes rows 1-5, queues rows 6-15
- POST /agent/queue/abc/next → processes rows 6-10, queues 11-15
- POST /agent/queue/abc/next → processes rows 11-15, queue empty
    """,
)
async def process_next_batch(
    job_id: str,
    request: Request,
    session: AsyncSession = Depends(get_async_db),
):
    t0 = time.time()
    user_id = getattr(request.state, "user_id", None) or "anonymous"

    from app.agent.structuring_agent import dequeue_batch, get_queue_length
    from app.agent.orchestrator import run_orchestrator
    from app.agent.nodes import validation_node
    from app.agent.state import PipelineState
    from app.services.model_registry import get_model_registry

    claims = dequeue_batch(job_id)
    if not claims:
        raise HTTPException(
            status_code=404,
            detail=f"No queued claims for job {job_id}. Queue may be empty or expired (24h TTL).",
        )

    dummy_state: PipelineState = {
        "job_id": job_id, "file_path": "", "file_type": "csv",
        "user_id": user_id, "raw_text_blocks": [], "ocr_full_text": "",
        "structured_claims": claims, "structuring_summary": {},
        "validated_claims": [], "failed_claims": [], "queued_count": 0,
        "fraud_flags": [], "error": None,
    }
    validated_state = validation_node(dummy_state)
    validated = validated_state.get("validated_claims", [])
    pre_failed = validated_state.get("failed_claims", [])

    loop = asyncio.get_event_loop()
    registry = await loop.run_in_executor(None, get_model_registry)

    orch = await run_orchestrator(
        validated_claims=validated,
        job_id=job_id,
        user_id=user_id,
        session=session,
        registry=registry,
    )

    remaining = get_queue_length(job_id)
    processed = len(orch.get("results", []))
    failed = len(pre_failed) + len(orch.get("failed_details", []))

    return PipelineResponse(
        job_id=job_id,
        status="partial" if remaining > 0 else "completed",
        filename=f"queue_batch_{job_id[:8]}",
        file_type="queued_batch",
        total_processed=processed,
        total_failed=failed,
        queued_count=remaining,
        models_triggered=orch.get("models_triggered", {}),
        results=[ClaimInferenceResult(**r) for r in orch.get("results", [])],
        failed_details=pre_failed + orch.get("failed_details", []),
        processing_time=round(time.time() - t0, 3),
        poll_url=f"/api/v1/jobs/{job_id}/status",
    )
