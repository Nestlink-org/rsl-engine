"""OCR endpoints — extract text from images and PDFs via PaddleOCR.

Endpoints:
  POST /api/v1/ocr/extract          — single file (image or PDF), returns OCRResponse
  POST /api/v1/ocr/batch            — multiple files, processes batch, returns batch_id + summary
  GET  /api/v1/ocr/batch/{batch_id} — retrieve stored batch results from Redis
  GET  /api/v1/ocr/cache/{file_hash} — check if a file hash is already cached
"""

import hashlib
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.ocr import OCRResponse, ErrorResponse
from app.services.ocr_service import (
    OCRService,
    _cache_get,
    OCR_BATCH_PREFIX,
    OCR_CACHE_PREFIX,
    get_ocr_service,
    init_ocr_service,
)
from app.utils.pdf_handler import get_pdf_page_count

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | {".pdf"}


def _file_hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@router.post(
    "/extract",
    response_model=OCRResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Extract text from a single image or PDF",
)
async def ocr_extract(
    file: UploadFile = File(..., description="Image (jpg/png/bmp/tiff) or PDF"),
):
    """
    Upload a single image or PDF and extract all text blocks via PaddleOCR.

    - Results are cached in Redis by file SHA-256 (TTL 24h).
    - On cache hit the stored result is returned immediately without re-running OCR.
    - Returns structured TextBlock list plus joined full_text.
    - First cold request takes ~20-60s while PaddleOCR initialises.
    """
    t0 = time.time()
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[-1].lower()

    logger.info(f"[ocr/extract] START filename={filename} ext={ext}")

    if ext not in ALLOWED_EXTS:
        logger.warning(f"[ocr/extract] rejected unsupported ext={ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        logger.warning(f"[ocr/extract] file too large size={len(content)}")
        raise HTTPException(status_code=413, detail="File exceeds size limit")

    file_hash = _file_hash_bytes(content)
    logger.info(f"[ocr/extract] file_hash={file_hash[:16]}... size={len(content)} bytes")

    # Check Redis cache before saving to disk
    is_pdf = ext == ".pdf"
    cache_suffix = "pdf" if is_pdf else "p1"
    cache_key = f"{OCR_CACHE_PREFIX}{file_hash}:{cache_suffix}"

    def _check_cache():
        return _cache_get(cache_key)

    cached = await run_in_threadpool(_check_cache)
    if cached:
        from app.schemas.ocr import TextBlock
        blocks = [TextBlock(**b) for b in cached["blocks"]]
        full_text = "\n".join(b.text for b in blocks)
        page_count = max((b.page_number for b in blocks), default=1)
        elapsed = time.time() - t0
        logger.info(
            f"[ocr/extract] CACHE HIT filename={filename} "
            f"blocks={len(blocks)} elapsed={elapsed:.3f}s"
        )
        return OCRResponse(
            success=True,
            filename=filename,
            file_type="pdf" if is_pdf else "image",
            page_count=page_count,
            text_blocks=blocks,
            full_text=full_text,
            processing_time=round(elapsed, 3),
            timestamp=datetime.now(),
            cached=True,
            file_hash=file_hash,
        )

    # Save to disk for processing
    file_path = None
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, f"ocr_{uuid.uuid4()}{ext}")
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"[ocr/extract] saved to {file_path}")

        # Run OCR in thread pool (blocking CPU-bound call)
        def _run():
            ocr = get_ocr_service()
            if is_pdf:
                blocks = ocr.process_pdf(file_path, use_cache=True)
                pages = get_pdf_page_count(file_path)
            else:
                blocks = ocr.process_image(file_path, page_number=1, use_cache=True)
                pages = 1
            return blocks, pages

        text_blocks, page_count = await run_in_threadpool(_run)
        full_text = "\n".join(b.text for b in text_blocks)
        elapsed = time.time() - t0

        logger.info(
            f"[ocr/extract] DONE filename={filename} "
            f"blocks={len(text_blocks)} pages={page_count} elapsed={elapsed:.3f}s"
        )

        return OCRResponse(
            success=True,
            filename=filename,
            file_type="pdf" if is_pdf else "image",
            page_count=page_count,
            text_blocks=text_blocks,
            full_text=full_text,
            processing_time=round(elapsed, 3),
            timestamp=datetime.now(),
            cached=False,
            file_hash=file_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ocr/extract] FAILED filename={filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {e}")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"[ocr/extract] cleaned up {file_path}")
            except Exception as ce:
                logger.warning(f"[ocr/extract] cleanup failed {file_path}: {ce}")


@router.post(
    "/batch",
    summary="Extract text from multiple files (batch)",
)
async def ocr_batch(
    files: List[UploadFile] = File(..., description="One or more images or PDFs"),
):
    """
    Upload multiple files for batch OCR processing.

    - Each file is processed sequentially in a thread pool.
    - Full batch result is stored in Redis under `ocr:batch:{batch_id}` (TTL 24h).
    - Returns batch_id and per-file summary.
    - Retrieve full results via GET /ocr/batch/{batch_id}.
    """
    t0 = time.time()
    batch_id = str(uuid.uuid4())
    logger.info(f"[ocr/batch] START batch_id={batch_id} files={len(files)}")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved_paths: List[str] = []
    file_names: List[str] = []

    for file in files:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in ALLOWED_EXTS:
            logger.warning(f"[ocr/batch] skipping unsupported file={filename}")
            continue
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            logger.warning(f"[ocr/batch] skipping oversized file={filename}")
            continue
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(settings.UPLOAD_DIR, f"ocr_batch_{uuid.uuid4()}{ext}")
        with open(save_path, "wb") as f:
            f.write(content)
        saved_paths.append(save_path)
        file_names.append(filename)
        logger.info(f"[ocr/batch] saved file={filename} path={save_path}")

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files to process")

    def _run_batch():
        ocr = get_ocr_service()
        return ocr.process_batch(saved_paths, batch_id)

    try:
        summary = await run_in_threadpool(_run_batch)
    finally:
        for p in saved_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"[ocr/batch] cleanup failed {p}: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"[ocr/batch] DONE batch_id={batch_id} "
        f"files={len(saved_paths)} elapsed={elapsed:.3f}s"
    )

    return {
        "batch_id": batch_id,
        "total_files": summary["total_files"],
        "succeeded": summary["succeeded"],
        "failed": summary["failed"],
        "total_blocks": summary["total_blocks"],
        "total_elapsed": summary["total_elapsed"],
        "retrieve_url": f"/api/v1/ocr/batch/{batch_id}",
        "results_summary": [
            {
                "file": r["file"],
                "status": r["status"],
                "block_count": r.get("block_count", 0),
                "processing_time": r["processing_time"],
                "error": r.get("error"),
            }
            for r in summary["results"]
        ],
    }


@router.get(
    "/batch/{batch_id}",
    summary="Retrieve stored batch OCR results from Redis",
)
async def ocr_batch_result(batch_id: str):
    """
    Retrieve full batch OCR results stored in Redis.
    Results expire after 24 hours.
    """
    logger.info(f"[ocr/batch/{batch_id}] retrieving from Redis")
    batch_key = f"{OCR_BATCH_PREFIX}{batch_id}"

    def _fetch():
        return _cache_get(batch_key)

    result = await run_in_threadpool(_fetch)
    if result is None:
        logger.warning(f"[ocr/batch/{batch_id}] not found in Redis")
        raise HTTPException(
            status_code=404,
            detail=f"Batch {batch_id} not found. Results may have expired (TTL 24h).",
        )
    logger.info(
        f"[ocr/batch/{batch_id}] found "
        f"files={result.get('total_files')} blocks={result.get('total_blocks')}"
    )
    return result


@router.get(
    "/cache/{file_hash}",
    summary="Check if a file hash has a cached OCR result",
)
async def ocr_cache_check(file_hash: str):
    """Check whether a file's OCR result is already cached in Redis."""
    image_key = f"{OCR_CACHE_PREFIX}{file_hash}:p1"
    pdf_key = f"{OCR_CACHE_PREFIX}{file_hash}:pdf"

    def _fetch():
        img = _cache_get(image_key)
        if img:
            return {"cached": True, "type": "image", "block_count": len(img.get("blocks", []))}
        pdf = _cache_get(pdf_key)
        if pdf:
            return {"cached": True, "type": "pdf", "block_count": len(pdf.get("blocks", []))}
        return {"cached": False, "file_hash": file_hash}

    return await run_in_threadpool(_fetch)
