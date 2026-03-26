"""OCR Service — PaddleOCR 2.9.1 wrapper with Redis result caching.

Features:
  - Single image processing (JPEG/PNG/BMP/TIFF)
  - Multi-page PDF processing via PyMuPDF
  - Redis caching: results keyed by SHA-256 of file content (TTL 24h)
  - Batch processing: process multiple files, results stored in Redis
  - Full structured logging at every stage
"""

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.schemas.ocr import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# Redis TTL for cached OCR results (24 hours)
OCR_CACHE_TTL = 86400
OCR_CACHE_PREFIX = "ocr:result:"
OCR_BATCH_PREFIX = "ocr:batch:"


def _file_hash(file_path: str) -> str:
    """Compute SHA-256 of file content for cache keying."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_redis():
    """Lazy Redis client — returns None if Redis is unavailable."""
    try:
        import redis
        from app.core.config import settings
        client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"[ocr_service] Redis unavailable — caching disabled: {e}")
        return None


def _cache_get(key: str) -> Optional[Dict]:
    """Fetch cached OCR result from Redis. Returns None on miss or error."""
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if raw:
            logger.info(f"[ocr_service] cache HIT key={key[:20]}...")
            return json.loads(raw)
        logger.debug(f"[ocr_service] cache MISS key={key[:20]}...")
        return None
    except Exception as e:
        logger.warning(f"[ocr_service] cache GET error: {e}")
        return None


def _cache_set(key: str, data: Dict, ttl: int = OCR_CACHE_TTL) -> None:
    """Store OCR result in Redis."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.set(key, json.dumps(data), ex=ttl)
        logger.info(f"[ocr_service] cache SET key={key[:20]}... ttl={ttl}s")
    except Exception as e:
        logger.warning(f"[ocr_service] cache SET error: {e}")


def _cache_delete(key: str) -> None:
    """Delete a cached result from Redis."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(key)
    except Exception as e:
        logger.warning(f"[ocr_service] cache DELETE error: {e}")


class OCRService:
    """PaddleOCR 2.9.1 service with Redis caching and batch support."""

    def __init__(self):
        self.ocr_engine = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialise PaddleOCR engine. Raises on failure."""
        t0 = time.time()
        logger.info("[ocr_service] Initialising PaddleOCR engine...")
        try:
            from paddleocr import PaddleOCR
            from app.core.config import settings

            # PaddleOCR 2.9.1 — only pass supported kwargs
            self.ocr_engine = PaddleOCR(
                lang=settings.OCR_LANG,
                use_gpu=settings.USE_GPU,
                show_log=False,
            )
            logger.info(f"[ocr_service] PaddleOCR engine ready elapsed={time.time()-t0:.2f}s")
        except Exception as e:
            logger.error(f"[ocr_service] PaddleOCR init FAILED: {e}", exc_info=True)
            raise RuntimeError(f"OCR engine initialisation failed: {e}") from e

    def _run_ocr(self, image: Union[str, np.ndarray]) -> List:
        """Run OCR engine and return raw result list."""
        if self.ocr_engine is None:
            raise RuntimeError("OCR engine is not initialised")
        try:
            result = self.ocr_engine.ocr(image, cls=True)
            return result
        except Exception as e:
            logger.error(f"[ocr_service] OCR engine error: {e}", exc_info=True)
            raise

    def _parse_result(self, result: List, page_number: int = 1) -> List[TextBlock]:
        """Convert raw PaddleOCR output to TextBlock list."""
        blocks: List[TextBlock] = []
        if not result or not result[0]:
            logger.warning(f"[ocr_service] page={page_number} — no text detected")
            return blocks

        for line in result[0]:
            try:
                box = line[0]
                text, conf = line[1][0], float(line[1][1])
                if text.strip():
                    blocks.append(TextBlock(
                        text=text.strip(),
                        confidence=round(conf, 4),
                        bounding_box=BoundingBox(points=box),
                        page_number=page_number,
                    ))
            except Exception as e:
                logger.warning(f"[ocr_service] skipping malformed block on page={page_number}: {e}")
                continue

        return blocks

    def process_image(
        self,
        image: Union[str, np.ndarray],
        page_number: int = 1,
        use_cache: bool = True,
    ) -> List[TextBlock]:
        """
        Run OCR on a single image file or numpy array.

        Args:
            image: file path string or numpy array
            page_number: page number label for TextBlock metadata
            use_cache: whether to check/store Redis cache (only for file paths)

        Returns:
            List of TextBlock objects
        """
        t0 = time.time()
        is_path = isinstance(image, str)
        cache_key = None

        if is_path:
            logger.info(f"[ocr_service] process_image START file={image} page={page_number}")
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")

            # Check cache
            if use_cache:
                file_hash = _file_hash(image)
                cache_key = f"{OCR_CACHE_PREFIX}{file_hash}:p{page_number}"
                cached = _cache_get(cache_key)
                if cached:
                    blocks = [TextBlock(**b) for b in cached["blocks"]]
                    logger.info(
                        f"[ocr_service] process_image CACHE HIT "
                        f"file={image} blocks={len(blocks)} elapsed={time.time()-t0:.3f}s"
                    )
                    return blocks
        else:
            logger.info(f"[ocr_service] process_image START array page={page_number}")

        try:
            raw = self._run_ocr(image)
            blocks = self._parse_result(raw, page_number)
            elapsed = time.time() - t0
            logger.info(
                f"[ocr_service] process_image DONE "
                f"{'file=' + image if is_path else 'array'} "
                f"page={page_number} blocks={len(blocks)} elapsed={elapsed:.3f}s"
            )

            # Store in cache
            if is_path and use_cache and cache_key:
                _cache_set(cache_key, {"blocks": [b.model_dump() for b in blocks]})

            return blocks

        except Exception as e:
            logger.error(
                f"[ocr_service] process_image FAILED "
                f"{'file=' + image if is_path else 'array'} "
                f"page={page_number} elapsed={time.time()-t0:.3f}s: {e}",
                exc_info=True,
            )
            raise

    def process_pdf(self, pdf_path: str, use_cache: bool = True) -> List[TextBlock]:
        """
        Run OCR on all pages of a PDF file.

        Args:
            pdf_path: path to PDF file
            use_cache: whether to check/store Redis cache

        Returns:
            List of TextBlock objects across all pages
        """
        t0 = time.time()
        logger.info(f"[ocr_service] process_pdf START file={pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check cache for full PDF result
        cache_key = None
        if use_cache:
            file_hash = _file_hash(pdf_path)
            cache_key = f"{OCR_CACHE_PREFIX}{file_hash}:pdf"
            cached = _cache_get(cache_key)
            if cached:
                blocks = [TextBlock(**b) for b in cached["blocks"]]
                logger.info(
                    f"[ocr_service] process_pdf CACHE HIT "
                    f"file={pdf_path} blocks={len(blocks)} elapsed={time.time()-t0:.3f}s"
                )
                return blocks

        try:
            from app.utils.pdf_handler import pdf_to_images, get_pdf_page_count
            page_count = get_pdf_page_count(pdf_path)
            logger.info(f"[ocr_service] process_pdf pages={page_count} file={pdf_path}")

            images = pdf_to_images(pdf_path)
            all_blocks: List[TextBlock] = []

            for page_num, img in enumerate(images, start=1):
                logger.info(f"[ocr_service] process_pdf processing page={page_num}/{page_count}")
                page_blocks = self.process_image(img, page_number=page_num, use_cache=False)
                all_blocks.extend(page_blocks)
                logger.info(
                    f"[ocr_service] process_pdf page={page_num} "
                    f"blocks={len(page_blocks)} cumulative={len(all_blocks)}"
                )

            elapsed = time.time() - t0
            logger.info(
                f"[ocr_service] process_pdf DONE file={pdf_path} "
                f"pages={page_count} total_blocks={len(all_blocks)} elapsed={elapsed:.3f}s"
            )

            # Cache full PDF result
            if use_cache and cache_key:
                _cache_set(cache_key, {"blocks": [b.model_dump() for b in all_blocks]})

            return all_blocks

        except Exception as e:
            logger.error(
                f"[ocr_service] process_pdf FAILED file={pdf_path} "
                f"elapsed={time.time()-t0:.3f}s: {e}",
                exc_info=True,
            )
            raise

    def process_batch(self, file_paths: List[str], batch_id: str) -> Dict[str, Any]:
        """
        Process multiple files (images/PDFs) and store results in Redis.

        Args:
            file_paths: list of absolute file paths
            batch_id: unique identifier for this batch job

        Returns:
            dict with batch summary and per-file results
        """
        t0 = time.time()
        logger.info(
            f"[ocr_service] process_batch START "
            f"batch_id={batch_id} files={len(file_paths)}"
        )

        results = []
        total_blocks = 0
        failed = 0

        for idx, file_path in enumerate(file_paths):
            file_t0 = time.time()
            filename = os.path.basename(file_path)
            logger.info(
                f"[ocr_service] batch file {idx+1}/{len(file_paths)} "
                f"batch_id={batch_id} file={filename}"
            )

            try:
                is_pdf = file_path.lower().endswith(".pdf")
                if is_pdf:
                    blocks = self.process_pdf(file_path)
                    file_type = "pdf"
                else:
                    blocks = self.process_image(file_path)
                    file_type = "image"

                full_text = "\n".join(b.text for b in blocks)
                file_result = {
                    "file": filename,
                    "file_type": file_type,
                    "status": "success",
                    "block_count": len(blocks),
                    "full_text": full_text,
                    "blocks": [b.model_dump() for b in blocks],
                    "processing_time": round(time.time() - file_t0, 3),
                }
                total_blocks += len(blocks)
                logger.info(
                    f"[ocr_service] batch file {idx+1} OK "
                    f"blocks={len(blocks)} elapsed={time.time()-file_t0:.3f}s"
                )

            except Exception as e:
                failed += 1
                file_result = {
                    "file": filename,
                    "status": "failed",
                    "error": str(e),
                    "processing_time": round(time.time() - file_t0, 3),
                }
                logger.error(
                    f"[ocr_service] batch file {idx+1} FAILED "
                    f"file={filename}: {e}",
                    exc_info=True,
                )

            results.append(file_result)

        elapsed = time.time() - t0
        batch_summary = {
            "batch_id": batch_id,
            "total_files": len(file_paths),
            "succeeded": len(file_paths) - failed,
            "failed": failed,
            "total_blocks": total_blocks,
            "total_elapsed": round(elapsed, 3),
            "results": results,
        }

        # Store batch result in Redis
        batch_key = f"{OCR_BATCH_PREFIX}{batch_id}"
        _cache_set(batch_key, batch_summary, ttl=OCR_CACHE_TTL)

        logger.info(
            f"[ocr_service] process_batch DONE batch_id={batch_id} "
            f"succeeded={len(file_paths)-failed} failed={failed} "
            f"total_blocks={total_blocks} elapsed={elapsed:.3f}s"
        )
        return batch_summary


_ocr_service_instance: Optional["OCRService"] = None
_ocr_service_lock = threading.Lock()


def get_ocr_service() -> "OCRService":
    """
    Thread-safe singleton OCR service.
    Must be pre-warmed via init_ocr_service() at startup before use in threads.
    """
    global _ocr_service_instance
    if _ocr_service_instance is None:
        with _ocr_service_lock:
            if _ocr_service_instance is None:
                logger.info("[ocr_service] Creating OCRService singleton...")
                _ocr_service_instance = OCRService()
    return _ocr_service_instance


def init_ocr_service() -> "OCRService":
    """
    Explicitly initialise the OCR singleton — call this from the main thread
    at application startup so PaddleOCR is ready before any request arrives.
    """
    return get_ocr_service()
