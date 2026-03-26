"""LangGraph pipeline nodes — OCR → structuring → validation → inference.

Node responsibilities:
  ocr_node          — run PaddleOCR on image/PDF, produce full_text
  structuring_node  — use gpt-5.4-nano (OCR input) or pandas (CSV/Excel) to
                      extract canonical claim fields + attach model_validation
  validation_node   — field-level validation, range checks, date checks
  inference_node    — placeholder; actual inference handled in graph.py
"""

import logging
import time
from typing import Any, Dict, List

from app.agent.state import PipelineState
from app.agent.structuring_agent import (
    get_structuring_summary,
    structure_from_file,
    structure_from_ocr,
)
from app.services.validation_service import validate_claim

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}
FILE_EXTENSIONS = {"csv", "xlsx", "xls"}


# ── OCR node ──────────────────────────────────────────────────────────────────

def ocr_node(state: PipelineState) -> PipelineState:
    """
    Run PaddleOCR on image or PDF. Produces raw_text_blocks and ocr_full_text.
    Skipped for CSV/Excel — those go directly to structuring_node.
    """
    t0 = time.time()
    file_path = state["file_path"]
    file_type = state.get("file_type", "pdf").lower().lstrip(".")
    logger.info(f"[ocr_node] START file={file_path} type={file_type}")

    # CSV/Excel — skip OCR entirely
    if file_type in FILE_EXTENSIONS:
        logger.info(f"[ocr_node] SKIP — file type={file_type} goes directly to structuring")
        return {**state, "raw_text_blocks": [], "ocr_full_text": "", "error": None}

    try:
        from app.services.ocr_service import get_ocr_service
        ocr = get_ocr_service()

        if file_type in IMAGE_EXTENSIONS:
            text_blocks = ocr.process_image(file_path, page_number=1)
        else:
            text_blocks = ocr.process_pdf(file_path)

        full_text = "\n".join(b.text for b in text_blocks)
        elapsed = time.time() - t0
        logger.info(
            f"[ocr_node] DONE blocks={len(text_blocks)} "
            f"text_len={len(full_text)} elapsed={elapsed:.3f}s"
        )
        return {**state, "raw_text_blocks": text_blocks, "ocr_full_text": full_text, "error": None}

    except Exception as e:
        logger.error(f"[ocr_node] ERROR elapsed={time.time()-t0:.3f}s: {e}", exc_info=True)
        return {**state, "raw_text_blocks": [], "ocr_full_text": "", "error": str(e)}


# ── Structuring node ──────────────────────────────────────────────────────────

def structuring_node(state: PipelineState) -> PipelineState:
    """
    Structure raw input into canonical claim dicts with model_validation attached.

    - CSV/Excel: pandas column normalisation, no LLM call
    - OCR text: gpt-5.4-nano extraction
    """
    t0 = time.time()
    file_type = state.get("file_type", "").lower().lstrip(".")
    job_id = state.get("job_id", "")
    logger.info(f"[structuring_node] START job={job_id} file_type={file_type}")

    if state.get("error"):
        logger.warning(f"[structuring_node] SKIP — upstream error: {state['error']}")
        return state

    failed_claims: List[Dict] = list(state.get("failed_claims", []))
    queued_count = 0

    if file_type in FILE_EXTENSIONS:
        # CSV / Excel path — no LLM
        structured, queued_count = structure_from_file(
            file_path=state["file_path"],
            job_id=job_id,
        )
    else:
        # OCR text path — LLM extraction
        full_text = state.get("ocr_full_text", "")
        if not full_text.strip():
            failed_claims.append({"reason": "structuring_failed: no OCR text available"})
            logger.warning("[structuring_node] SKIP — empty OCR text")
            return {
                **state,
                "structured_claims": [],
                "structuring_summary": {},
                "failed_claims": failed_claims,
                "queued_count": 0,
            }
        structured = structure_from_ocr(full_text=full_text, job_id=job_id)

    summary = get_structuring_summary(structured)
    elapsed = time.time() - t0
    logger.info(
        f"[structuring_node] DONE job={job_id} "
        f"claims={len(structured)} queued={queued_count} "
        f"summary={summary} elapsed={elapsed:.3f}s"
    )
    return {
        **state,
        "structured_claims": structured,
        "structuring_summary": summary,
        "failed_claims": failed_claims,
        "queued_count": queued_count,
    }


# ── Validation node ───────────────────────────────────────────────────────────

def validation_node(state: PipelineState) -> PipelineState:
    """
    Validate each structured claim.
    Only validates fields required by at least one eligible model.
    Valid claims → validated_claims; failures → failed_claims.
    """
    t0 = time.time()
    n_in = len(state.get("structured_claims", []))
    logger.info(f"[validation_node] START claims_in={n_in}")

    if state.get("error"):
        logger.warning(f"[validation_node] SKIP — upstream error: {state['error']}")
        return state

    validated: List[Dict] = []
    failed: List[Dict] = list(state.get("failed_claims", []))
    seen_ids: set = set()

    for claim in state.get("structured_claims", []):
        mv = claim.get("_model_validation", {})

        # If no model is eligible, still pass through — agent will report it
        # but don't hard-fail; let the orchestrator decide
        if not mv.get("any_eligible"):
            logger.info(
                f"[validation_node] claim_id={claim.get('claim_id')} "
                f"no eligible models — passing through with warning"
            )
            claim["_validation_warning"] = "No models eligible for this claim"
            validated.append(claim)
            continue

        is_valid, reason, enriched = validate_claim(claim, seen_ids)
        if is_valid:
            validated.append(enriched)
            seen_ids.add(str(enriched["claim_id"]))
        else:
            failed.append({"claim_id": claim.get("claim_id"), "reason": reason})

    elapsed = time.time() - t0
    logger.info(
        f"[validation_node] DONE valid={len(validated)} "
        f"failed={len(failed)} elapsed={elapsed:.3f}s"
    )
    return {**state, "validated_claims": validated, "failed_claims": failed}


# ── Inference node ────────────────────────────────────────────────────────────

def inference_node(state: PipelineState) -> PipelineState:
    """
    Placeholder — actual inference and DB persistence handled in graph.py
    after the graph completes (needs async session + model registry).
    """
    return {**state, "fraud_flags": state.get("fraud_flags", [])}
