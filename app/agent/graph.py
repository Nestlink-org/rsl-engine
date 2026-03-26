"""LangGraph pipeline — OCR/CSV/Excel → structuring → validation → orchestrator → DB.

Flow:
  ocr_node → structuring_node → validation_node → inference_node → END
  (inference_node is a placeholder; real inference + DB write done by run_orchestrator)

Supports:
  - PDF / image files: OCR → gpt-5.4-nano structuring → validation → orchestrator
  - CSV / Excel files: pandas structuring → validation → orchestrator
"""

import logging
import time
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agent.nodes import inference_node, ocr_node, structuring_node, validation_node
from app.agent.state import PipelineState
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def build_pipeline():
    """Build and compile the LangGraph processing pipeline."""
    graph = StateGraph(PipelineState)
    graph.add_node("ocr", ocr_node)
    graph.add_node("structuring", structuring_node)
    graph.add_node("validation", validation_node)
    graph.add_node("inference", inference_node)

    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "structuring")
    graph.add_edge("structuring", "validation")
    graph.add_edge("validation", "inference")
    graph.add_edge("inference", END)

    return graph.compile()


# Aliases for backward compatibility
build_ocr_pipeline = build_pipeline
build_pdf_pipeline = build_pipeline

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


get_ocr_pipeline = get_pipeline


async def run_pipeline(
    file_path: str,
    file_type: str,
    job_id: str,
    user_id: str,
    session: AsyncSession,
    registry: ModelRegistry,
) -> Dict[str, Any]:
    """
    Run the full pipeline for any supported file type and persist results to DB.

    Routing:
      - csv / xlsx / xls  → structuring_node (pandas, no OCR)
      - pdf / jpg / jpeg / png / bmp / tiff → ocr_node → structuring_node (LLM)

    Returns:
        {"all": [...], "failed": [...], "orchestrator_summary": {...}}
    """
    from app.agent.orchestrator import run_orchestrator

    initial_state: PipelineState = {
        "job_id": job_id,
        "file_path": file_path,
        "file_type": file_type,
        "user_id": user_id,
        "raw_text_blocks": [],
        "ocr_full_text": "",
        "structured_claims": [],
        "structuring_summary": {},
        "validated_claims": [],
        "failed_claims": [],
        "queued_count": 0,
        "fraud_flags": [],
        "error": None,
    }

    t0 = time.time()
    logger.info(f"[pipeline] START job={job_id} file={file_path} type={file_type}")

    pipeline = get_pipeline()
    final_state = await pipeline.ainvoke(initial_state)

    logger.info(
        f"[pipeline] graph done elapsed={time.time()-t0:.3f}s "
        f"validated={len(final_state.get('validated_claims', []))} "
        f"failed={len(final_state.get('failed_claims', []))} "
        f"queued={final_state.get('queued_count', 0)}"
    )

    if final_state.get("error"):
        logger.error(f"[pipeline] pipeline error job={job_id}: {final_state['error']}")
        return {
            "all": [],
            "failed": [{"reason": f"pipeline_error: {final_state['error']}"}],
            "orchestrator_summary": {},
        }

    validated = final_state.get("validated_claims", [])
    pre_failed = list(final_state.get("failed_claims", []))

    if not validated:
        logger.warning(f"[pipeline] no validated claims job={job_id}")
        return {
            "all": [],
            "failed": pre_failed,
            "orchestrator_summary": {"queued_count": final_state.get("queued_count", 0)},
        }

    # Run orchestrator — triggers eligible models and persists to DB
    orch_summary = await run_orchestrator(
        validated_claims=validated,
        job_id=job_id,
        user_id=user_id,
        session=session,
        registry=registry,
    )

    all_results = orch_summary.get("results", [])
    failed_results = pre_failed + orch_summary.get("failed_details", [])

    elapsed = time.time() - t0
    logger.info(
        f"[pipeline] DONE job={job_id} processed={len(all_results)} "
        f"failed={len(failed_results)} elapsed={elapsed:.3f}s"
    )
    return {
        "all": all_results,
        "failed": failed_results,
        "orchestrator_summary": orch_summary,
        "queued_count": final_state.get("queued_count", 0),
    }


# Backward-compatible aliases
async def run_ocr_pipeline(
    file_path: str,
    file_type: str,
    job_id: str,
    user_id: str,
    session: AsyncSession,
    registry: ModelRegistry,
) -> Dict[str, Any]:
    return await run_pipeline(file_path, file_type, job_id, user_id, session, registry)


async def run_pdf_pipeline(
    file_path: str,
    job_id: str,
    user_id: str,
    session: AsyncSession,
    registry: ModelRegistry,
) -> Dict[str, Any]:
    return await run_pipeline(file_path, "pdf", job_id, user_id, session, registry)
