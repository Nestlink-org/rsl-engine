"""LangGraph pipeline state definitions — RSL multi-agent system."""

from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict):
    """State for the OCR → structuring → validation → inference pipeline."""
    job_id: str
    file_path: str
    file_type: str              # "pdf" | "jpg" | "jpeg" | "png" | "csv" | "xlsx"
    user_id: str

    # OCR stage
    raw_text_blocks: List[Any]          # OCR TextBlock objects
    ocr_full_text: str                  # joined full text from OCR

    # Structuring stage
    structured_claims: List[Dict]       # LLM/CSV-extracted claim dicts with _model_validation
    structuring_summary: Dict           # summary from get_structuring_summary()

    # Validation stage
    validated_claims: List[Dict]        # claims that passed field validation
    failed_claims: List[Dict]           # {"claim_id"?, "reason": str}

    # Batch queue
    queued_count: int                   # overflow claims pushed to Redis

    # Inference stage (Phase 2)
    fraud_flags: List[Dict]

    error: Optional[str]


class ChatState(TypedDict):
    """State for the conversational chat agent."""
    session_id: str
    user_id: str
    messages: List[Any]                 # LangChain message objects
    context: Optional[Dict]             # optional structured context (e.g. claim data)
