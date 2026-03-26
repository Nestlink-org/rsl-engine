from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class BoundingBox(BaseModel):
    """Four corner points of the detected text region."""
    points: List[List[float]]


class TextBlock(BaseModel):
    """A single detected text region with position and confidence."""
    text: str
    confidence: float
    bounding_box: BoundingBox
    page_number: int = 1


class OCRResponse(BaseModel):
    success: bool
    filename: str
    file_type: str          # "image" | "pdf"
    page_count: int
    text_blocks: List[TextBlock]
    full_text: str          # all text joined for quick reading
    processing_time: float
    timestamp: datetime
    cached: bool = False    # True if result was served from Redis cache
    file_hash: Optional[str] = None  # SHA-256 of file content


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
