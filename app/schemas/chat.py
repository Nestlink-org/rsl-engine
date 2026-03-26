from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    visualization_urls: List[str] = []
    tool_calls_made: int = 0
