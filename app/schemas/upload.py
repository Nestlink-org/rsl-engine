from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str = "Processing started in background."
    status_url: Optional[str] = None
    results_url: Optional[str] = None
