from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Job(SQLModel, table=True):
    __tablename__ = "job"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(unique=True, index=True)          # UUID string
    user_id: str = Field(index=True)
    filename: str
    file_type: str                                         # csv | xlsx | xls | pdf
    status: str = Field(default="pending")                 # pending|processing|completed|failed|partial
    total_claims: int = Field(default=0)
    processed_claims: int = Field(default=0)
    failed_claims: int = Field(default=0)
    error_detail: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
