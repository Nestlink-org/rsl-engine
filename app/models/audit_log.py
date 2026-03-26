from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class AuditLog(SQLModel, table=True):
    __tablename__ = "auditlog"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    action: str                        # upload_file | view_claim | generate_report | chat_query
    resource_type: str                 # job | claim | facility | patient | report | chat
    resource_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    extra_data: Optional[str] = Field(default=None, sa_column_kwargs={"name": "metadata"})  # JSON-encoded dict
