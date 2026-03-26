"""Application's configuration"""

from pathlib import Path
from typing import List, Optional
from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

env_path = Path(__file__).resolve().parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True
    )

    # OCR Settings
    OCR_LANG: str = "en"
    USE_GPU: bool = False
    USE_ANGLE_CLS: bool = True
    DET_DB_BOX_THRESH: float = 0.6

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_API_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-5.4-nano"

    # Agent Settings
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TIMEOUT: int = 120
    AGENT_MEMORY_WINDOW: int = 20
    AGENT_BATCH_SIZE: int = 5  # max claims per batch before Redis queuing

    # Database
    PSQL_URI: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rsl"
    PSQL_URI_ASYNC: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rsl"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "rsl-engine"

    # ML Models
    MODELS_DIR: str = "models"
    MAX_BATCH_SIZE: int = 10000

    # Server Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB upload limit
    # Structured formats (CSV pipeline) + unstructured formats (OCR pipeline)
    ALLOWED_EXTENSIONS: list = [".csv", ".xlsx", ".xls", ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    UPLOAD_DIR: str = "uploads"

    # Performance
    OCR_BATCH_SIZE: int = 4

settings = Settings()