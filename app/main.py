import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.middleware.jwt_middleware import JWTMiddleware
from app.routes import chat, claims, dashboard, jobs, patients, reports, upload
from app.routes.agent import router as agent_router
from app.routes.ocr import router as ocr_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_static_directories() -> None:
    for d in ["static", "static/visualizations", "static/temp"]:
        Path(d).mkdir(parents=True, exist_ok=True)


ensure_static_directories()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_static_directories()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # OCR must initialise before TensorFlow — PaddlePaddle + TF CUDA conflict
    try:
        from app.services.ocr_service import init_ocr_service
        logger.info("Initialising OCR engine (~20-40s)...")
        init_ocr_service()
        logger.info("OCR engine ready")
    except Exception as e:
        logger.warning(f"OCR init failed (non-fatal): {e}")

    # ML model registry (TF/Keras) — after OCR
    try:
        from app.services.model_registry import get_model_registry
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_model_registry)
        logger.info("Model registry loaded")
    except Exception as e:
        logger.warning(f"Model registry load failed (non-fatal): {e}")

    # Kafka (non-fatal)
    try:
        from app.services.kafka_service import get_kafka_producer
        await get_kafka_producer().start()
    except Exception as e:
        logger.warning(f"Kafka producer start failed (non-fatal): {e}")
    try:
        from app.services.kafka_service import get_kafka_consumer
        await get_kafka_consumer().start()
    except Exception as e:
        logger.warning(f"Kafka consumer start failed (non-fatal): {e}")

    logger.info("RSL Engine started")
    yield

    try:
        from app.services.kafka_service import get_kafka_producer, get_kafka_consumer
        await get_kafka_producer().stop()
        await get_kafka_consumer().stop()
    except Exception as e:
        logger.warning(f"Kafka shutdown error: {e}")
    logger.info("RSL Engine stopped")


app = FastAPI(
    title="ResultShield Lite — Fraud Detection Engine",
    description="""
AI-powered medical insurance fraud detection microservice.

## Modules

### OCR Engine
Extract text from scanned claim images and PDFs via PaddleOCR with Redis caching.

### Agent Pipeline
Structure raw claim data (CSV/Excel/OCR) → validate → run CBC fraud detection models → persist results.

### Analytics Dashboard
Comprehensive fraud analytics: trends, top facilities, anomalous patients, disease breakdown, model performance.

### Chat Assistant
Conversational AI agent with 14 tools for data queries and chart generation.

### Claims & Jobs
Track processing jobs, retrieve claim details with fraud flags.

### Reports
ROI reports in JSON or PDF format.

## Authentication
JWT token in `Authorization: Bearer <token>` header. `user_id` and `role` extracted by middleware.
Admins see all data; regular users see only their own claims.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(JWTMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

API = "/api/v1"

# ── Core pipeline ─────────────────────────────────────────────────────────────
app.include_router(upload.router,    prefix=API, tags=["Upload"])
app.include_router(agent_router,     prefix=API, tags=["Agent — Utilities"])
app.include_router(ocr_router,       prefix=API, tags=["OCR"])

# ── Results & analytics ───────────────────────────────────────────────────────
app.include_router(jobs.router,      prefix=API, tags=["Jobs"])
app.include_router(claims.router,    prefix=API, tags=["Claims"])
app.include_router(patients.router,  prefix=API, tags=["Patients"])
app.include_router(dashboard.router, prefix=API, tags=["Analytics"])
app.include_router(reports.router,   prefix=API, tags=["Reports"])

# ── Chat ──────────────────────────────────────────────────────────────────────
app.include_router(chat.router,      prefix=API, tags=["Chat"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ResultShield Lite",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": API,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "gpu": settings.USE_GPU, "ocr_lang": settings.OCR_LANG}
