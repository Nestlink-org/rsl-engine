# Multi-stage Dockerfile for AEGIS-X Backend
# Stage 1: Builder stage for dependencies
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install dependencies
RUN uv venv && \
    uv sync --frozen --no-dev

# Stage 2: Runtime stage
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 aegis && \
    mkdir -p /app /app/uploads && \
    chown -R aegis:aegis /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV UPLOAD_DIR="/app/uploads"
ENV MAX_UPLOAD_SIZE_MB=100
ENV MAX_CONCURRENT_PIPELINES=10

# Copy application code
COPY --chown=aegis:aegis ./app ./app
COPY --chown=aegis:aegis alembic.ini ./

# Copy ML models (if they exist)
COPY --chown=aegis:aegis app/models/ ./app/models/

# Create necessary directories
RUN mkdir -p /app/uploads && \
    chown -R aegis:aegis /app/uploads

# Switch to non-root user
USER aegis

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


