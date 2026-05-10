# Production ML Pipeline Dockerfile
# Multi-stage build for optimized image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Production
# ============================================
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser config/ ./config/

# Create necessary directories
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command - API server
CMD ["python", "-m", "uvicorn", "src.mlProject.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# ============================================
# Stage 3: MLflow Server (optional)
# ============================================
FROM python:3.10-slim as mlflow-server

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow[extras] sqlalchemy psycopg2-binary

WORKDIR /app

# MLflow default artifact root
ENV MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@localhost:5432/mlflow
ENV MLFLOW_ARTIFACT_ROOT=s3://ml-pipeline-artifacts/

EXPOSE 5000

CMD ["mlflow", "ui", "--backend-store-uri", "${MLFLOW_TRACKING_URI}", "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}"]

# ============================================
# Build instructions
# ============================================
# Build production image:
# docker build -t ml-pipeline-api:latest --target production .
#
# Build MLflow server:
# docker build -t mlflow-server:latest --target mlflow-server .
#
# Or build all targets:
# docker build -t ml-pipeline-api:latest -t mlflow-server:latest --target production --target mlflow-server .