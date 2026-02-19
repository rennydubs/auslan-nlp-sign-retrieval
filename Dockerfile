# ============================================================
# Auslan Sign Retrieval System — Backend Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder)  — installs Python deps into /install
#   Stage 2 (runtime)  — CUDA runtime image, copies /install
#
# Build:
#   docker build -t auslan-backend .
# Run (standalone):
#   docker run --gpus all -p 8000:8000 auslan-backend
# ============================================================

# ----------------------------------------------------------------
# Stage 1: Builder — install all Python dependencies
# ----------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /install

# System build tools needed by some packages (e.g. tokenizers, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a separate prefix so we can COPY it cleanly
RUN pip install --prefix=/install/packages --no-cache-dir -r requirements.txt

# Download spaCy model at build time so the container is self-contained
RUN pip install spacy && \
    python -m spacy download en_core_web_sm --direct

# Pre-download NLTK corpora (WordNet)
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# ----------------------------------------------------------------
# Stage 2: Runtime — CUDA-enabled, slim image
# ----------------------------------------------------------------
# Use CUDA 12.1 + cuDNN 8 for GPU inference (sentence-transformers/torch).
# Falls back gracefully to CPU if no GPU is present.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

# System Python + runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install/packages /usr/local

# Copy spaCy model data
COPY --from=builder /usr/local/lib/python3.11/dist-packages/en_core_web_sm \
                    /usr/local/lib/python3.11/dist-packages/en_core_web_sm

# Copy NLTK data
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy application source
COPY . .

# Create cache directory for embedding storage
RUN mkdir -p /app/.cache /app/media/videos

# ----------------------------------------------------------------
# Environment
# ----------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Sentence-transformers cache (persisted by volume in docker-compose)
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    # Ollama host (override at runtime if using external Ollama)
    OLLAMA_HOST=http://ollama:11434 \
    # FastAPI settings
    HOST=0.0.0.0 \
    PORT=8000

# ----------------------------------------------------------------
# Healthcheck — poll the /api/health endpoint
# ----------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

EXPOSE 8000

# ----------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
