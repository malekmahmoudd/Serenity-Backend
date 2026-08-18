# syntax=docker/dockerfile:1
#
# Built for Hugging Face Spaces' Docker SDK. HF Spaces routes traffic to
# whatever listens on port 7860 -- that convention (not $PORT, unlike Render
# or most other PaaS) is why the CMD below hardcodes it.
#
# Deliberately NOT the same image as local dev: this uses
# requirements-onnx.txt, not requirements.txt. No TensorFlow, no tf2onnx --
# those are training/export-time tools (see export_onnx.py), and the whole
# point of task 10 was to stop paying their ~600MB cost at serve time. Confirmed
# working end to end with this exact dependency set in verify_slim_boot.py
# before this Dockerfile was written -- not an assumption, a tested one.
#
# ONNX model must exist before building: run export_onnx.py locally first
# (needs the full requirements.txt / TensorFlow environment), then this
# Dockerfile copies the resulting artifacts/ directory in. This image never
# installs TensorFlow, so it cannot run export_onnx.py itself.

FROM python:3.12-slim

# HF Spaces containers run as a non-root user by convention; matches the
# official HF Docker Space quickstart so file permissions on /app don't
# fight the platform.
RUN useradd -m -u 1000 appuser
WORKDIR /app

# System deps for cryptography/psycopg's C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=appuser requirements-onnx.txt .
RUN pip install --no-cache-dir -r requirements-onnx.txt

# Application code. Explicitly NOT `COPY . .` -- see .dockerignore for what
# that would otherwise pull in (venv/, __pycache__/, serenity.db, .env,
# model_v5.keras, test_*.py, export_onnx.py). None of that belongs in a
# production image; most of it (serenity.db, .env) actively should not leave
# this machine.
COPY --chown=appuser *.py .
COPY --chown=appuser artifacts/model_v5.onnx artifacts/model_v5.onnx
COPY --chown=appuser artifacts/tokenizer.json artifacts/tokenizer.json
COPY --chown=appuser artifacts/calibrators.pkl artifacts/calibrators.pkl
COPY --chown=appuser artifacts/thresholds.json artifacts/thresholds.json
COPY --chown=appuser artifacts/label_map.json artifacts/label_map.json
COPY --chown=appuser artifacts/preprocessing_config.json artifacts/preprocessing_config.json
COPY --chown=appuser artifacts/crisis_patterns.json artifacts/crisis_patterns.json

USER appuser

# HF Spaces expects a health check on the root or a defined port; /health
# already exists and answers even in degraded mode (see classifier.py's
# design), so a HEAD request against it is a real health check, not a stub.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=5)" || exit 1

EXPOSE 7860

# No --reload (that is a dev-only uvicorn flag -- the opposite of what a
# stable public deployment wants), single worker (the in-process rate
# limiter, session cache and metrics counters in ratelimit.py/sessions.py/
# metrics.py all assume one process; see their own docstrings on this).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
