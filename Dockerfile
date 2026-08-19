# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app

# System dependencies for cryptography/psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --chown=appuser requirements-onnx.txt .
RUN pip install --no-cache-dir -r requirements-onnx.txt

# Copy application code and ONNX artifacts
COPY --chown=appuser *.py .
COPY --chown=appuser artifacts/model_v5.onnx artifacts/model_v5.onnx
COPY --chown=appuser artifacts/tokenizer.json artifacts/tokenizer.json
COPY --chown=appuser artifacts/calibrators.pkl artifacts/calibrators.pkl
COPY --chown=appuser artifacts/thresholds.json artifacts/thresholds.json
COPY --chown=appuser artifacts/label_map.json artifacts/label_map.json
COPY --chown=appuser artifacts/preprocessing_config.json artifacts/preprocessing_config.json
COPY --chown=appuser artifacts/crisis_patterns.json artifacts/crisis_patterns.json

USER appuser

# Railway will inject $PORT, so expose a generic port
EXPOSE 8000

# Healthcheck against /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request, os; \
    urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health', timeout=5)" || exit 1

# Start FastAPI app with uvicorn, binding to $PORT
CMD ["python", "main.py"]
