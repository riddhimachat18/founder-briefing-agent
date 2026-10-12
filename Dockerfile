# ── Stage 1: dependencies ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --upgrade pip==24.3.1 \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/     ./app/
COPY static/  ./static/
COPY samples/ ./samples/

# Data directory for SQLite (writable by appuser only)
RUN mkdir -p /data && chown appuser:appgroup /data

USER appuser

ENV PYTHONUNBUFFERED=1 \
    BRIEF_DB_PATH=/data/briefs.db \
    GEMINI_MODEL=gemini-1.5-flash \
    DAILY_TOKEN_CAP=500000 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1"]
