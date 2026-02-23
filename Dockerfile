# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /build

# Copy all dependency manifests first to maximize layer-cache hits.
COPY requirements.txt requirements-core.txt requirements-optional.txt requirements-dev.txt ./

# BuildKit cache keeps pip downloads between builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PORT=8000 \
    APP_MODULE=app.main:app \
    WEB_CONCURRENCY=1

# Non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

# Copy resolved dependencies from builder stage only.
COPY --from=builder /opt/venv /opt/venv

# Copy application source after dependency layers for faster rebuilds.
COPY . /app

# Make entrypoint executable.
RUN chmod +x /app/scripts/docker-entrypoint.sh

RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "/app/scripts/healthcheck.py"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# APP_MODULE/PORT/WEB_CONCURRENCY can be overridden at runtime.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
