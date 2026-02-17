# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /build

# Copy dependency manifest first to maximize layer-cache hits.
COPY requirements.txt ./

# BuildKit cache keeps pip downloads between builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt


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

RUN chown -R app:app /app

USER app

EXPOSE 8000

# APP_MODULE/PORT/WEB_CONCURRENCY can be overridden at runtime.
CMD ["sh", "-c", "exec uvicorn \"$APP_MODULE\" --host 0.0.0.0 --port \"$PORT\" --workers \"$WEB_CONCURRENCY\""]
