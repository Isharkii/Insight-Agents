"""Structured logging configuration for API and pipeline services."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging import LogRecord
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Render log records as one-line JSON objects."""

    def format(self, record: LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_root_logging() -> None:
    """Configure root logger with text or structured JSON formatting."""
    log_level = str(os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    log_format = str(os.getenv("LOG_FORMAT", "text")).strip().lower()
    level = getattr(logging, log_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )

    root_logger.addHandler(handler)

