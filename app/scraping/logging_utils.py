"""
Structured logging helpers for scraping workflows.
"""

from __future__ import annotations

import json
import logging
from typing import Any


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """
    Emit one structured log line as compact JSON.
    """

    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, default=str, sort_keys=True))
