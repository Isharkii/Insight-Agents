"""
Simple container health check for HTTP backends.
"""

from __future__ import annotations

import os
import sys
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    port = os.getenv("PORT", "8000")
    path = os.getenv("HEALTHCHECK_PATH", "/health")
    url = f"http://127.0.0.1:{port}{path}"

    try:
        with urlopen(url, timeout=2) as response:
            return 0 if 200 <= response.status < 400 else 1
    except (URLError, TimeoutError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
