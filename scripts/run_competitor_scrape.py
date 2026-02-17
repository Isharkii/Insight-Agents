"""
Run competitor scraping from CLI.
"""

from __future__ import annotations

import argparse
import json

from app.services.competitor_scraping_service import CompetitorScrapingService
from db.session import SessionLocal


def main() -> int:
    parser = argparse.ArgumentParser(description="Run competitor scraping ingestion.")
    parser.add_argument(
        "--competitor",
        dest="competitor",
        default=None,
        help="Optional competitor name from config file.",
    )
    args = parser.parse_args()

    service = CompetitorScrapingService()
    with SessionLocal() as db:
        summaries = service.ingest(db=db, competitor=args.competitor)

    payload = [
        {
            "competitor": summary.competitor,
            "records_scraped": summary.records_scraped,
            "records_inserted": summary.records_inserted,
            "failed_pages": summary.failed_pages,
            "status": summary.status,
            "errors": summary.errors,
        }
        for summary in summaries
    ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
