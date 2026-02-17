# Competitor Scraping Framework

## Overview

The framework lives under `app/scraping/` and is organized by layers:

- `base.py`: `ScraperBase` shared scraping flow (robots policy, retries, rate limiting).
- `scrapers/`: domain-specific scraper subclasses.
- `parsing/`: BeautifulSoup extraction logic.
- `normalization/`: converts parsed data into `CanonicalInsightInput`.
- `storage/`: persistence abstraction and SQLAlchemy implementation.
- `config/`: JSON domain registration and environment-based runtime settings.
- `engine.py`: orchestration for multi-competitor runs.

## Config-Driven Registration

Competitors are defined in:

- `app/scraping/config/competitors.json`

Supported registration paths:

1. Built-in `scraper_type` values (`configurable`, `saas`)
2. Dynamic `scraper_class` import path (`module.path:ClassName`) for custom plugins

Adding a new competitor using built-in scrapers only requires adding one JSON entry.

## API Trigger

Run scraping via:

- `POST /ingest-competitors`
- optional query param: `competitor=<name>`
