# InsightAgent: Local Database Command Guide

This guide lists the commands you can run **after your local PostgreSQL database is created**.

## 1) Set environment config (PowerShell)

Create a local env file from the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```env
ENVIRONMENT=local
DATABASE_URL=postgresql+psycopg://<user>:<password>@localhost:5432/<db_name>
```

If you prefer split URLs, use `LOCAL_DATABASE_URL` instead of `DATABASE_URL`.

## 2) Install dependencies

```powershell
py -m pip install -r requirements.txt
```

## 3) Generate and run migrations

Create migration from model changes:

```powershell
py -m alembic revision --autogenerate -m "describe change"
```

Apply all pending migrations:

```powershell
py -m alembic upgrade head
```

Apply next single migration:

```powershell
py -m alembic upgrade +1
```

Rollback one migration:

```powershell
py -m alembic downgrade -1
```

Rollback to base (empty schema state managed by Alembic):

```powershell
py -m alembic downgrade base
```

## 4) Migration inspection commands

Show current applied revision:

```powershell
py -m alembic current
```

Show migration history:

```powershell
py -m alembic history
```

Show current head revision(s):

```powershell
py -m alembic heads
```

Show details of one revision:

```powershell
py -m alembic show <revision_id>
```

## 5) Special commands

Mark DB as a revision **without running SQL** (use carefully):

```powershell
py -m alembic stamp head
```

Run against a one-off DB URL (overrides env URL for one command):

```powershell
py -m alembic -x db_url="postgresql+psycopg://<user>:<password>@localhost:5432/<db_name>" upgrade head
```

## 6) Recommended workflow for every schema change

1. Update SQLAlchemy models in `db/models/`.
2. Generate migration:
   ```powershell
   py -m alembic revision --autogenerate -m "your change message"
   ```
3. Review generated migration in `alembic/versions/`.
4. Apply migration:
   ```powershell
   py -m alembic upgrade head
   ```

## 7) Quick sanity check

After upgrade, confirm Alembic sees the DB at head:

```powershell
py -m alembic current
py -m alembic heads
```

If `current` equals `head`, your local schema is up to date.

## 8) Competitor scraping framework

The competitor scraping pipeline is available at:

- `POST /ingest-competitors`
- Optional query filter: `competitor=<name>`
- CLI: `py scripts/run_competitor_scrape.py --competitor acme_cloud`

Config-driven competitor registration is in:

- `app/scraping/config/competitors.json`

Architecture notes:

- `docs/competitor-scraping-framework.md`

## 9) Dynamic CSV schema mapping

CSV ingestion now supports dynamic schema mapping with:

- Fuzzy column auto-detection
- Manual override mappings
- DB-backed mapping configs (`mapping_configs` table)
- Structured mapping errors for invalid/missing canonical mappings

Use optional query params on `POST /upload-csv`:

- `client_name=<client>` to load an active client-scoped mapping config
- `mapping_config_name=<config>` to load a specific active mapping config
- Architecture notes: `docs/dynamic-schema-mapping.md`
