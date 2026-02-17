# Production Docker Deployment Strategy

This project can be deployed in production as a single backend container with an external managed PostgreSQL instance.

## 1) Container strategy

- Build once and publish immutable image tags (example: `ghcr.io/your-org/insight-agent:1.0.0`).
- Run only the backend container in production.
- Do not run PostgreSQL inside your production compose stack.
- Keep application state external (managed DB, object storage, managed filesystems).

Use:

```powershell
docker compose -f docker-compose.prod.yml --env-file .env.production up -d
```

## 2) Managed PostgreSQL

- Set `DATABASE_URL` to your managed provider endpoint.
- Prefer TLS by appending `?sslmode=require` (or provider equivalent).
- Restrict DB network access to app nodes only.
- Rotate credentials using your platform secret manager, not checked-in files.

## 3) Environment-based configuration

Config comes from env vars only:

- `.env.production` for non-secret defaults
- Platform/secret manager for sensitive values (`DATABASE_URL`, API keys)

Baseline vars are documented in `.env.production.example`.

## 4) Logging strategy

- Write all logs to stdout/stderr from the app container.
- Configure the container runtime for log rotation to prevent disk growth.
- Forward logs to your platform collector (CloudWatch, Datadog, ELK, Loki, etc).
- Include request ID/correlation ID in app logs for traceability.

`docker-compose.prod.yml` sets:

- `logging.driver: json-file`
- `max-size: 10m`
- `max-file: 5`

## 5) Health checks

- Container healthcheck runs `python /app/scripts/healthcheck.py`.
- The script probes `http://127.0.0.1:${PORT}${HEALTHCHECK_PATH}`.
- Implement `HEALTHCHECK_PATH` endpoint in your backend (default `/health`).
- Recommended behavior:
  - Liveness: return 200 if process is running.
  - Readiness: optionally verify DB connectivity before returning 200.

## 6) Restart policies

For plain Docker/Compose:

- `restart: unless-stopped` for backend service.
- Keep `stop_grace_period` for clean shutdowns and in-flight request drain.

Equivalent intent on other platforms:

- Kubernetes: `restartPolicy: Always` + liveness/readiness probes.
- ECS/Nomad/Systemd: service-level auto-restart on failure.

## 7) Deployment workflow

1. Build image in CI.
2. Run tests and migrations validation in CI.
3. Push versioned image tag.
4. Run DB migration job (`alembic upgrade head`) against managed DB.
5. Roll out container with updated image tag.
6. Monitor health, error rate, and DB connection saturation.
