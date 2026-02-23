#!/bin/sh
set -e

echo "==> Running database migrations..."
python -m alembic upgrade head
echo "==> Migrations complete."

echo "==> Starting application..."
exec "$@"
