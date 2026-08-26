#!/usr/bin/env bash
# Run Didibood Price locally (predict + optional train if DATABASE_URL works).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.local.example .env
  echo "Created .env from .env.local.example"
  echo "Edit DATABASE_URL only if you need train/refresh-data from this machine."
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

mkdir -p artifacts/models artifacts/datasets

echo "Starting Didibood Price at http://127.0.0.1:${PORT:-8093}/docs"
exec .venv/bin/python main.py
