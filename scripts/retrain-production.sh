#!/usr/bin/env bash
# Retrain all sale models on the production server (requires crawl DATABASE_URL).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${GITHUB_ACTIONS:-}" != "true" && "${LOCAL_DEPLOY:-}" != "true" ]]; then
  echo "ERROR: set LOCAL_DEPLOY=true to retrain on the production server." >&2
  exit 1
fi

if [[ -f "${ROOT}/scripts/deploy_via_ssh.py" ]] && ! command -v sshpass >/dev/null 2>&1; then
  exec "${ROOT}/.venv/bin/python" "${ROOT}/scripts/retrain_via_ssh.py" "$@"
fi

ENV_REMOTE="${ROOT}/../didibood_crawler_divar/.env.remote"
if [[ -f "$ENV_REMOTE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_REMOTE" 2>/dev/null || true
fi

DEPLOY_HOST="${DEPLOY_HOST:-${CRAWL_SSH_HOST:?Set DEPLOY_HOST or CRAWL_SSH_HOST}}"
DEPLOY_USER="${DEPLOY_USER:-${CRAWL_SSH_USER:-ubuntu}}"
SSH_IDENTITY="${DEPLOY_SSH_IDENTITY_FILE:-${CRAWL_SSH_IDENTITY_FILE:-$HOME/.ssh/didibood_deploy_ed25519}}"
DEPLOY_PATH="${PRICE_DEPLOY_PATH:-/opt/didibood/Didibood_Price}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
if [[ -f "$SSH_IDENTITY" ]]; then
  SSH_OPTS+=(-i "$SSH_IDENTITY" -o IdentitiesOnly=yes -o BatchMode=yes)
fi

PURPOSE="${1:-sale}"
REFRESH_FLAG=""
if [[ "${REFRESH_DATA:-true}" == "true" ]]; then
  REFRESH_FLAG="--refresh-data"
fi

echo "==> Retrain Didibood Price models on ${DEPLOY_USER}@${DEPLOY_HOST} (purpose=${PURPOSE})"
ssh "${SSH_OPTS[@]}" "${DEPLOY_USER}@${DEPLOY_HOST}" \
  "cd '${DEPLOY_PATH}' && .venv/bin/python -m pricing train-all --purpose '${PURPOSE}' ${REFRESH_FLAG}"

echo "==> Done. Restart service to load new models if needed."
ssh "${SSH_OPTS[@]}" "${DEPLOY_USER}@${DEPLOY_HOST}" "sudo systemctl restart didibood-price"
echo "==> didibood-price restarted"
