#!/usr/bin/env bash
# Deploy Didibood Price to production (rsync + venv + systemd).
#
# From your Mac:
#   export DEPLOY_HOST=37.32.12.208
#   export DEPLOY_USER=ubuntu
#   LOCAL_DEPLOY=true ./scripts/deploy-production.sh
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" && "${LOCAL_DEPLOY:-}" != "true" ]]; then
  echo "ERROR: set LOCAL_DEPLOY=true for manual deploy from your machine." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Optional: reuse crawl server credentials from didibood_crawler_divar/.env.remote
ENV_REMOTE="${ROOT}/../didibood_crawler_divar/.env.remote"
if [[ -f "$ENV_REMOTE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line//$'\r'/}"
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    case "$key" in
      CRAWL_SSH_USER) : "${DEPLOY_SSH_USER:=${val}}" ;;
      CRAWL_SSH_PASSWORD) : "${DEPLOY_SSH_PASSWORD:=${val}}" ;;
      CRAWL_SSH_IDENTITY_FILE) : "${DEPLOY_SSH_IDENTITY_FILE:=${val}}" ;;
      PRICE_DEPLOY_HOST) : "${DEPLOY_HOST:=${val}}" ;;
    esac
  done < "$ENV_REMOTE"
fi

: "${DEPLOY_HOST:=37.32.12.208}"
DEPLOY_HOST="${DEPLOY_HOST:?Set DEPLOY_HOST or PRICE_DEPLOY_HOST}"
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
SSH_IDENTITY="${DEPLOY_SSH_IDENTITY_FILE:-${CRAWL_SSH_IDENTITY_FILE:-$HOME/.ssh/didibood_deploy_ed25519}}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
RSYNC_SSH=(ssh "${SSH_OPTS[@]}")
if [[ -f "$SSH_IDENTITY" ]]; then
  SSH_OPTS+=(-i "$SSH_IDENTITY" -o IdentitiesOnly=yes -o BatchMode=yes)
elif [[ -n "${DEPLOY_SSH_PASSWORD:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "==> sshpass not found; using Python deploy (paramiko)"
    exec "${ROOT}/.venv/bin/python" "${ROOT}/scripts/deploy_via_ssh.py"
  fi
  export SSHPASS="$DEPLOY_SSH_PASSWORD"
  SSH_OPTS+=(-o PreferredAuthentications=password -o PubkeyAuthentication=no)
  RSYNC_SSH=(sshpass -e ssh "${SSH_OPTS[@]}")
else
  SSH_OPTS+=(-o BatchMode=yes)
fi

ssh_cmd() {
  if [[ -n "${DEPLOY_SSH_PASSWORD:-}" && ! -f "$SSH_IDENTITY" ]]; then
    sshpass -e ssh "${SSH_OPTS[@]}" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$@"
  fi
}

_didibood_resolve_deploy_path() {
  local canonical="$1"
  local override_var="$2"
  local override_val="${!override_var-}"
  if [[ -n "${override_val}" ]]; then
    printf '%s\n' "${override_val}"
    return 0
  fi
  if [[ -n "${DEPLOY_PATH:-}" && "${DEPLOY_PATH}" != "${canonical}" ]]; then
    echo "==> WARN: ignoring foreign DEPLOY_PATH=${DEPLOY_PATH}; using ${canonical}" >&2
  fi
  printf '%s\n' "${canonical}"
}

DEPLOY_PATH="$(_didibood_resolve_deploy_path "/opt/didibood/Didibood_Price" "PRICE_DEPLOY_PATH")"
RSYNC_EXCLUDES="${ROOT}/scripts/rsync-excludes.txt"

echo "==> Sync Didibood_Price → ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}"
rsync -az --delete \
  --exclude-from="$RSYNC_EXCLUDES" \
  -e "${RSYNC_SSH[*]}" \
  "${ROOT}/" "${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}/"

echo "==> Install and restart on server"
ssh_cmd "${DEPLOY_USER}@${DEPLOY_HOST}" \
  "DEPLOY_PATH='${DEPLOY_PATH}' DEPLOY_USER='${DEPLOY_USER}' bash -s" <<'REMOTE'
set -euo pipefail
cd "$DEPLOY_PATH"

sudo mkdir -p "$DEPLOY_PATH"
sudo chown -R "$(whoami):$(id -gn)" .

set_env_var() {
  local key="$1"
  local val="$2"
  if grep -q "^${key}=" .env 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" .env
  else
    echo "${key}=${val}" >> .env
  fi
}

if [[ ! -f .env ]]; then
  cp .env.production.example .env
  chmod 600 .env
  echo "    Created .env from .env.production.example"
fi

APP_ROOT="$(dirname "$DEPLOY_PATH")"
for crawl_env in \
  "${APP_ROOT}/Didibood_Crawler_Divar/.env" \
  "${APP_ROOT}/Didibood_Crawler/.env"; do
  if [[ -f "$crawl_env" ]]; then
    db_url="$(grep '^DATABASE_URL=' "$crawl_env" | cut -d= -f2- || true)"
    if [[ -z "$db_url" ]]; then
      db_url="$(grep '^CRAWL_DATABASE_URL=' "$crawl_env" | cut -d= -f2- || true)"
    fi
    if [[ -n "$db_url" ]]; then
      set_env_var DATABASE_URL "$db_url"
      echo "    DATABASE_URL from $(basename "$(dirname "$crawl_env")")"
      break
    fi
  fi
done

set_env_var HOST 0.0.0.0
set_env_var PORT 8093
set_env_var DEBUG false
chmod 600 .env

mkdir -p artifacts/models artifacts/datasets

echo "==> Python venv + dependencies"
if [[ -d .venv ]]; then
  sudo chown -R "$(whoami):$(id -gn)" .venv
fi
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "==> systemd unit"
UNIT_SRC="deploy/systemd/didibood-price.service"
UNIT_TMP="/tmp/didibood-price.service"
sed "s|__DEPLOY_USER__|${DEPLOY_USER}|g" "$UNIT_SRC" > "$UNIT_TMP"
sudo cp "$UNIT_TMP" /etc/systemd/system/didibood-price.service
sudo systemctl daemon-reload
sudo systemctl enable didibood-price

GW="$(docker network ls --format '{{.Name}}' 2>/dev/null | grep -i gateway | grep _default | head -1 | xargs -I{} docker network inspect {} -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)"
if [[ -n "$GW" && "$GW" != "<no value>" ]]; then
  if command -v ufw >/dev/null 2>&1; then
    SUBNET="$(docker network ls --format '{{.Name}}' 2>/dev/null | grep -i gateway | grep _default | head -1 | xargs -I{} docker network inspect {} -f '{{(index .IPAM.Config 0).Subnet}}' 2>/dev/null || true)"
    if [[ -n "$SUBNET" ]]; then
      sudo ufw allow from "$SUBNET" to any port 8093 comment 'didibood gateway price' >/dev/null 2>&1 || true
    fi
  fi
fi

sudo systemctl restart didibood-price

echo "    Waiting for Price health..."
for i in $(seq 1 30); do
  if curl -sf --max-time 5 "http://127.0.0.1:8093/health" >/dev/null; then
    echo "==> Didibood Price deploy OK (attempt ${i})"
    exit 0
  fi
  sleep 2
done

echo "Price health check failed"
sudo journalctl -u didibood-price -n 40 --no-pager || true
exit 1
REMOTE

echo "Done: Price health at http://${DEPLOY_HOST}:8093/health (loopback on server)"
