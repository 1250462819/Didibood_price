#!/usr/bin/env python3
"""Retrain Didibood Price models on the production server via SSH."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.deploy_via_ssh import (  # noqa: E402
    connect,
    load_env_remote,
    run_remote,
    run_sudo,
    sh_quote,
)

DEFAULT_DEPLOY_PATH = "/opt/didibood/Didibood_Price"
DEFAULT_SERVICE_USER = "ubuntu"


def build_retrain_script(deploy_path: str, service_user: str, purpose: str, refresh: bool) -> str:
    dp = sh_quote(deploy_path)
    su = sh_quote(service_user)
    refresh_flag = "--refresh-data" if refresh else ""
    return f"""set -euo pipefail
DEPLOY_PATH={dp}
SERVICE_USER={su}
sudo -u "$SERVICE_USER" env DEPLOY_PATH="$DEPLOY_PATH" bash -s <<'TRAIN_EOF'
set -euo pipefail
cd "$DEPLOY_PATH"
.venv/bin/python -m pricing train-all --purpose '{purpose}' {refresh_flag}
TRAIN_EOF
systemctl restart didibood-price
for i in $(seq 1 30); do
  if curl -sf --max-time 5 "http://127.0.0.1:8093/health" >/dev/null; then
    echo "==> Retrain OK (attempt ${{i}})"
    exit 0
  fi
  sleep 2
done
echo "Retrain finished but health check failed"
journalctl -u didibood-price -n 40 --no-pager || true
exit 1
"""


def main() -> int:
    remote_env = load_env_remote()
    host = os.environ.get("DEPLOY_HOST") or remote_env.get("CRAWL_SSH_HOST")
    ssh_user = (
        os.environ.get("DEPLOY_SSH_USER")
        or os.environ.get("DEPLOY_USER")
        or remote_env.get("CRAWL_SSH_USER")
        or "ubuntu"
    )
    password = os.environ.get("DEPLOY_SSH_PASSWORD") or remote_env.get("CRAWL_SSH_PASSWORD")
    key_file = os.environ.get("DEPLOY_SSH_IDENTITY_FILE") or remote_env.get("CRAWL_SSH_IDENTITY_FILE")
    deploy_path = os.environ.get("DEPLOY_PATH") or DEFAULT_DEPLOY_PATH
    service_user = os.environ.get("SERVICE_USER") or DEFAULT_SERVICE_USER
    sudo_password = os.environ.get("DEPLOY_SUDO_PASSWORD") or password
    purpose = os.environ.get("TRAIN_PURPOSE", "sale")
    refresh = os.environ.get("REFRESH_DATA", "true").lower() != "false"

    if not host:
        print("ERROR: set DEPLOY_HOST or CRAWL_SSH_HOST", file=sys.stderr)
        return 1

    print(f"==> Retrain Didibood Price on {ssh_user}@{host} (purpose={purpose}, refresh={refresh})")
    client = connect(host, ssh_user, password, key_file)
    try:
        script = build_retrain_script(deploy_path, service_user, purpose, refresh)
        if sudo_password:
            code, out, err = run_sudo(client, script, sudo_password)
        else:
            code, out, err = run_remote(client, script)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
