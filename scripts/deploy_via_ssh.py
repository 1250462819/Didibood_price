#!/usr/bin/env python3
"""Deploy Didibood Price via SSH (password/key) when sshpass is unavailable."""
from __future__ import annotations

import base64
import fnmatch
import os
import stat
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEPLOY_PATH = "/opt/didibood/Didibood_Price"
DEFAULT_SERVICE_USER = "ubuntu"


def load_env_remote() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT.parent / "didibood_crawler_divar" / ".env.remote"
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().replace("\r", "")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def load_excludes() -> list[str]:
    path = ROOT / "scripts" / "rsync-excludes.txt"
    if not path.is_file():
        return []
    return [
        line.strip().replace("\r", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def excluded(rel: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if rel.startswith(pattern.rstrip("/")) or f"/{pattern.rstrip('/')}/" in f"/{rel}/":
                return True
        elif fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(Path(rel).name, pattern):
            return True
    return False


def connect(host: str, user: str, password: str | None, key_file: str | None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = {"hostname": host, "username": user, "timeout": 30}
    if key_file and Path(key_file).expanduser().is_file():
        kwargs["key_filename"] = str(Path(key_file).expanduser())
        kwargs["look_for_keys"] = False
        kwargs["allow_agent"] = False
    elif password:
        kwargs["password"] = password
    else:
        kwargs["look_for_keys"] = True
        kwargs["allow_agent"] = True
    client.connect(**kwargs)
    return client


def run_remote(client: paramiko.SSHClient, script: str, *, get_pty: bool = False) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command("bash -s", get_pty=get_pty)
    assert stdin is not None
    stdin.write(script)
    stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_sudo(client: paramiko.SSHClient, script: str, sudo_password: str) -> tuple[int, str, str]:
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
    cmd = (
        f"echo {sh_quote(sudo_password)} | sudo -S -p '' "
        f"bash -c 'echo {payload} | base64 -d | bash -s'"
    )
    _, stdout, stderr = client.exec_command(cmd, get_pty=False)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def ensure_subdir(sftp: paramiko.SFTPClient, base: str, rel_dir: str) -> None:
    if not rel_dir or rel_dir == ".":
        return
    current = base.rstrip("/")
    for part in rel_dir.split("/"):
        current = f"{current}/{part}"
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def sync_tree(sftp: paramiko.SFTPClient, local: Path, remote: str, patterns: list[str]) -> int:
    sftp.stat(remote)
    count = 0
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local).as_posix()
        if excluded(rel, patterns):
            continue
        target = f"{remote.rstrip('/')}/{rel}"
        parent_rel = str(Path(rel).parent.as_posix())
        if parent_rel != ".":
            ensure_subdir(sftp, remote, parent_rel)
        sftp.put(str(path), target)
        mode = path.stat().st_mode
        if mode & stat.S_IXUSR:
            sftp.chmod(target, mode & 0o777)
        count += 1
        if count % 50 == 0:
            print(f"    uploaded {count} files...", flush=True)
    return count


def build_remote_script(deploy_path: str, service_user: str) -> str:
    dp = sh_quote(deploy_path)
    su = sh_quote(service_user)
    prefix = f"set -euo pipefail\nDEPLOY_PATH={dp}\nSERVICE_USER={su}\n"
    body = r"""
mkdir -p "$DEPLOY_PATH/artifacts/models" "$DEPLOY_PATH/artifacts/datasets"

if [[ ! -f "$DEPLOY_PATH/.env" ]]; then
  cp "$DEPLOY_PATH/.env.production.example" "$DEPLOY_PATH/.env"
  chmod 600 "$DEPLOY_PATH/.env"
  echo "    Created .env from .env.production.example"
fi

APP_ROOT="$(dirname "$DEPLOY_PATH")"
for crawl_env in \
  "$APP_ROOT/Didibood_Crawler_Divar/.env" \
  "$APP_ROOT/Didibood_Crawler/.env"; do
  if [[ -f "$crawl_env" ]]; then
    db_url="$(grep '^DATABASE_URL=' "$crawl_env" | cut -d= -f2- || true)"
    if [[ -z "$db_url" ]]; then
      db_url="$(grep '^CRAWL_DATABASE_URL=' "$crawl_env" | cut -d= -f2- || true)"
    fi
    if [[ -n "$db_url" ]]; then
      if grep -q '^DATABASE_URL=' "$DEPLOY_PATH/.env"; then
        sed -i "s|^DATABASE_URL=.*|DATABASE_URL=${db_url}|" "$DEPLOY_PATH/.env"
      else
        echo "DATABASE_URL=${db_url}" >> "$DEPLOY_PATH/.env"
      fi
      echo "    DATABASE_URL from $(basename "$(dirname "$crawl_env")")"
      break
    fi
  fi
done

set_env_var() {
  local key="$1"
  local val="$2"
  if grep -q "^${key}=" "$DEPLOY_PATH/.env"; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$DEPLOY_PATH/.env"
  else
    echo "${key}=${val}" >> "$DEPLOY_PATH/.env"
  fi
}

set_env_var HOST 0.0.0.0
set_env_var PORT 8093
set_env_var DEBUG false
chown "$SERVICE_USER:$SERVICE_USER" "$DEPLOY_PATH/.env"
chmod 600 "$DEPLOY_PATH/.env"

echo "==> Python venv + dependencies (as $SERVICE_USER)"
sudo -u "$SERVICE_USER" env DEPLOY_PATH="$DEPLOY_PATH" bash -s <<'VENV_EOF'
set -euo pipefail
cd "$DEPLOY_PATH"
if [[ -d .venv ]]; then
  chown -R "$(whoami):$(id -gn)" .venv
fi
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
VENV_EOF

echo "==> systemd unit"
sed "s|__DEPLOY_USER__|${SERVICE_USER}|g" "$DEPLOY_PATH/deploy/systemd/didibood-price.service" \
  > /tmp/didibood-price.service
cp /tmp/didibood-price.service /etc/systemd/system/didibood-price.service
systemctl daemon-reload
systemctl enable didibood-price

GW="$(docker network ls --format '{{.Name}}' 2>/dev/null | grep -i gateway | grep _default | head -1 | xargs -I{} docker network inspect {} -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)"
if [[ -n "$GW" && "$GW" != "<no value>" ]]; then
  if command -v ufw >/dev/null 2>&1; then
    SUBNET="$(docker network ls --format '{{.Name}}' 2>/dev/null | grep -i gateway | grep _default | head -1 | xargs -I{} docker network inspect {} -f '{{(index .IPAM.Config 0).Subnet}}' 2>/dev/null || true)"
    if [[ -n "$SUBNET" ]]; then
      ufw allow from "$SUBNET" to any port 8093 comment 'didibood gateway price' >/dev/null 2>&1 || true
    fi
  fi
fi

systemctl restart didibood-price

echo "    Waiting for Price health..."
for i in $(seq 1 30); do
  if curl -sf --max-time 5 "http://127.0.0.1:8093/health" >/dev/null; then
    echo "==> Didibood Price deploy OK (attempt ${i})"
    exit 0
  fi
  sleep 2
done

echo "Price health check failed"
journalctl -u didibood-price -n 40 --no-pager || true
exit 1
"""
    return prefix + body


def main() -> int:
    remote_env = load_env_remote()
    host = (
        os.environ.get("DEPLOY_HOST")
        or remote_env.get("PRICE_DEPLOY_HOST")
        or "37.32.12.208"
    )
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

    if not host:
        print("ERROR: set DEPLOY_HOST or PRICE_DEPLOY_HOST", file=sys.stderr)
        return 1

    patterns = load_excludes()
    print(f"==> Sync Didibood_Price → {ssh_user}@{host}:{deploy_path} (runs as {service_user})")

    client = connect(host, ssh_user, password, key_file)
    try:
        if sudo_password:
            bootstrap = (
                f"mkdir -p {sh_quote(deploy_path)} && "
                f"chown -R {sh_quote(ssh_user)}:{sh_quote(ssh_user)} {sh_quote(deploy_path)}\n"
            )
            code, out, err = run_sudo(client, bootstrap, sudo_password)
            if out:
                print(out, end="" if out.endswith("\n") else "\n")
            if code != 0:
                if err:
                    print(err, file=sys.stderr)
                return code

        sftp = client.open_sftp()
        try:
            n = sync_tree(sftp, ROOT, deploy_path, patterns)
            print(f"    uploaded {n} files")
        finally:
            sftp.close()

        if sudo_password:
            handoff = f"chown -R {sh_quote(service_user)}:{sh_quote(service_user)} {sh_quote(deploy_path)}\n"
            code, out, err = run_sudo(client, handoff, sudo_password)
            if out:
                print(out, end="" if out.endswith("\n") else "\n")
            if code != 0:
                if err:
                    print(err, file=sys.stderr)
                return code

        print("==> Install and restart on server")
        install = build_remote_script(deploy_path, service_user)
        if sudo_password:
            code, out, err = run_sudo(client, install, sudo_password)
        else:
            code, out, err = run_remote(client, install)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
        if code != 0:
            return code
    finally:
        client.close()

    print(f"Done: Price health at http://{host}:8093/health (loopback on server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
