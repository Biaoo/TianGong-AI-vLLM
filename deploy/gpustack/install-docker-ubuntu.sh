#!/usr/bin/env bash
# Optional host preparation. Run --check first; --install requires an interactive sudo session.
set -euo pipefail
action=${1:---check}
if [[ "$action" != --check && "$action" != --install ]]; then
  echo 'Usage: bash deploy/gpustack/install-docker-ubuntu.sh [--check|--install]' >&2
  exit 2
fi
if [[ ! -f /etc/os-release ]] || ! command -v apt-get >/dev/null; then
  echo 'This helper requires Ubuntu with apt-get.' >&2
  exit 1
fi
. /etc/os-release
if [[ "$ID" != ubuntu ]]; then
  echo 'This helper supports Ubuntu only.' >&2
  exit 1
fi
if [[ "$action" == --check ]]; then
  if command -v docker >/dev/null; then
    docker version --format '{{.Server.Version}}'
    docker compose version --short
  else
    echo 'Docker is not installed; --install requires sudo.'
    apt-cache policy docker.io docker-compose-v2
  fi
  exit 0
fi
if [[ $(id -u) != 0 || -z ${SUDO_USER:-} || ${SUDO_USER:-root} == root ]]; then
  echo 'Run this helper with sudo from the intended non-root account.' >&2
  exit 1
fi
if command -v docker >/dev/null || command -v dockerd >/dev/null; then
  echo 'An existing Docker installation was found; keep its host configuration and review manually.' >&2
  exit 1
fi
# Docker may otherwise set the forwarding policy to DROP. This host can also carry VM traffic.
# Preserve other daemon settings; never overwrite an explicit conflicting policy.
python3 - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

path = Path('/etc/docker/daemon.json')
if path.is_symlink():
    raise SystemExit('Review the existing daemon.json symlink before installation.')
data = json.loads(path.read_text()) if path.exists() else {}
if 'ip-forward-no-drop' in data and data['ip-forward-no-drop'] is not True:
    raise SystemExit('Existing forwarding policy conflicts with this helper; nothing changed.')
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = path.with_name('daemon.json.before-gpustack-' + stamp)
    with backup.open('x'):
        pass
    shutil.copy2(path, backup)
data['ip-forward-no-drop'] = True
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.daemon-')
try:
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    os.replace(temporary, path)
finally:
    Path(temporary).unlink(missing_ok=True)
PY
apt-get update
apt-get install -y docker.io docker-compose-v2
systemctl enable --now docker.service
usermod -aG docker "$SUDO_USER"
docker version --format '{{.Server.Version}}'
docker compose version --short
echo 'Docker installed. The invoking account was added to the Docker group; use a new login session.'
echo 'GPUStack and GPU workers have not been started.'
