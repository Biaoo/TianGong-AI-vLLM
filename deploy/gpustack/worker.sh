#!/usr/bin/env bash
# CDI worker only. Configuration and Docker errors can contain credentials.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
private="$root/.secrets/gpustack-worker/worker.env"
action=${1:-config}
if [[ $# -gt 1 ]]; then
  echo "Usage: worker.sh {config|pull|start|restart|stop|status}" >&2
  exit 2
fi
case "$action" in
  config) args=(config --quiet) ;;
  pull) args=(pull worker) ;;
  start) args=(up -d --no-recreate --no-build --pull never worker) ;;
  restart) args=(up -d --force-recreate --no-build --pull never worker) ;;
  stop) args=(stop worker) ;;
  status) args=(ps --all worker) ;;
  *) echo "Unknown worker action" >&2; exit 2 ;;
esac
if [[ ! -f "$private" || -L "$private" || $(stat -c '%a' "$private") != 600 ]]; then
  echo "Worker configuration must be a regular private file with mode 600" >&2
  exit 1
fi
umask 077
log=$(mktemp "$root/.secrets/gpustack-worker/command.XXXXXX")
trap 'rm -f -- "$log"' EXIT
if ! docker compose --project-directory "$root" --env-file "$private" \
  -f "$root/deploy/gpustack/worker.compose.yaml" "${args[@]}" >"$log" 2>&1; then
  # Retain errors locally without leaking rendered values into shared output.
  mv -- "$log" "$root/.secrets/gpustack-worker/last-error.log"
  echo "Worker $action failed; inspect .secrets/gpustack-worker/last-error.log locally" >&2
  exit 1
fi
if [[ "$action" == status ]]; then cat -- "$log"; else echo "Worker $action: OK"; fi
