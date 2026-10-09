#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Preserve explicit Nemotron commands while generic instances use profile files.
case "${2:-model}" in
  embed|embed3) exec bash "$repo_root/deploy/manage-legacy.sh" "$@" ;;
esac
exec python3 "$repo_root/scripts/manage_models.py" "$@"
