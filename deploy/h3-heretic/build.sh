#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
base=tiangong-h3:ee37ec06-vllm032-r6
expected=sha256:7c15800cfc4c259d77218eca5c6283dbbedc90dbe180b25a54524c3ded3dd5a0
image_tag=${1:-tiangong-h3-heretic:ee37ec06-fp8-r2}
[[ $# -le 1 ]] || exit 2
[[ $(docker image inspect --format '{{.Id}}' "$base") == "$expected" ]] || exit 1
if docker image inspect "$image_tag" >/dev/null 2>&1; then
  echo 'Refusing to overwrite an existing Heretic image tag' >&2
  exit 1
fi
context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
mkdir -p "$context/deploy/h3-heretic" "$context/scripts"
cp "$root"/deploy/h3-heretic/{Dockerfile,serve.py,quality.yaml,release.json,patch_cpu_init.py} "$context/deploy/h3-heretic/"
cp "$root/scripts/prepare_h3_heretic.py" "$context/scripts/"
docker build --pull=false --network none --build-arg "H3_BASE_IMAGE=$base" \
  -f "$context/deploy/h3-heretic/Dockerfile" -t "$image_tag" "$context"
