#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo 'Usage: build.sh [new-image-tag]' >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
base=tiangong-h3:ee37ec06-vllm032-r6
expected=sha256:7c15800cfc4c259d77218eca5c6283dbbedc90dbe180b25a54524c3ded3dd5a0
image_tag=${1:-tiangong-h3-fl2va:ee37ec06-fp8-r1}
if [[ $(docker image inspect --format '{{.Id}}' "$base") != "$expected" ]]; then
  echo 'Pinned H3 base image digest does not match' >&2
  exit 1
fi
if docker image inspect "$image_tag" >/dev/null 2>&1; then
  echo 'Refusing to overwrite an existing FL2VA image tag' >&2
  exit 1
fi
docker build --pull=false --network none --build-arg "H3_BASE_IMAGE=$base" \
  -t "$image_tag" "$root/deploy/h3-fl2va"
