#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
base=tiangong-h3-fl2va:ee37ec06-fp8-r1
expected=sha256:96c018e4a2cd678c0acfa8ee1cc7e400d2aa6af2234adb2bd348697116ff0f28
image_tag=${1:-tiangong-h3-heretic-fl2va:ee37ec06-fp8-r1}
[[ $# -le 1 ]] || exit 2
[[ $(docker image inspect --format '{{.Id}}' "$base") == "$expected" ]] || exit 1
if docker image inspect "$image_tag" >/dev/null 2>&1; then
  echo 'Refusing to overwrite an existing Heretic FL2VA image tag' >&2
  exit 1
fi
context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
mkdir -p "$context/deploy/h3-heretic-fl2va" "$context/scripts"
cp "$root"/deploy/h3-heretic-fl2va/{Dockerfile,serve.py,release.json} "$context/deploy/h3-heretic-fl2va/"
cp "$root/scripts/prepare_h3_heretic.py" "$context/scripts/"
docker build --pull=false --network none --build-arg "H3_BASE_IMAGE=$base" \
  -f "$context/deploy/h3-heretic-fl2va/Dockerfile" -t "$image_tag" "$context"
