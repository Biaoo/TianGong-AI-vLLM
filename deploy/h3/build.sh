#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: build.sh <vllm-omni-checkout> [new-image-tag]' >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_commit=ee37ec0620a99b3787dab5e5e3bd1f5cdfd0615f
image_tag=${2:-tiangong-h3:ee37ec06-vllm032-r6}
if docker image inspect "$image_tag" >/dev/null 2>&1; then
  echo 'Refusing to overwrite an existing H3 image tag' >&2
  exit 1
fi
git -C "$1" cat-file -e "$source_commit^{commit}"
mkdir -p "$root/output"
context=$(mktemp -d "$root/output/h3-build.XXXXXX")
trap 'rm -rf -- "$context"' EXIT
mkdir "$context/upstream"
git -C "$1" archive "$source_commit" | tar -xf - -C "$context/upstream"
cp -- "$root/deploy/h3/"{Dockerfile,serve.py,h3_runtime.py,quality.yaml,runtime-constraints.txt} "$context/"
cp -- "$root/scripts/download_hf_snapshot.py" "$context/"
extra_hosts=()
if [[ -n ${H3_PACKAGE_IPV4:-} ]]; then
  # Optional, task-local workaround for an unavailable IPv6/DNS path.
  extra_hosts+=(--add-host "mirrors.aliyun.com:$H3_PACKAGE_IPV4")
fi
docker build --pull=false --network host "${extra_hosts[@]}" -t "$image_tag" "$context"
