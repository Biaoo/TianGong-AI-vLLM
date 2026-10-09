"""Download the pinned ModelScope snapshot with resumable curl and SHA256 checks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def read_env() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.exists():
        return dict(os.environ)
    return {
        **{
            key.strip(): value.strip().strip("\"'")
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line
            for key, value in [line.split("=", 1)]
        },
        **os.environ,
    }


def verified(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]


def fetch(directory: Path, manifest: dict, entry: dict, verify_only: bool) -> None:
    path = directory / entry["path"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Manifest contains a path outside the model directory")
    if verified(path, entry):
        print(f"Verified {entry['path']}", flush=True)
        return
    if verify_only:
        raise RuntimeError(f"Missing or corrupt model file: {entry['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    url = (
        f"https://modelscope.cn/models/{manifest['model_id']}/resolve/"
        f"{manifest['revision']}/{quote(entry['path'])}"
    )
    print(f"Downloading {entry['path']}", flush=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    if not verified(partial, entry):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch: {entry['path']}; run download again")
    partial.replace(path)
    print(f"Verified {entry['path']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--group", choices=("model", "embedding"), default="model")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    embedding = args.group == "embedding"
    directory = args.directory or ROOT / read_env().get(
        "EMBEDDING_MODEL_DIR" if embedding else "MODEL_DIR",
        "models/Qwen3-Embedding-8B" if embedding else "models/Qwen3.8-Flash-Next-NVFP4",
    )
    if not args.verify_only:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = f"deploy/{'embedding' if embedding else 'vllm'}/model-manifest.json"
    manifest = json.loads((args.manifest or ROOT / manifest_path).read_text())
    lock_context = nullcontext() if args.verify_only else (directory / ".download.lock").open("w")
    with lock_context as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(
                pool.map(
                    lambda entry: fetch(directory, manifest, entry, args.verify_only),
                    manifest["files"],
                )
            )
    print(f"Snapshot verified: {manifest['model_id']} @ {manifest['revision']}")


if __name__ == "__main__":
    main()
