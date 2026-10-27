"""Verify the pinned native FL2VA partition, then serve first/last-frame video."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

MODEL_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


def validate_model(model: Path) -> Path:
    model = model.resolve()
    if model.name != "FL2VA" or model.parent.name != MODEL_REVISION:
        raise ValueError("Expected the pinned native FL2VA partition")
    metadata = json.loads((model / "model_index.json").read_text())["_minimax_h3"]
    if metadata.get("partition") != "fl2va" or "fl2va" not in metadata.get("tasks", []):
        raise ValueError("Checkpoint does not declare FL2VA support")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--allowed-clients", required=True)
    parser.add_argument("--memory-fraction", default=0.30, type=float)
    parser.add_argument("--read-only-model-root", required=True, type=Path)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 0.5:
        parser.error("Memory fraction must be in (0, 0.5]")
    try:
        model = validate_model(args.model)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    mount_root = args.read_only_model_root.resolve()
    if not model.is_relative_to(mount_root) or not os.path.ismount(mount_root):
        parser.error("Expected a dedicated model bind mount containing FL2VA")
    subprocess.run(
        ["mount", "-o", "remount,bind,ro", "--", str(mount_root)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            "/opt/h3/download_hf_snapshot.py",
            "--verify-only",
            "--model",
            "MiniMaxAI/MiniMax-H3",
            "--revision",
            MODEL_REVISION,
            "--directory",
            str(model.parent),
            "--include-prefix",
            "FL2VA/",
            "--include-prefix",
            "model_index.json",
        ],
        check=True,
    )
    os.environ.update(
        {
            "H3_ALLOWED_CLIENTS": args.allowed_clients,
            "H3_TORCH_MEMORY_FRACTION": str(args.memory_fraction),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "7200",
        }
    )
    command = [
        "vllm",
        "serve",
        str(model),
        "--omni",
        "--config",
        "/opt/h3-fl2va/quality.yaml",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--middleware",
        "h3_runtime.AllowedClients",
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
