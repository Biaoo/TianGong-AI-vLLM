"""Serve the verified Heretic-conditioned native H3 FL2VA/T2VA partition."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def validate_model(model):
    model = model.resolve()
    metadata = json.loads((model / "model_index.json").read_text())["_minimax_h3"]
    if (
        model.name != "FL2VA"
        or metadata.get("partition") != "fl2va"
        or not {"fl2va", "t2va"}.issubset(metadata.get("tasks", []))
    ):
        raise ValueError("Expected native FL2VA and T2VA support")
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
    root = args.read_only_model_root.resolve()
    if not 0 < args.memory_fraction <= 0.5 or not args.model.resolve().is_relative_to(root):
        parser.error("Invalid model location or allocator budget")
    if not os.path.ismount(root):
        parser.error("Expected a dedicated model bind mount")
    subprocess.run(["mount", "-o", "remount,bind,ro", "--", str(root)], check=True)
    release = json.loads(Path("/opt/h3-heretic-fl2va/release.json").read_text())
    subprocess.run(
        [
            sys.executable,
            "/opt/h3-heretic-fl2va/prepare_h3_heretic.py",
            "--verify-only",
            "--output",
            str(args.model.resolve()),
            "--expected-manifest-sha256",
            release["assembly_manifest_sha256"],
            "--allowed-root",
            str(root),
        ],
        check=True,
    )
    model = validate_model(args.model)
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
