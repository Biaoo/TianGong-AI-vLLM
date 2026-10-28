"""Serve the pinned Heretic encoder / FastH3 video assembly offline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--allowed-clients", required=True)
    parser.add_argument("--memory-fraction", default=0.30, type=float)
    parser.add_argument("--read-only-model-root", required=True, type=Path)
    args = parser.parse_args()
    model, root = args.model.resolve(), args.read_only_model_root.resolve()
    if not 0 < args.memory_fraction <= 0.5 or not model.is_relative_to(root):
        parser.error("Invalid model location or allocator budget")
    if not os.path.ismount(root):
        parser.error("Expected a dedicated model bind mount")
    subprocess.run(["mount", "-o", "remount,bind,ro", "--", str(root)], check=True)
    release = json.loads(Path("/opt/h3-heretic/release.json").read_text())
    subprocess.run(
        [
            sys.executable,
            "/opt/h3-heretic/prepare_h3_heretic.py",
            "--verify-only",
            "--output",
            str(model),
            "--expected-manifest-sha256",
            release["assembly_manifest_sha256"],
            "--allowed-root",
            str(root),
        ],
        check=True,
    )
    base_revision = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
    subprocess.run(
        [
            sys.executable,
            "/opt/h3/download_hf_snapshot.py",
            "--verify-only",
            "--model",
            "MiniMaxAI/MiniMax-H3",
            "--revision",
            base_revision,
            "--directory",
            str(root / "hub/models--MiniMaxAI--MiniMax-H3/snapshots" / base_revision),
            "--include-prefix",
            "Ref2VA/video_vae/",
            "--include-prefix",
            "Ref2VA/audio_vae/",
        ],
        check=True,
    )
    os.environ.update(
        {
            "H3_ALLOWED_CLIENTS": args.allowed_clients,
            "H3_TORCH_MEMORY_FRACTION": str(args.memory_fraction),
            "HF_HUB_CACHE": str(root / "hub"),
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
        "/opt/h3-heretic/quality.yaml",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--middleware",
        "h3_runtime.AllowedClients",
        "--diffusion-quantization-config",
        json.dumps({"text_encoder": {"method": "fp8"}}),
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
