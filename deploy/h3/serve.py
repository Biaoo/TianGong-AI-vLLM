"""Verify fixed H3 components, then launch the upstream video API offline."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

MODEL_REVISION = "f01e6a23ac6066aa9eb622cf94b6259442e70ec2"
BASE_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--allowed-clients", required=True)
    parser.add_argument("--memory-fraction", default=0.35, type=float)
    parser.add_argument("--decoder", choices=["quality", "fast"], default="quality")
    parser.add_argument("--read-only-model-mount", action="store_true")
    args = parser.parse_args()
    model = args.model.resolve()
    if model.name != MODEL_REVISION or not 0 < args.memory_fraction <= 0.5:
        parser.error("Expected the fixed model revision and a valid allocator budget")
    parent = model.parent
    base = parent / "hub/models--MiniMaxAI--MiniMax-H3/snapshots" / BASE_REVISION
    verifier = [sys.executable, "/opt/h3/download_hf_snapshot.py", "--verify-only"]
    subprocess.run(
        verifier
        + [
            "--model",
            "FastVideo/FastH3-Omni-NVFP4-QAD",
            "--revision",
            MODEL_REVISION,
            "--directory",
            str(model),
            "--exclude-prefix",
            "transformer/",
        ],
        check=True,
    )
    subprocess.run(
        verifier
        + [
            "--model",
            "MiniMaxAI/MiniMax-H3",
            "--revision",
            BASE_REVISION,
            "--directory",
            str(base),
            "--include-prefix",
            "Ref2VA/video_vae/",
            "--include-prefix",
            "Ref2VA/audio_vae/",
        ],
        check=True,
    )
    if args.read_only_model_mount:
        if not os.path.ismount(parent):
            parser.error("Expected a dedicated model parent bind mount")
        subprocess.run(
            ["mount", "-o", "remount,bind,ro", "--", str(parent)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    os.environ.update(
        {
            "H3_ALLOWED_CLIENTS": args.allowed_clients,
            "H3_TORCH_MEMORY_FRACTION": str(args.memory_fraction),
            "HF_HUB_CACHE": str(parent / "hub"),
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
        "/opt/h3/quality.yaml",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--middleware",
        "h3_runtime.AllowedClients",
    ]
    if args.decoder == "fast":
        import json

        command += [
            "--additional-config",
            json.dumps(
                {
                    "video_vae_decoder": str(
                        model / "vae_decoder_nvfp4_fast/decoder-nvfp4.safetensors"
                    )
                }
            ),
        ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
