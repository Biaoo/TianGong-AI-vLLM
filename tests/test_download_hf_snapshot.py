"""Integrity and scope checks for the public snapshot downloader."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_hf_snapshot.py"
REVISION = "a" * 40


def run_snapshot(root, name="weights.bin", content=b"weights", extra=()):
    manifest = {
        "model_id": "public/model",
        "revision": REVISION,
        "files": [
            {
                "path": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_oid": None,
            }
        ],
    }
    (root / "snapshot-manifest.json").write_text(json.dumps(manifest))
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "public/model",
            "--revision",
            REVISION,
            "--directory",
            str(root),
            "--verify-only",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_verify_accepts_hash_and_rejects_same_size_corruption(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"weights")
    assert run_snapshot(tmp_path).returncode == 0
    assert not (tmp_path / ".snapshot.lock").exists()
    (tmp_path / "weights.bin").write_bytes(b"corrupt")
    result = run_snapshot(tmp_path)
    assert result.returncode != 0
    assert "Missing or corrupt" in result.stderr
    assert (tmp_path / "weights.bin").read_bytes() == b"corrupt"


@pytest.mark.parametrize("name", ["../escape.bin", "/tmp/escape.bin"])
def test_manifest_path_cannot_escape(tmp_path, name):
    result = run_snapshot(tmp_path, name=name)
    assert result.returncode != 0
    assert "Unsafe snapshot path" in result.stderr


def test_selection_cannot_silently_reuse_incompatible_manifest(tmp_path):
    result = run_snapshot(tmp_path, extra=("--include-prefix", "vae/"))
    assert result.returncode != 0
    assert "outside requested selection" in result.stderr


def test_rejects_mutable_revision(tmp_path):
    result = run_snapshot(tmp_path, extra=("--revision", "main"))
    assert result.returncode != 0
    assert "immutable revision" in result.stderr


def test_corrupt_mirror_download_never_becomes_final_file(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("hf_snapshot", SCRIPT)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    # Create the expected immutable manifest without downloading anything.
    run_snapshot(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model",
            "public/model",
            "--revision",
            REVISION,
            "--directory",
            str(tmp_path),
            "--transport",
            "modelscope",
            "--mirror-model",
            "mirror/model",
            "--mirror-revision",
            "b" * 40,
        ],
    )

    def corrupt_download(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"corrupt")

    monkeypatch.setattr(downloader.subprocess, "run", corrupt_download)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader.main()
    assert not (tmp_path / "weights.bin").exists()
