"""Guard the native FL2VA partition and offline launch boundary."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "h3_fl2va", Path(__file__).resolve().parents[1] / "deploy/h3-fl2va/serve.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.fixture
def partition(tmp_path):
    model = tmp_path / runtime.MODEL_REVISION / "FL2VA"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text(
        json.dumps({"_minimax_h3": {"partition": "fl2va", "tasks": ["t2va", "fl2va"]}})
    )
    return model


def test_accepts_native_fl2va(partition):
    assert runtime.validate_model(partition) == partition


def test_rejects_ref2va_disguised_as_fl2va(partition):
    (partition / "model_index.json").write_text(
        json.dumps({"_minimax_h3": {"partition": "ref2va", "tasks": ["ref2va"]}})
    )
    with pytest.raises(ValueError, match="FL2VA support"):
        runtime.validate_model(partition)


def test_rejects_unpinned_revision(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        runtime.validate_model(tmp_path / "main/FL2VA")


def test_verifies_readonly_snapshot_before_offline_launch(partition, monkeypatch):
    calls = []
    for key in (
        "H3_ALLOWED_CLIENTS",
        "H3_TORCH_MEMORY_FRACTION",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(runtime.os.path, "ismount", lambda path: True)
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(runtime.os, "execvp", lambda _, command: calls.append(command))
    monkeypatch.setattr(
        runtime.sys,
        "argv",
        [
            "serve.py",
            "--model",
            str(partition),
            "--host",
            "192.0.2.1",
            "--port",
            "8000",
            "--allowed-clients",
            "127.0.0.1,192.0.2.2",
            "--read-only-model-root",
            str(partition.parent),
        ],
    )
    runtime.main()
    assert calls[0][:3] == ["mount", "-o", "remount,bind,ro"]
    assert "--verify-only" in calls[1]
    assert "FL2VA/" in calls[1]
    assert calls[2][:3] == ["vllm", "serve", str(partition)]
    assert "h3_runtime.AllowedClients" in calls[2]
    assert runtime.os.environ["HF_HUB_OFFLINE"] == "1"
    assert runtime.os.environ["H3_TORCH_MEMORY_FRACTION"] == "0.3"


@pytest.mark.parametrize("indices", [[0], [-1], [0, -1]])
def test_request_preserves_endpoint_image_order(indices):
    spec = importlib.util.spec_from_file_location(
        "check_h3_fl2va", Path(__file__).resolve().parents[1] / "scripts/check_h3_fl2va.py"
    )
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    images = [b"\x89PNG\r\n\x1a\n" + str(i).encode() for i in indices]
    body, boundary, fields = check.request_body(
        images, indices, prompt="Endpoint test", seconds=5, size="832x480", seed=42
    )
    assert body.count(b'name="input_references"') == len(indices)
    assert json.loads(fields["extra_params"])["frame_indices"] == indices
    assert body.endswith(f"--{boundary}--\r\n".encode())
    if len(images) == 2:
        assert body.index(images[0]) < body.index(images[1])
