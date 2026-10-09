import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "laya_server", Path(__file__).resolve().parents[1] / "deploy/laya/server.py"
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def payload():
    return {
        "state": "Please refund the duplicate charge.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"x": {"type": "unknown", "instructions": "test"}},
        {"x": {"type": "choice", "instructions": "test", "criteria": ["duplicate", "duplicate"]}},
        {"x": {"type": "noul", "instructions": ""}},
    ],
)
def test_rejects_invalid_questions_before_gpu_use(questions):
    p = payload()
    p["questions"] = questions
    with pytest.raises(ValueError):
        server.validate_payload(p, "model", 32)


def test_bounds_batch_and_input_size():
    p = payload()
    p["questions"] = {str(i): p["questions"]["refund"] for i in range(33)}
    with pytest.raises(ValueError, match="1 to 32"):
        server.validate_payload(p, "model", 32)
    p = payload()
    p["state"] = "x" * (128 * 1024 + 1)
    with pytest.raises(ValueError, match="128 KiB"):
        server.validate_payload(p, "model", 32)


def test_unknown_model_is_distinct_from_invalid_input():
    p = payload()
    p["model"] = "unknown"
    with pytest.raises(LookupError):
        server.validate_payload(p, "model", 32)


def test_sdk_tokenizer_fix_cannot_modify_verified_original(tmp_path):
    source = tmp_path / "weights"
    (source / "tokenizer").mkdir(parents=True)
    (source / "tokenizer/tokenizer_config.json").write_text('{"extra_special_tokens": []}')
    (source / "model.safetensors").write_bytes(b"weights")
    target = tmp_path / "overlay"
    server.prepare_model(source, target)
    (target / "tokenizer/tokenizer_config.json").write_text('{"extra_special_tokens": {}}')
    assert (
        source / "tokenizer/tokenizer_config.json"
    ).read_text() == '{"extra_special_tokens": []}'
    assert (target / "model.safetensors").is_symlink()
    assert (target / "model.safetensors").read_bytes() == b"weights"
