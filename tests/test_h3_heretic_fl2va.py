"""Keep Heretic native FL2VA/T2VA separate from the FastH3 Ref2VA assembly."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "heretic_fl2va", Path(__file__).resolve().parents[1] / "deploy/h3-heretic-fl2va/serve.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.mark.parametrize(
    "partition,tasks,accepted",
    [
        ("fl2va", ["fl2va", "t2va"], True),
        ("fl2va", ["fl2va"], False),
        ("ref2va", ["ref2va"], False),
    ],
)
def test_only_accepts_native_partition_with_both_tasks(tmp_path, partition, tasks, accepted):
    model = tmp_path / "FL2VA"
    model.mkdir()
    (model / "model_index.json").write_text(
        json.dumps({"_minimax_h3": {"partition": partition, "tasks": tasks}})
    )
    if accepted:
        assert runtime.validate_model(model) == model
    else:
        with pytest.raises(ValueError, match="FL2VA and T2VA"):
            runtime.validate_model(model)
