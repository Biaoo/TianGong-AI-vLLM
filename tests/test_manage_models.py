import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("manage_models", ROOT / "scripts/manage_models.py")
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


def profile():
    return json.loads((ROOT / "deploy/profiles/qwen3-embedding-8b.json").read_text())


def instance():
    return json.loads((ROOT / "deploy/examples/embedding.json").read_text())


@pytest.mark.parametrize("gpu_count", [1, 3, 4])
def test_machine_topology_is_not_hardcoded(gpu_count):
    config = instance()
    config["gpus"] = [str(i) for i in range(gpu_count)]
    config["options"]["data-parallel-size"] = gpu_count
    config["options"]["max-model-len"] = 8192
    service = manager.compose_config("test", profile(), config, {})["services"]["inference"]
    devices = service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]
    assert devices == config["gpus"]
    assert service["runtime"] == "nvidia"
    assert service["command"][service["command"].index("--max-model-len") + 1] == "8192"


def test_rejects_parallelism_that_exceeds_devices():
    config = instance()
    config["options"]["tensor-parallel-size"] = 2
    with pytest.raises(ValueError, match="assigned GPUs"):
        manager.compose_config("test", profile(), config, {})


def test_instances_share_weights_but_isolate_ports_and_caches():
    one = instance()
    two = copy.deepcopy(one)
    two["port"] = 7711
    a = manager.compose_config("one", profile(), one, {})
    b = manager.compose_config("two", profile(), two, {})
    assert a["name"] != b["name"]  # Compose scopes named cache volumes by project.
    sa, sb = a["services"]["inference"], b["services"]["inference"]
    assert sa["ports"] != sb["ports"]
    assert sa["volumes"][0] == sb["volumes"][0]
    assert sa["volumes"][0]["read_only"]
    assert not sa["volumes"][0]["bind"]["create_host_path"]


def test_new_profile_loads_without_manager_changes(tmp_path, monkeypatch):
    (tmp_path / "deploy/profiles").mkdir(parents=True)
    (tmp_path / "deployments").mkdir()
    new = profile()
    new["model_id"] = "example/future-model"
    (tmp_path / "deploy/profiles/future.json").write_text(json.dumps(new))
    config = instance()
    config["profile"] = "future"
    (tmp_path / "deployments/local.json").write_text(json.dumps(config))
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    loaded, local = manager.load_instance("local")
    rendered = manager.compose_config("local", loaded, local, {})
    assert "example/future-model" in rendered["services"]["inference"]["command"]


def test_null_removes_defaults_and_false_disables_boolean_flags():
    assert manager.options_args({"speculative-config": None, "enable-prefix-caching": False}) == [
        "--no-enable-prefix-caching"
    ]
    with pytest.raises(ValueError, match="instance fields"):
        manager.options_args({"api-key": "secret"})


def test_api_key_is_literal_and_not_in_cli():
    secret = "test-${not_an_environment_variable}"
    rendered = manager.compose_config("test", profile(), instance(), {"EMBEDDING_API_KEY": secret})
    service = rendered["services"]["inference"]
    assert service["environment"]["VLLM_API_KEY"] == secret
    assert secret not in service["command"]
    assert manager.escape_compose(rendered)["services"]["inference"]["environment"][
        "VLLM_API_KEY"
    ] == secret.replace("$", "$$")


def test_cdi_uses_only_assigned_devices():
    config = instance()
    config["gpu_mode"] = "cdi"
    service = manager.compose_config("test", profile(), config, {})["services"]["inference"]
    assert service["devices"] == ["nvidia.com/gpu=0", "nvidia.com/gpu=1", "nvidia.com/gpu=2"]
    assert "deploy" not in service


def test_all_only_operates_explicit_local_instances(tmp_path, monkeypatch):
    (tmp_path / "deployments").mkdir()
    (tmp_path / "deployments/embedding.json").write_text("{}")
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(manager, "run_instance", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["manage_models.py", "stop", "all"])
    manager.main()
    assert calls == [("stop", "embedding")]


@pytest.mark.parametrize("action", ["download", "verify"])
def test_model_transfer_uses_container_without_gpu(tmp_path, monkeypatch, action):
    config = instance()
    config["model_dir"] = str(tmp_path)
    monkeypatch.setattr(manager, "load_instance", lambda _: (profile(), config))
    calls = []
    monkeypatch.setattr(manager.subprocess, "run", lambda command, **kwargs: calls.append(command))
    manager.run_instance(action, "test")
    command = calls[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--user" in command
    assert "--gpus" not in command
    assert any("target=/models,readonly" in part for part in command) == (action == "verify")
    assert ("--verify-only" in command) == (action == "verify")


@pytest.mark.parametrize("count", [1, 3])
def test_http_replicas_each_receive_one_gpu_and_private_port(count):
    p = json.loads((ROOT / "deploy/profiles/laya-multilingual.json").read_text())
    local = json.loads((ROOT / "deploy/examples/laya.json").read_text())
    local["gpus"] = [str(i) for i in range(count)]
    rendered = manager.compose_config("laya-test", p, local, {})
    services = rendered["services"]
    assert len(services) == count + 1
    assert len(rendered["volumes"]) == count
    for i in range(count):
        worker = services[f"worker-{i}"]
        assert "ports" not in worker
        devices = worker["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["device_ids"] == [str(i)]
        assert "--tensor-parallel-size" not in worker["command"]
        assert "--data-parallel-size" not in worker["command"]
    assert "runtime" not in services["gateway"]
    assert len(services["gateway"]["depends_on"]) == count
    assert "least_conn;" in rendered["configs"]["gateway"]["content"]


def test_http_replicas_reject_cross_gpu_parallel_options():
    p = json.loads((ROOT / "deploy/profiles/laya-multilingual.json").read_text())
    local = json.loads((ROOT / "deploy/examples/laya.json").read_text())
    local["options"]["tensor-parallel-size"] = 3
    with pytest.raises(ValueError, match="complete model per GPU"):
        manager.compose_config("test", p, local, {})


def test_lifecycle_targets_workers_and_gateway(monkeypatch):
    p = json.loads((ROOT / "deploy/profiles/laya-multilingual.json").read_text())
    local = json.loads((ROOT / "deploy/examples/laya.json").read_text())
    monkeypatch.setattr(manager, "load_instance", lambda _: (p, local))
    calls = []
    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    manager.run_instance("stop", "test")
    assert calls[0][-5:] == ["stop", "worker-0", "worker-1", "worker-2", "gateway"]


def test_scaling_down_removes_only_orphans_in_the_instance_project(monkeypatch):
    p = json.loads((ROOT / "deploy/profiles/laya-multilingual.json").read_text())
    local = json.loads((ROOT / "deploy/examples/laya-single.json").read_text())
    monkeypatch.setattr(manager, "load_instance", lambda _: (p, local))
    calls = []
    monkeypatch.setattr(
        manager.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs))
    )
    manager.run_instance("restart-loaded", "test")
    command, kwargs = calls[0]
    assert "--remove-orphans" in command
    assert command[-2:] == ["worker-0", "gateway"]
    assert json.loads(kwargs["input"])["name"] == "tiangong-test"
