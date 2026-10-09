import importlib.util
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manage_gpustack", ROOT / "scripts/manage_gpustack.py"
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "deploy/gpustack").mkdir(parents=True)
    for name in ["release.json", "server.example.json"]:
        (tmp_path / "deploy/gpustack" / name).write_text(
            (ROOT / "deploy/gpustack" / name).read_text()
        )
    monkeypatch.setattr(server, "ROOT", tmp_path)
    server.initialize()
    return tmp_path


def test_init_is_private_non_destructive_and_outside_native_all(workspace):
    config = workspace / "deployments/gpustack/server.json"
    password = workspace / ".secrets/gpustack/bootstrap-password"
    original = password.read_text()
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(password.stat().st_mode) == 0o600
    assert not list((workspace / "deployments").glob("*.json"))
    with pytest.raises(ValueError, match="already exists"):
        server.initialize()
    assert password.read_text() == original


def test_server_has_no_gpu_socket_worker_or_production_mount(workspace):
    rendered = server.compose_config(server.load_config(), "test-${literal}")
    assert rendered["name"] == "tiangong-gpustack"
    service = rendered["services"]["server"]
    assert "--disable-worker" in service["command"]
    assert "devices" not in service and "privileged" not in service
    assert service["volumes"] == [
        {"type": "volume", "source": "data", "target": "/var/lib/gpustack"}
    ]
    assert service["ports"][0]["host_ip"] == "127.0.0.1"
    assert "v2.2.3@sha256:" in service["image"]
    assert (
        server.escape_compose(rendered)["services"]["server"]["environment"][
            "GPUSTACK_BOOTSTRAP_PASSWORD"
        ]
        == "test-$${literal}"
    )


@pytest.mark.parametrize("action", ["start", "restart", "stop", "config"])
def test_commands_are_scoped_and_secrets_never_reach_argv_or_console(
    workspace, monkeypatch, capsys, action
):
    calls = []
    secret = (workspace / ".secrets/gpustack/bootstrap-password").read_text().strip()

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr=secret)

    monkeypatch.setattr(server.subprocess, "run", run)
    server.run_compose(action, server.load_config())
    command, kwargs = calls[0]
    assert command[command.index("--project-name") + 1] == "tiangong-gpustack"
    assert secret not in " ".join(command)
    assert command[-1] == "server" or action == "config"
    assert json.loads(kwargs["input"])["services"]["server"]["environment"]
    if action in {"start", "restart"}:
        assert command[command.index("--pull") + 1] == "never"
        assert "--no-build" in command
    if action == "start":
        assert "--no-recreate" in command
    assert secret not in capsys.readouterr().out


def test_failed_docker_output_is_not_disclosed(workspace, monkeypatch):
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="credential", stderr="credential"),
    )
    with pytest.raises(RuntimeError, match="exit 1") as exc:
        server.run_compose("config", server.load_config())
    assert "credential" not in str(exc.value)


@pytest.mark.parametrize(
    "host,port", [("0.0.0.0", 17880), ("::1", 17880), ("127.0.0.1", 80), ("127.0.0.1", True)]
)
def test_invalid_bind_configuration_is_rejected(workspace, host, port):
    (workspace / "deployments/gpustack/server.json").write_text(
        json.dumps({"host": host, "port": port})
    )
    with pytest.raises(ValueError):
        server.load_config()


def test_public_credentials_are_rejected(workspace):
    path = workspace / ".secrets/gpustack/bootstrap-password"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode 600"):
        server.private_read(path)


def test_gateway_check_is_get_only_and_missing_route_fails(workspace, monkeypatch):
    server.private_write(workspace / ".secrets/gpustack/api-key", "fake-test-key\n")
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            return b'{"data": [{"id": "nemotron-embed"}]}'

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return Response()

    monkeypatch.setattr(server.urllib.request, "build_opener", lambda *args: Opener())
    server.check_gateway(server.load_config(), "nemotron-embed")
    with pytest.raises(ValueError, match="missing"):
        server.check_gateway(server.load_config(), "missing")
    assert all(r.get_method() == "GET" and r.data is None for r in requests)
    assert all(r.full_url.endswith("/v1/models") for r in requests)
    assert server.NoRedirect().redirect_request(None, None, 302, "", {}, "http://other") is None
