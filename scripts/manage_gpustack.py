"""Opt-in GPUStack server server; does not operate native model instances or workers."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "tiangong-gpustack"


def private_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)


def private_read(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing private file: {path.relative_to(ROOT)}")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(f"Private file requires mode 600: {path.relative_to(ROOT)}")
    value = path.read_text().strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Private credential must be a nonempty single line")
    return value


def initialize():
    config = ROOT / "deployments/gpustack/server.json"
    password = ROOT / ".secrets/gpustack/bootstrap-password"
    if config.exists():
        raise ValueError("Server configuration already exists; it was not overwritten")
    if password.exists():
        private_read(password)
    else:
        private_write(password, secrets.token_urlsafe(32) + "\n")
    template = ROOT / "deploy/gpustack/server.example.json"
    private_write(config, template.read_text())
    print("Initialized deployments/gpustack/server.json")
    print("Admin bootstrap credential: .secrets/gpustack/bootstrap-password")
    print("No containers started; no models or workers registered.")


def load_config():
    config = json.loads((ROOT / "deployments/gpustack/server.json").read_text())
    if set(config) != {"host", "port"}:
        raise ValueError("Server configuration requires only host and port")
    address = ipaddress.ip_address(config["host"])
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise ValueError("Set a concrete IPv4 bind address, preferably loopback for the server")
    if not (address.is_private or address.is_loopback):
        raise ValueError("Use a local or private LAN bind address")
    if type(config["port"]) is not int or not 1024 <= config["port"] <= 65535:
        raise ValueError("Server port must be an integer between 1024 and 65535")
    return config


def compose_config(config, password):
    release = json.loads((ROOT / "deploy/gpustack/release.json").read_text())
    return {
        "name": PROJECT,
        "services": {
            "server": {
                "image": release["image"],
                "restart": "unless-stopped",
                "command": ["--disable-worker", "--disable-update-check"],
                "environment": {"GPUSTACK_BOOTSTRAP_PASSWORD": password},
                "ports": [
                    {"host_ip": config["host"], "published": str(config["port"]), "target": 80}
                ],
                "volumes": [{"type": "volume", "source": "data", "target": "/var/lib/gpustack"}],
                "stop_grace_period": "60s",
                "logging": {"driver": "json-file", "options": {"max-size": "20m", "max-file": "3"}},
            }
        },
        "volumes": {"data": {}},
    }


def escape_compose(value):
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, list):
        return [escape_compose(item) for item in value]
    if isinstance(value, dict):
        return {key: escape_compose(item) for key, item in value.items()}
    return value


def run_compose(action, config):
    password = private_read(ROOT / ".secrets/gpustack/bootstrap-password")
    commands = {
        "config": ["config", "--quiet"],
        "pull": ["pull", "server"],
        "start": ["up", "-d", "--no-recreate", "--no-build", "--pull", "never", "server"],
        "restart": ["up", "-d", "--force-recreate", "--no-build", "--pull", "never", "server"],
        "stop": ["stop", "server"],
        "status": ["ps", "--all", "server"],
    }
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "--project-name",
        PROJECT,
        "-f",
        "-",
        *commands[action],
    ]
    # Neither the password nor rendered configuration appears in argv or console output.
    result = subprocess.run(
        command,
        input=json.dumps(escape_compose(compose_config(config, password))),
        text=True,
        capture_output=True,
        timeout=1800 if action == "pull" else 180,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"GPUStack server {action} failed (exit {result.returncode}); inspect Docker locally"
        )
    if action == "status":
        print(result.stdout, end="")
    else:
        print(f"GPUStack server {action}: OK")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check_gateway(config, model=None):
    key = private_read(ROOT / ".secrets/gpustack/api-key")
    base = f"http://{config['host']}:{config['port']}"
    request = urllib.request.Request(
        base + "/v1/models", headers={"Authorization": "Bearer " + key}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            if response.status != 200:
                raise ValueError("Gateway model-list request did not return HTTP 200")
            body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise ValueError("Gateway model-list response exceeds the read limit")
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Gateway model-list request failed (HTTP {exc.code})") from None
    except urllib.error.URLError:
        raise ValueError("Gateway model-list connection failed") from None
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("Gateway returned an invalid model list")
    identifiers = [item.get("id") for item in data["data"] if isinstance(item, dict)]
    if model is not None and model not in identifiers:
        raise ValueError("Expected model route is missing or not authorized for this API key")
    print(f"Gateway model-list check: OK ({len(identifiers)} models)")
    print("This does not validate inference, replica distribution or failover.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["init", "config", "pull", "start", "restart", "stop", "status", "check"]
    )
    parser.add_argument("--model", help="Required route name for the read-only check action")
    args = parser.parse_args()
    if args.model and args.action != "check":
        parser.error("--model is only valid with check")
    if args.action == "init":
        initialize()
        return
    config = load_config()
    if args.action == "check":
        check_gateway(config, args.model)
    else:
        run_compose(args.action, config)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        # JSON decoding errors may contain snippets; subprocess errors may contain command details.
        if isinstance(exc, (json.JSONDecodeError, OSError, subprocess.TimeoutExpired)):
            print(f"GPUStack server failed: {type(exc).__name__}", file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        sys.exit(1)
