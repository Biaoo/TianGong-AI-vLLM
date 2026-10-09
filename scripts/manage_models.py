"""Manage Docker instances using model profiles and private machine configuration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys

from download_model import ROOT, read_env


def valid_name(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Names may contain lowercase letters, digits, - and _ only")
    return name


def legacy_model(env):
    """Preserve the original .env generation deployment and its Compose identity."""
    options = {
        "tensor-parallel-size": int(env.get("TENSOR_PARALLEL_SIZE", "4")),
        "gpu-memory-utilization": float(env.get("GPU_MEMORY_UTILIZATION", "0.85")),
        "max-model-len": int(env.get("MAX_MODEL_LEN", "262144")),
        "max-num-seqs": int(env.get("MAX_NUM_SEQS", "16")),
        "max-num-batched-tokens": int(env.get("MAX_NUM_BATCHED_TOKENS", "8192")),
        "speculative-config": json.loads(env.get("SPECULATIVE_CONFIG") or "null"),
    }
    return {
        "profile": "qwen38-flash-next-nvfp4",
        "project_name": "tiangong-vllm",
        "service_name": "model",
        "options": options,
        "model_dir": env.get("MODEL_DIR", "models/Qwen3.8-Flash-Next-NVFP4"),
        "image": env.get("VLLM_IMAGE", "tiangong-vllm:qwen38-autotune-v1"),
        "served_model_name": env.get("SERVED_MODEL_NAME", "nv-community/Qwen3.8-Flash-Next-NVFP4"),
        "gpus": [env.get(f"GPU_{i}", str(i)) for i in range(4)],
        "gpu_mode": "cdi",
        "host": env.get("VLLM_HOST", "127.0.0.1"),
        "port": int(env.get("VLLM_PORT", "7730")),
        "api_key_env": "VLLM_API_KEY",
        "environment": {
            "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": env.get(
                "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"
            )
        },
    }


def load_instance(name):
    path = ROOT / "deployments" / f"{valid_name(name)}.json"
    if path.exists():
        instance = json.loads(path.read_text())
    elif name == "model" and (ROOT / ".env").exists():
        instance = legacy_model(read_env())
    else:
        raise ValueError(f"Missing deployments/{name}.json; use init <example> <instance>")
    profile = json.loads(
        (ROOT / "deploy/profiles" / f"{valid_name(instance['profile'])}.json").read_text()
    )
    return profile, instance


def options_args(options):
    result = []
    for key, value in options.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key):
            raise ValueError(f"Invalid vLLM option: {key}")
        if key in {"model", "host", "port", "served-model-name", "api-key"}:
            raise ValueError(f"Configure {key} in instance fields, not options")
        if value is None:
            continue
        result.append("--" + key)
        if value is False:
            result[-1] = "--no-" + key
        elif value is True:
            continue
        elif isinstance(value, dict):
            result.append(json.dumps(value, separators=(",", ":")))
        elif isinstance(value, list):
            result.extend(str(item) for item in value)
        else:
            result.append(str(value))
    return result


def replicated_config(name, profile, instance, env):
    """Run one isolated HTTP model server per GPU behind a single gateway."""
    gpus = [str(gpu) for gpu in instance["gpus"]]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("gpus must be a nonempty list of distinct device IDs or UUIDs")
    options = {**profile.get("options", {}), **instance.get("options", {})}
    if any(
        key in options
        for key in ("tensor-parallel-size", "data-parallel-size", "pipeline-parallel-size")
    ):
        raise ValueError("replicated-http uses one complete model per GPU; do not set TP/DP/PP")
    services, volumes = {}, {}
    for index, gpu in enumerate(gpus):
        worker_name = f"worker-{index}"
        single = {**instance, "gpus": [gpu], "service_name": worker_name}
        worker_profile = {**profile, "runtime": "vllm"}
        worker = compose_config(name, worker_profile, single, env)["services"][worker_name]
        worker.pop("ports")
        worker["environment"]["MODEL_REPLICA_ID"] = worker_name
        worker["environment"]["MODEL_HOST_GPU"] = gpu
        worker["volumes"][1]["source"] = f"cache-{index}"
        volumes[f"cache-{index}"] = {}
        services[worker_name] = worker
    upstreams = "\n".join(
        f"        server {worker}:8000 resolve max_fails=1 fail_timeout=5s;" for worker in services
    )
    nginx = """worker_processes auto;
events { worker_connections 1024; }
http {
    resolver 127.0.0.11 valid=10s ipv6=off;
    upstream models {
        zone models 64k;
        least_conn;
UPSTREAMS
        keepalive 32;
    }
    server {
        listen 8000;
        client_max_body_size 1m;
        location / {
            proxy_pass http://models;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_connect_timeout 3s;
            proxy_read_timeout 120s;
            proxy_next_upstream error timeout http_502 http_503 http_504;
        }
    }
}
""".replace("UPSTREAMS", upstreams)
    template = next(iter(services.values()))
    services["gateway"] = {
        "image": profile["gateway_image"],
        "restart": "unless-stopped",
        "ports": [
            {
                "host_ip": instance.get("host", "127.0.0.1"),
                "published": str(instance["port"]),
                "target": 8000,
            }
        ],
        "configs": [{"source": "gateway", "target": "/etc/nginx/nginx.conf"}],
        "depends_on": {worker: {"condition": "service_healthy"} for worker in services},
        "stop_grace_period": "120s",
        "logging": copy.deepcopy(template["logging"]),
        "healthcheck": {
            "test": ["CMD", "curl", "-fsS", "http://127.0.0.1:8000/health"],
            "interval": "15s",
            "timeout": "5s",
            "retries": 3,
        },
    }
    return {
        "name": instance.get("project_name", "tiangong-" + valid_name(name)),
        "services": services,
        "volumes": volumes,
        "configs": {"gateway": {"content": nginx}},
    }


def compose_config(name, profile, instance, env):
    runtime = profile.get("runtime", "vllm")
    if runtime == "replicated-http":
        return replicated_config(name, profile, instance, env)
    if runtime != "vllm":
        raise ValueError(f"Unsupported runtime: {runtime}")
    gpus = [str(gpu) for gpu in instance["gpus"]]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("gpus must be a nonempty list of distinct device IDs or UUIDs")
    options = {**profile.get("options", {}), **instance.get("options", {})}
    topology = [
        int(options.get(key, 1))
        for key in ("tensor-parallel-size", "data-parallel-size", "pipeline-parallel-size")
    ]
    if min(topology) < 1 or topology[0] * topology[1] * topology[2] > len(gpus):
        raise ValueError("TP × DP × PP must be positive and cannot exceed assigned GPUs")
    port = int(instance["port"])
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    environment = {
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        **profile.get("environment", {}),
        **instance.get("environment", {}),
        "VLLM_API_KEY": env.get(instance.get("api_key_env", "VLLM_API_KEY"), ""),
    }
    service = {
        "image": instance.get("image", profile["image"]),
        "restart": "unless-stopped",
        "init": True,
        "shm_size": instance.get("shm_size", "16gb"),
        "stop_grace_period": "120s",
        "ports": [
            {"host_ip": instance.get("host", "127.0.0.1"), "published": str(port), "target": 8000}
        ],
        "environment": {key: str(value) for key, value in environment.items()},
        "volumes": [
            {
                "type": "bind",
                "source": str((ROOT / instance["model_dir"]).resolve()),
                "target": "/model",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
            {"type": "volume", "source": "vllm-cache", "target": "/root/.cache"},
        ],
        "entrypoint": profile.get("entrypoint", ["vllm", "serve"]),
        "command": [
            "/model",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--served-model-name",
            instance.get("served_model_name", profile["model_id"]),
            *options_args(options),
        ],
        "healthcheck": {
            "test": [
                "CMD",
                "python3",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)",
            ],
            "interval": "30s",
            "timeout": "10s",
            "retries": 5,
            "start_period": "30m",
        },
        "logging": {"driver": "json-file", "options": {"max-size": "20m", "max-file": "3"}},
    }
    if "build" in profile:
        service["build"] = {
            **profile["build"],
            "context": str(ROOT / profile["build"].get("context", ".")),
        }
    mode = instance.get("gpu_mode", "cdi")
    if mode == "cdi":
        service["devices"] = [f"nvidia.com/gpu={gpu}" for gpu in gpus]
    elif mode == "nvidia":
        service["runtime"] = "nvidia"
        service["deploy"] = {
            "resources": {
                "reservations": {
                    "devices": [{"driver": "nvidia", "device_ids": gpus, "capabilities": ["gpu"]}]
                }
            }
        }
        service["devices"] = [
            "/dev/nvidia-uvm:/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools",
        ]
    else:
        raise ValueError("gpu_mode must be cdi or nvidia")
    return {
        "name": instance.get("project_name", "tiangong-" + valid_name(name)),
        "services": {instance.get("service_name", "inference"): service},
        "volumes": {"vllm-cache": {}},
    }


def escape_compose(value):
    """Preserve literal dollars in Compose interpolation, including API secrets."""
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, list):
        return [escape_compose(item) for item in value]
    if isinstance(value, dict):
        return {key: escape_compose(item) for key, item in value.items()}
    return value


def run_instance(action, name):
    profile, instance = load_instance(name)
    config = compose_config(name, profile, instance, read_env())
    service = next(iter(config["services"]))
    services = list(config["services"])
    if action in {"download", "verify"}:
        directory = (ROOT / instance["model_dir"]).resolve()
        if action == "download":
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise ValueError("Model directory does not exist; download or configure it first")
        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--entrypoint",
            "python3",
            "--mount",
            f"type=bind,source={ROOT / 'scripts/download_model.py'},target=/opt/download_model.py,readonly",
            "--mount",
            f"type=bind,source={ROOT / profile['manifest']},target=/opt/manifest.json,readonly",
            "--mount",
            f"type=bind,source={directory},target=/models"
            + (",readonly" if action == "verify" else ""),
            config["services"][service]["image"],
            "/opt/download_model.py",
            "--directory",
            "/models",
            "--manifest",
            "/opt/manifest.json",
        ]
        if action == "verify":
            command.append("--verify-only")
        subprocess.run(command, check=True)
        return
    if action == "check":
        if not profile.get("check_script"):
            raise ValueError("Profile has no check_script; add model-specific acceptance tests")
        host = instance.get("host", "127.0.0.1")
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        api_key = config["services"][service]["environment"]["VLLM_API_KEY"]
        subprocess.run(
            [sys.executable, str(ROOT / profile["check_script"])],
            check=True,
            env={
                **os.environ,
                "VLLM_HOST": host,
                "VLLM_PORT": str(instance["port"]),
                "SERVED_MODEL_NAME": instance.get("served_model_name", profile["model_id"]),
                "VLLM_API_KEY": api_key,
                "EMBEDDING_HOST": host,
                "EMBEDDING_PORT": str(instance["port"]),
                "EMBEDDING_SERVED_MODEL_NAME": instance.get(
                    "served_model_name", profile["model_id"]
                ),
                "EMBEDDING_API_KEY": api_key,
                "MODEL_BASE_URL": f"http://{host}:{instance['port']}",
                "MODEL_ID": instance.get("served_model_name", profile["model_id"]),
                "MODEL_API_KEY": api_key,
                "MODEL_REPLICA_COUNT": str(len(instance["gpus"])),
                "MODEL_INSTANCE_NAME": name,
            },
        )
        return
    commands = {
        "start": ["up", "-d", "--build", "--no-recreate", *services],
        "restart": ["up", "-d", "--build", "--force-recreate", "--remove-orphans", *services],
        "start-loaded": ["up", "-d", "--no-build", "--pull", "never", "--no-recreate", *services],
        "restart-loaded": [
            "up",
            "-d",
            "--no-build",
            "--pull",
            "never",
            "--force-recreate",
            "--remove-orphans",
            *services,
        ],
        "stop": ["stop", *services],
        "status": ["ps", "-a", *services],
        "logs": ["logs", "--tail", "100", "-f", *services],
        "config": ["config", "--quiet"],
        "pull": ["build", "--pull", service] if "build" in profile else ["pull", service],
        "build": ["build", service] if "build" in profile else ["pull", service],
    }
    if action == "pull" and profile.get("runtime") == "replicated-http":
        subprocess.run(
            ["docker", "compose", "--project-directory", str(ROOT), "-f", "-", "pull", "gateway"],
            input=json.dumps(escape_compose(config)),
            text=True,
            check=True,
        )
    subprocess.run(
        ["docker", "compose", "--project-directory", str(ROOT), "-f", "-", *commands[action]],
        input=json.dumps(escape_compose(config)),
        text=True,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=[
            "init",
            "list",
            "profiles",
            "start",
            "restart",
            "start-loaded",
            "restart-loaded",
            "stop",
            "status",
            "logs",
            "pull",
            "build",
            "config",
            "check",
            "download",
            "verify",
        ],
    )
    parser.add_argument("target", nargs="?", default="model")
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.action == "profiles":
        for path in sorted((ROOT / "deploy/profiles").glob("*.json")):
            profile = json.loads(path.read_text())
            print(f"{path.stem}\t{profile['model_id']}")
    elif args.action == "list":
        for path in sorted((ROOT / "deployments").glob("*.json")):
            instance = json.loads(path.read_text())
            print(
                f"{path.stem}\t{instance['profile']}\tGPU={instance['gpus']}\tport={instance['port']}"
            )
    elif args.action == "init":
        source = ROOT / "deploy/examples" / f"{valid_name(args.target)}.json"
        content = source.read_text()
        dest = ROOT / "deployments" / f"{valid_name(args.name or args.target)}.json"
        dest.parent.mkdir(exist_ok=True)
        with dest.open("x") as stream:
            os.chmod(dest, 0o600)
            stream.write(content)
        print(
            f"Created {dest.relative_to(ROOT)}; edit GPU, port, paths and options before starting"
        )
    elif args.target == "all":
        if args.action == "logs":
            raise ValueError("Select one instance for streaming logs")
        names = [p.stem for p in sorted((ROOT / "deployments").glob("*.json"))]
        if not names:
            raise ValueError("No explicit instances in deployments/")
        for name in names:
            run_instance(args.action, name)
    else:
        run_instance(args.action, valid_name(args.target))


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        KeyError,
        TypeError,
        FileNotFoundError,
        FileExistsError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
