"""Verify the backend's gateway-only boundary independently from GPU inference."""

import asyncio
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "h3_runtime", Path(__file__).resolve().parents[1] / "deploy/h3/h3_runtime.py"
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_untrusted_peer_cannot_spoof_gateway_with_forwarded_header(monkeypatch):
    monkeypatch.setenv("H3_ALLOWED_CLIENTS", "127.0.0.1,192.0.2.2")
    calls = []

    async def app(scope, receive, send):
        calls.append(scope)

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "client": ("192.0.2.99", 1234),
        "headers": [(b"x-forwarded-for", b"192.0.2.2")],
    }
    asyncio.run(runtime.AllowedClients(app)(scope, None, send))
    assert not calls
    assert messages[0]["status"] == 403
    scope["client"] = ("192.0.2.2", 1234)
    asyncio.run(runtime.AllowedClients(app)(scope, None, send))
    assert len(calls) == 1
