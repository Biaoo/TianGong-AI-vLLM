"""Small deployment controls for the pinned upstream H3 runtime."""

import ipaddress
import os


class MemoryBudget:
    """vLLM-Omni worker extension; applies before model weights are loaded."""

    def init_device(self):
        super().init_device()
        import torch

        fraction = float(os.environ["H3_TORCH_MEMORY_FRACTION"])
        if not 0 < fraction <= 0.5:
            raise ValueError("H3_TORCH_MEMORY_FRACTION must be in (0, 0.5]")
        torch.cuda.set_per_process_memory_fraction(fraction, self.device)


class AllowedClients:
    """Only the explicitly configured gateway, worker and loopback may connect."""

    def __init__(self, app):
        self.app = app
        self.allowed = {
            ipaddress.ip_address(value.strip())
            for value in os.environ["H3_ALLOWED_CLIENTS"].split(",")
        }

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            client = scope.get("client")
            try:
                address = ipaddress.ip_address(client[0]) if client else None
            except ValueError:
                address = None
            if address not in self.allowed:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 403, "headers": []})
                    await send({"type": "http.response.body", "body": b"Forbidden"})
                return
        await self.app(scope, receive, send)
