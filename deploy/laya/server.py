"""One CUDA-resident Laya model per container, with bounded asynchronous admission."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import time

LOG = logging.getLogger("laya-api")


def validate_payload(payload, model, max_questions):
    if not isinstance(payload, dict):
        raise ValueError("The request must be a JSON object")
    if payload.get("model", model) != model:
        raise LookupError("Unknown model")
    state = payload.get("state")
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object or array")
    if len(json.dumps(state, ensure_ascii=False).encode()) > 128 * 1024:
        raise ValueError("state exceeds 128 KiB")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= max_questions:
        raise ValueError(f"questions must contain 1 to {max_questions} entries")
    for name, question in questions.items():
        if not name or not isinstance(question, dict):
            raise ValueError("Each question must have a nonempty ID and an object definition")
        kind = question.get("type")
        if kind not in {"choice", "score", "noul"}:
            raise ValueError("Question type must be choice, score or noul")
        if (
            not isinstance(question.get("instructions"), str)
            or not question["instructions"].strip()
        ):
            raise ValueError("Every question needs nonempty instructions")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, (dict, list)) or not 2 <= len(criteria) <= 20:
                raise ValueError("choice criteria must contain 2 to 20 options")
            if not all(isinstance(key, str) and key for key in criteria):
                raise ValueError("choice labels must be nonempty strings")
            if isinstance(criteria, list) and len(set(criteria)) != len(criteria):
                raise ValueError("choice labels must be unique")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 20:
                raise ValueError("score criteria must contain 2 to 20 ordered levels")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object when provided")
    return state, questions


def prepare_model(source, target):
    """Let the SDK normalize its tokenizer config without mutating verified weights."""
    for path in Path(source).rglob("*"):
        dest = Path(target) / path.relative_to(source)
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "tokenizer_config.json":
                shutil.copyfile(path, dest)
            else:
                dest.symlink_to(path.resolve())


class Backend:
    def __init__(self, args):
        os.environ.setdefault("USE_TF", "0")
        import torch
        import laya

        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each replica requires exactly one visible CUDA GPU")
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory(prefix="laya-")
        prepare_model(args.model, self.directory.name)
        self.agent = laya.load(self.directory.name, device="cuda:0")
        if self.agent.device.type != "cuda":
            raise RuntimeError("Laya fell back to CPU while loading")
        if not 256 < args.max_model_len <= self.agent.model.encoder.config.max_position_embeddings:
            raise ValueError("max-model-len must exceed 256 and fit the encoder position limit")
        self.agent.cfg["max_len"] = args.max_model_len
        self.predict("Hello", {"positive": {"type": "noul", "instructions": "Is this a greeting?"}})
        torch.cuda.synchronize()
        self.version = laya.__version__
        LOG.info("Laya %s ready on %s", self.version, torch.cuda.get_device_name(0))

    def predict(self, state, questions):
        if self.agent.device.type != "cuda":
            raise RuntimeError("Replica is unhealthy: CUDA residency was lost")
        result = self.agent.predict(state, questions)
        if self.agent.device.type != "cuda":
            raise RuntimeError("SDK fell back to CPU; CUDA inference failed")
        return result

    def healthy(self):
        return self.agent.device.type == "cuda"

    def memory(self):
        cuda = self.torch.cuda
        return {
            "allocated": cuda.memory_allocated(),
            "reserved": cuda.memory_reserved(),
            "peak_allocated": cuda.max_memory_allocated(),
        }


def create_app(args):
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse

    replica = os.environ.get("MODEL_REPLICA_ID", "worker-0")
    api_key = os.environ.get("VLLM_API_KEY", "")
    stats = {"running": 0, "waiting": 0, "completed": 0, "failed": 0}
    slot = asyncio.Semaphore(1)
    backend = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal backend
        backend = await asyncio.to_thread(Backend, args)
        yield

    app = FastAPI(title="Laya multilingual decisions", lifespan=lifespan)

    def authorize(request):
        if api_key and not secrets.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + api_key
        ):
            raise HTTPException(401, "Invalid or missing API key")

    @app.get("/health")
    async def health():
        if backend is None or not backend.healthy():
            raise HTTPException(503, "Model is not ready on CUDA")
        return {"status": "ok", "replica": replica, "device": "cuda:0"}

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "owned_by": "convaiinnovations",
                    "max_model_len": args.max_model_len,
                }
            ],
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        authorize(request)
        values = {
            **{f"requests_{key}": value for key, value in stats.items()},
            **{f"cuda_{key}_bytes": value for key, value in backend.memory().items()},
        }
        return Response(
            "\n".join(f'laya_{key}{{replica="{replica}"}} {value}' for key, value in values.items())
            + "\n",
            media_type="text/plain",
        )

    @app.post("/v1/decisions")
    async def decisions(request: Request):
        authorize(request)
        raw = await request.body()
        if len(raw) > 1024 * 1024:
            raise HTTPException(413, "Request exceeds 1 MiB")
        try:
            payload = json.loads(raw)
            state, questions = validate_payload(payload, args.served_model_name, args.max_questions)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if not backend.healthy():
            raise HTTPException(503, "Model is not ready on CUDA")
        if stats["running"] + stats["waiting"] >= args.max_pending:
            raise HTTPException(429, "Replica queue is full", headers={"Retry-After": "1"})
        stats["waiting"] += 1
        try:
            await asyncio.wait_for(slot.acquire(), timeout=30)
        except TimeoutError as exc:
            raise HTTPException(503, "Replica queue wait exceeded 30 seconds") from exc
        finally:
            stats["waiting"] -= 1
        stats["running"] += 1
        try:
            start = time.monotonic()
            work = asyncio.create_task(asyncio.to_thread(backend.predict, state, questions))
            try:
                result = await asyncio.shield(work)
            except asyncio.CancelledError:
                # A cancelled HTTP task must not release the GPU slot while its thread runs.
                await work
                raise
            result["model"] = args.served_model_name
            # JSONResponse rejects NaN, so invalid model output fails explicitly.
            response = JSONResponse(
                result,
                headers={
                    "X-Model-Replica": replica,
                    "Server-Timing": f"inference;dur={(time.monotonic() - start) * 1000:.2f}",
                },
            )
            stats["completed"] += 1
            return response
        except ValueError as exc:
            stats["failed"] += 1
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            stats["failed"] += 1
            LOG.exception("Inference failed on %s", replica)
            raise HTTPException(503, "Model inference failed") from exc
        finally:
            stats["running"] -= 1
            slot.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-questions", type=int, default=32)
    parser.add_argument("--max-pending", type=int, default=64)
    args = parser.parse_args()
    if args.max_questions < 1 or args.max_pending < 1:
        parser.error("max-questions and max-pending must be positive")
    import uvicorn

    uvicorn.run(
        create_app(args),
        host=args.host,
        port=args.port,
        workers=1,
        timeout_graceful_shutdown=110,
        log_level="info",
    )


if __name__ == "__main__":
    main()
