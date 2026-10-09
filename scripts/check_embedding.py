"""Validate the real embedding API, optionally comparing a native-service baseline."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request

from download_model import ROOT, read_env

TEXTS = [
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: What is the capital of China?",
    "The capital of China is Beijing.",
    "Apples are a type of fruit.",
    "生命周期评价用于量化产品的环境影响。",
]


def vectors(response: dict, count: int, dimensions: int) -> list[list[float]]:
    assert len(response["data"]) == count, "Wrong batch size"
    result = []
    for index, item in enumerate(response["data"]):
        assert item["index"] == index, "Wrong batch order"
        vector = item["embedding"]
        if isinstance(vector, str):
            raw = base64.b64decode(vector, validate=True)
            assert len(raw) == dimensions * 4, "Wrong base64 vector length"
            vector = list(struct.unpack(f"<{dimensions}f", raw))
        assert len(vector) == dimensions, "Wrong vector dimensions"
        assert all(math.isfinite(x) for x in vector), "Non-finite embedding"
        assert abs(math.sqrt(sum(x * x for x in vector)) - 1) < 0.02, "Not normalized"
        result.append(vector)
    assert response["usage"]["prompt_tokens"] > 0, "Missing token accounting"
    return result


def cosine(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b), "Incompatible embedding dimensions"
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    env = read_env()
    host = env.get("EMBEDDING_HOST", "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    base = f"http://{host}:{env.get('EMBEDDING_PORT', '7710')}"
    model = env.get("EMBEDDING_SERVED_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")
    key = env.get("EMBEDDING_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    evidence: dict = {"model": model, "checks": {}}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "output/validation" / f"embedding-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def request(path: str, payload: dict | None = None, authenticated: bool = True):
        req = urllib.request.Request(
            base + path,
            headers=headers if authenticated else {"Content-Type": "application/json"},
            data=None if payload is None else json.dumps(payload).encode(),
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            evidence["last_http_error"] = {"status": exc.code, "body": exc.read().decode()}
            raise

    def embed(inputs, **extra):
        return request(
            "/v1/embeddings",
            {"model": model, "input": inputs, "encoding_format": "float", **extra},
        )

    start = time.monotonic()
    try:
        request("/health", authenticated=False)
        evidence["checks"]["version"] = request("/version")
        models = request("/v1/models")
        assert model in [item["id"] for item in models["data"]], "Model identity mismatch"
        evidence["checks"]["models"] = models
        if key:
            try:
                request("/v1/models", authenticated=False)
            except urllib.error.HTTPError as exc:
                assert exc.code == 401
            else:
                raise AssertionError("Unauthenticated request was accepted")
        evidence["checks"]["authentication"] = {"enabled": bool(key)}
        full = embed(TEXTS)
        full_vectors = vectors(full, len(TEXTS), 4096)
        relevant = cosine(full_vectors[0], full_vectors[1])
        unrelated = cosine(full_vectors[0], full_vectors[2])
        assert relevant > unrelated, "Retrieval ordering failed"
        evidence["checks"]["batch"] = full
        evidence["checks"]["retrieval"] = {"relevant": relevant, "unrelated": unrelated}
        single = embed(TEXTS[1])
        assert cosine(vectors(single, 1, 4096)[0], full_vectors[1]) > 0.995
        evidence["checks"]["single"] = single
        reduced = embed(TEXTS, dimensions=1024)
        vectors(reduced, len(TEXTS), 1024)
        evidence["checks"]["dimensions"] = reduced
        encoded = embed(TEXTS[1], encoding_format="base64")
        assert cosine(vectors(encoded, 1, 4096)[0], full_vectors[1]) > 0.995
        evidence["checks"]["base64"] = encoded
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: embed(TEXTS[3]), range(12)))
        for result in results:
            assert cosine(vectors(result, 1, 4096)[0], full_vectors[3]) > 0.995
        evidence["checks"]["concurrent_requests"] = len(results)
        for name, extra, status in [
            ("invalid_dimensions", {"dimensions": 4097}, 400),
            ("unknown_model", {"model": "nonexistent-embedding-model"}, 404),
        ]:
            try:
                embed(TEXTS[1], **extra)
            except urllib.error.HTTPError as exc:
                assert exc.code == status, f"{name}: expected {status}, got {exc.code}"
                evidence["checks"][name] = {"status": exc.code}
            else:
                raise AssertionError(f"{name}: invalid request accepted")
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            response = request("/v1/embeddings", baseline["payload"])
            count = len(baseline["payload"]["input"])
            old = vectors(baseline["response"], count, 4096)
            new = vectors(response, count, 4096)
            similarities = [cosine(a, b) for a, b in zip(old, new)]
            assert min(similarities) > 0.995, f"Baseline divergence: {similarities}"
            evidence["checks"]["baseline_cosine"] = similarities
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        evidence["seconds"] = round(time.monotonic() - start, 3)
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(f"Evidence: {output.relative_to(ROOT)}")
    print("Passed: " + ", ".join(evidence["checks"]))


if __name__ == "__main__":
    main()
