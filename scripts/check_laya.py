"""Real Laya API acceptance and repeatable HTTP throughput measurements."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request

from download_model import ROOT, read_env

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "payments, invoices, refunds",
            "technical": "software bugs and outages",
            "sales": "pricing and new purchases",
        },
    },
    "refund": {"type": "noul", "instructions": "Does the sender request a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "urgent today"],
    },
}
TEXT = "这笔订单被重复扣款了，请财务部门今天把多收的钱退给我。"


def validate_result(result, questions, model):
    assert result["model"] == model
    assert set(result["answers"]) == set(questions)
    assert result["usage"]["input_tokens"] > 0
    for key, question in questions.items():
        answer = result["answers"][key]
        assert answer["type"] == question["type"]
        assert math.isfinite(answer["confidence"]) and 0 <= answer["confidence"] <= 1
        if question["type"] in {"choice", "score"}:
            probabilities = answer["probabilities"]
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
            assert abs(sum(probabilities.values()) - 1) < 0.002
            if question["type"] == "choice":
                assert set(probabilities) == set(question["criteria"])
                assert answer["choice"] in question["criteria"]
            else:
                assert 0 <= answer["score"] <= len(question["criteria"]) - 1
        else:
            assert 0 <= answer["noul"] <= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=os.environ.get("MODEL_INSTANCE_NAME", "laya"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--requests", type=int, default=90)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--label", default="acceptance")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.rounds) < 1:
        parser.error("benchmark counts must be positive")
    if "MODEL_BASE_URL" in os.environ:
        base = os.environ["MODEL_BASE_URL"]
        model = os.environ["MODEL_ID"]
        key = os.environ.get("MODEL_API_KEY", "")
        replicas = int(os.environ["MODEL_REPLICA_COUNT"])
    else:
        from manage_models import load_instance

        profile, instance = load_instance(args.instance)
        host = instance.get("host", "127.0.0.1")
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        base = f"http://{host}:{instance['port']}"
        model = instance.get("served_model_name", profile["model_id"])
        key = read_env().get(instance.get("api_key_env", "VLLM_API_KEY"), "")
        replicas = len(instance["gpus"])
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    evidence = {
        "instance": args.instance,
        "model": model,
        "replicas": replicas,
        "label": args.label,
        "checks": {},
    }
    output = (
        ROOT
        / "output/validation"
        / ("laya-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    def request(path, payload=None, auth=True):
        start = time.perf_counter()
        req = urllib.request.Request(
            base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers=headers if auth else {},
        )
        with urllib.request.urlopen(req, timeout=150) as response:
            return (
                json.load(response),
                response.headers.get("X-Model-Replica"),
                time.perf_counter() - start,
            )

    def predict(text=TEXT, questions=QUESTIONS):
        result, worker, seconds = request(
            "/v1/decisions", {"model": model, "state": text, "questions": questions}
        )
        validate_result(result, questions, model)
        assert worker
        return {"response": result, "replica": worker, "seconds": seconds}

    try:
        health = request("/health", auth=False)[0]
        assert health["device"] == "cuda:0"
        evidence["checks"]["health"] = health
        assert model in [m["id"] for m in request("/v1/models")[0]["data"]]
        if key:
            try:
                request("/v1/models", auth=False)
            except urllib.error.HTTPError as exc:
                assert exc.code == 401
            else:
                raise AssertionError("API accepted an unauthenticated request")
        evidence["checks"]["authentication"] = {"enabled": bool(key)}
        evidence["checks"]["chinese"] = predict()
        assert (
            evidence["checks"]["chinese"]["response"]["answers"]["department"]["choice"]
            == "billing"
        )
        assert evidence["checks"]["chinese"]["response"]["answers"]["refund"]["noul"] > 0.5
        evidence["checks"]["english"] = predict(
            "I was charged twice for this order. Please refund the duplicate charge today."
        )
        evidence["checks"]["structured_state"] = predict({"body": TEXT, "order": "TEST-42"})
        for name, payload, code in [
            ("empty_questions", {"state": TEXT, "questions": {}}, 400),
            (
                "unknown_type",
                {"state": TEXT, "questions": {"q": {"type": "invalid", "instructions": "test"}}},
                400,
            ),
            ("unknown_model", {"model": "nonexistent", "state": TEXT, "questions": QUESTIONS}, 404),
        ]:
            try:
                request("/v1/decisions", payload)
            except urllib.error.HTTPError as exc:
                assert exc.code == code
                evidence["checks"][name] = code
            else:
                raise AssertionError(f"{name} was accepted")
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: predict(), range(max(18, replicas * 6))))
        distribution = Counter(result["replica"] for result in results)
        assert (
            len(distribution) == replicas
        ), f"Not all GPU replicas served requests: {distribution}"
        evidence["checks"]["concurrency"] = dict(distribution)
        if args.benchmark:
            text = (
                "订单资料：客户购买了年度服务套餐，同一订单收到两次扣款通知。客户要求核对账单并退还重复收取的费用。"
                * 24
            )
            questions = {f"q{i}": QUESTIONS["department"] for i in range(8)}
            for _ in range(replicas * 2):
                predict(text, questions)
            rounds = []
            for _ in range(args.rounds):
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    results = list(
                        pool.map(lambda _: predict(text, questions), range(args.requests))
                    )
                elapsed = time.perf_counter() - start
                latencies = sorted(result["seconds"] for result in results)
                rounds.append(
                    {
                        "seconds": elapsed,
                        "requests_per_second": args.requests / elapsed,
                        "questions_per_second": args.requests * len(questions) / elapsed,
                        "p50_seconds": statistics.median(latencies),
                        "p95_seconds": latencies[math.ceil(len(latencies) * 0.95) - 1],
                        "replicas": dict(Counter(result["replica"] for result in results)),
                        "input_tokens_per_request": results[0]["response"]["usage"]["input_tokens"],
                    }
                )
            evidence["benchmark"] = {
                "requests_per_round": args.requests,
                "concurrency": args.concurrency,
                "questions_per_request": len(questions),
                "rounds": rounds,
                "median_requests_per_second": statistics.median(
                    r["requests_per_second"] for r in rounds
                ),
            }
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
        print(f"Evidence: {output.relative_to(ROOT)}", flush=True)
    print("Passed: " + ", ".join(evidence["checks"]), flush=True)
    if "benchmark" in evidence:
        print(json.dumps(evidence["benchmark"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
