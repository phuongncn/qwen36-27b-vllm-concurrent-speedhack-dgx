#!/usr/bin/env python3
"""Deterministic HTTP benchmark and exact-output gate for ThinkingCap A/Bs.

The benchmark intentionally uses only the OpenAI-compatible API and vLLM's
Prometheus endpoint.  Every result is JSON so a candidate can be compared after
the server has been restarted.  A run is marked contaminated when vLLM reports
more completed requests than this process submitted during the measured span.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics_snapshot import fetch as fetch_metrics


BENCH_PROMPT = (
    "Write a complete Python REST API using FastAPI with CRUD operations and "
    "JWT authentication. Include models, validation, error handling, tests, "
    "and concise explanations of security decisions."
)

EXACT_PROMPTS = [
    "Return only the result of 1729 + 4096 as decimal digits.",
    "List three invariants of a binary search implementation, one per line.",
    "Write a Python function gcd(a, b) without imports. Return only code.",
    "Explain in two sentences why database transactions need atomicity.",
    "Return valid JSON with keys name='alice', active=true, and score=7.",
    "Translate 'the request completed successfully' into Vietnamese.",
    "Fix this Python expression and return only code: [x*x for x of range(5)]",
    "What is the hexadecimal representation of decimal 255? Return one token.",
    "Give a SQL query selecting id from users where deleted_at is NULL.",
    "Write a Rust match expression mapping Some(x) to x and None to 0.",
    "State De Morgan's two Boolean laws using plain ASCII notation.",
    "Return the sorted sequence: 9, -2, 4, 4, 0. No explanation.",
    "Name the HTTP status code for Too Many Requests. Return digits only.",
    "Write a bash test that succeeds when file.txt exists. Return only command.",
    "In one sentence, distinguish latency from throughput.",
    "Return a JSON array containing null, false, 3, and the string x.",
    "Compute 2^20. Return decimal digits only.",
    "Write one short sentence explaining what a mutex is.",
    "Write a TypeScript type for an object with string id and optional boolean ok.",
    "Complete the sequence and explain briefly: 1, 1, 2, 3, 5, 8, ?",
]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 900) -> Any:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def request_success_total(metrics: dict[str, float]) -> float:
    return sum(
        value
        for key, value in metrics.items()
        if key == "vllm:request_success_total"
        or key.startswith("vllm:request_success_total[")
    )


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    old = before["metrics"]
    return {
        key: value - old.get(key, 0.0)
        for key, value in after["metrics"].items()
        if key.endswith("_total") or "_total[" in key
    }


def mtp_derived(delta: dict[str, float]) -> dict[str, float | None]:
    drafts = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    steps = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
    result: dict[str, float | None] = {
        "draft_tokens": drafts,
        "accepted_tokens": accepted,
        "acceptance": accepted / drafts if drafts else None,
        "draft_steps": steps,
    }
    for position in range(8):
        key = f"vllm:spec_decode_num_accepted_tokens_per_pos_total[position={position}]"
        if key in delta:
            result[f"accepted_per_step_pos_{position + 1}"] = (
                delta[key] / steps if steps else None
            )
    return result


@dataclass
class Completion:
    ok: bool
    elapsed_s: float
    completion_tokens: int
    finish_reason: str | None
    message: dict[str, Any] | None
    error: str | None

    def as_dict(self, include_message: bool = False) -> dict[str, Any]:
        result = {
            "ok": self.ok,
            "elapsed_s": self.elapsed_s,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        if include_message:
            result["message"] = self.message
        return result


class Client:
    def __init__(self, base_url: str, model: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        max_tokens: int,
        seed: int,
        barrier: threading.Barrier | None = None,
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
            "stream": False,
        }
        if barrier is not None:
            barrier.wait()
        started = time.perf_counter()
        try:
            response = http_json(
                f"{self.base_url}/v1/chat/completions", payload, timeout=self.timeout
            )
            elapsed = time.perf_counter() - started
            choice = response["choices"][0]
            return Completion(
                ok=True,
                elapsed_s=elapsed,
                completion_tokens=int(response.get("usage", {}).get("completion_tokens", 0)),
                finish_reason=choice.get("finish_reason"),
                message=choice.get("message"),
                error=None,
            )
        except Exception as exc:  # Preserve failed runs in the result artifact.
            return Completion(
                ok=False,
                elapsed_s=time.perf_counter() - started,
                completion_tokens=0,
                finish_reason=None,
                message=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def normalize_message(message: dict[str, Any] | None) -> dict[str, Any] | None:
    if message is None:
        return None
    # Exclude request-specific metadata while retaining all generated material.
    return {
        key: message.get(key)
        for key in ("role", "content", "reasoning_content", "tool_calls")
        if key in message
    }


def run_parallel(
    client: Client, concurrency: int, max_tokens: int, seed: int
) -> tuple[float, list[Completion]]:
    barrier = threading.Barrier(concurrency)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(client.complete, BENCH_PROMPT, max_tokens, seed, barrier)
            for _ in range(concurrency)
        ]
        completions = [future.result() for future in futures]
    return time.perf_counter() - started, completions


def run_benchmark(
    client: Client,
    metrics_url: str,
    concurrencies: list[int],
    repeats: int,
    max_tokens: int,
    seed: int,
    shape_warmups: int,
    shape_warmup_tokens: int,
) -> list[dict[str, Any]]:
    warm = client.complete("Reply with OK.", 32, seed)
    if not warm.ok:
        raise RuntimeError(f"warmup failed: {warm.error}")
    results: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        # Triton autotunes independently for the flattened MTP row count.  A c1
        # warmup does not compile/fault-in c4 or c8 kernels, so explicitly warm
        # every measured shape before starting its Prometheus span.
        for warmup_index in range(shape_warmups):
            _, warm_completions = run_parallel(
                client,
                concurrency,
                shape_warmup_tokens,
                seed + 10_000 + warmup_index,
            )
            if not all(completion.ok for completion in warm_completions):
                errors = [
                    completion.error
                    for completion in warm_completions
                    if not completion.ok
                ]
                raise RuntimeError(f"c{concurrency} shape warmup failed: {errors}")
        for repeat in range(1, repeats + 1):
            before = fetch_metrics(metrics_url)
            wall_s, completions = run_parallel(
                client, concurrency, max_tokens, seed + repeat
            )
            after = fetch_metrics(metrics_url)
            delta = metric_delta(before, after)
            ok = [completion for completion in completions if completion.ok]
            total_tokens = sum(completion.completion_tokens for completion in ok)
            per_request_tps = [
                completion.completion_tokens / completion.elapsed_s
                for completion in ok
                if completion.elapsed_s > 0
            ]
            completed_delta = request_success_total(after["metrics"]) - request_success_total(
                before["metrics"]
            )
            results.append(
                {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "wall_s": wall_s,
                    "ok": len(ok),
                    "submitted": concurrency,
                    "completion_tokens": total_tokens,
                    "aggregate_tps": total_tokens / wall_s if wall_s else None,
                    "request_tps_mean": statistics.mean(per_request_tps)
                    if per_request_tps
                    else None,
                    "request_tps_min": min(per_request_tps) if per_request_tps else None,
                    "request_tps_max": max(per_request_tps) if per_request_tps else None,
                    "request_latency_p50_s": percentile(
                        [completion.elapsed_s for completion in ok], 0.5
                    ),
                    "request_latency_p95_s": percentile(
                        [completion.elapsed_s for completion in ok], 0.95
                    ),
                    "server_completed_requests_delta": completed_delta,
                    "contaminated": completed_delta != concurrency,
                    "mtp": mtp_derived(delta),
                    "requests": [completion.as_dict() for completion in completions],
                }
            )
    return results


def run_exact(client: Client, max_tokens: int, seed: int) -> list[dict[str, Any]]:
    outputs = []
    for index, prompt in enumerate(EXACT_PROMPTS):
        completion = client.complete(prompt, max_tokens, seed + index)
        outputs.append(
            {
                "index": index,
                "prompt": prompt,
                "ok": completion.ok,
                "completion_tokens": completion.completion_tokens,
                "finish_reason": completion.finish_reason,
                "message": normalize_message(completion.message),
                "error": completion.error,
            }
        )
    return outputs


def compare_exact(
    baseline_path: Path, candidate_outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_document = json.loads(baseline_path.read_text())
    baseline_outputs = baseline_document.get("exact_outputs", baseline_document)
    mismatches = []
    for baseline, candidate in zip(baseline_outputs, candidate_outputs, strict=False):
        fields = ("ok", "completion_tokens", "finish_reason", "message")
        changed = {
            field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
            for field in fields
            if baseline.get(field) != candidate.get(field)
        }
        if changed:
            mismatches.append({"index": candidate["index"], "changed": changed})
    if len(baseline_outputs) != len(candidate_outputs):
        mismatches.append(
            {
                "length": {
                    "baseline": len(baseline_outputs),
                    "candidate": len(candidate_outputs),
                }
            }
        )
    return {
        "baseline": str(baseline_path),
        "passed": not mismatches,
        "compared": min(len(baseline_outputs), len(candidate_outputs)),
        "mismatches": mismatches,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for concurrency in sorted({result["concurrency"] for result in results}):
        clean = [
            result
            for result in results
            if result["concurrency"] == concurrency
            and not result["contaminated"]
            and result["ok"] == result["submitted"]
        ]
        throughputs = [result["aggregate_tps"] for result in clean]
        acceptances = [
            result["mtp"]["acceptance"]
            for result in clean
            if result["mtp"]["acceptance"] is not None
        ]
        summary[f"c{concurrency}"] = {
            "valid_repeats": len(clean),
            "aggregate_tps_mean": statistics.mean(throughputs) if throughputs else None,
            "aggregate_tps_median": statistics.median(throughputs) if throughputs else None,
            "aggregate_tps_min": min(throughputs) if throughputs else None,
            "aggregate_tps_max": max(throughputs) if throughputs else None,
            "mtp_acceptance_mean": statistics.mean(acceptances) if acceptances else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--exact-tokens", type=int, default=128)
    parser.add_argument("--shape-warmups", type=int, default=2)
    parser.add_argument("--shape-warmup-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--exact-baseline", type=Path)
    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.max_tokens < 1
        or args.exact_tokens < 1
        or args.shape_warmups < 0
        or args.shape_warmup_tokens < 1
    ):
        parser.error("repeats and token limits must be positive")
    try:
        concurrencies = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        parser.error("--concurrency must be comma-separated integers")
    if not concurrencies or any(value < 1 for value in concurrencies):
        parser.error("concurrency values must be positive")

    models = http_json(f"{args.base_url.rstrip('/')}/v1/models", timeout=10)
    model = models["data"][0]["id"]
    client = Client(args.base_url, model, args.timeout)
    exact_outputs = None if args.skip_exact else run_exact(
        client, args.exact_tokens, args.seed
    )
    exact_comparison = None
    if args.exact_baseline:
        if exact_outputs is None:
            parser.error("--exact-baseline cannot be combined with --skip-exact")
        exact_comparison = compare_exact(args.exact_baseline, exact_outputs)

    metrics_url = f"{args.base_url.rstrip('/')}/metrics"
    benchmark = run_benchmark(
        client,
        metrics_url,
        concurrencies,
        args.repeats,
        args.max_tokens,
        args.seed,
        args.shape_warmups,
        args.shape_warmup_tokens,
    )
    document = {
        "schema_version": 1,
        "label": args.label,
        "created_unix": time.time(),
        "base_url": args.base_url,
        "model": model,
        "settings": {
            "concurrency": concurrencies,
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "exact_tokens": args.exact_tokens,
            "shape_warmups": args.shape_warmups,
            "shape_warmup_tokens": args.shape_warmup_tokens,
            "seed": args.seed,
        },
        "exact_outputs": exact_outputs,
        "exact_comparison": exact_comparison,
        "benchmark": benchmark,
        "summary": summarize(benchmark),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": document["summary"], "exact": exact_comparison}, indent=2))


if __name__ == "__main__":
    main()
