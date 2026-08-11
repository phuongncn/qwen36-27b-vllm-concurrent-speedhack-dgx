#!/usr/bin/env python3
"""Measure repeated long-system-prefix latency and vLLM cache counters."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


METRIC = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>[-+\w.eE]+)$")


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def cache_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=10) as response:
        text = response.read().decode()
    result: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC.match(line)
        if not match:
            continue
        name = match.group("name")
        if "prefix_cache" not in name and name not in {
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
        }:
            continue
        # Sum label variants rather than silently retaining the last sample.
        try:
            result[name] = result.get(name, 0.0) + float(match.group("value"))
        except ValueError:
            pass
    return result


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def make_system_prompt(source: Path, target_chars: int) -> str:
    text = source.read_text().strip()
    if not text:
        raise ValueError(f"empty source: {source}")
    pieces = []
    length = 0
    while length < target_chars:
        pieces.append(text)
        length += len(text) + 2
    return "\n\n".join(pieces)[:target_chars]


def stream_completion(
    url: str,
    model: str,
    system_prompt: str,
    request_index: int,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "This suffix is intentionally unique for request "
                    f"{request_index}. Reply with a short Python function that returns "
                    f"the integer {request_index}."
                ),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20260811,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_s = None
    usage = None
    finish_reason = None
    generated_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta_message = choice.get("delta", {})
                generated = (
                    delta_message.get("reasoning")
                    or delta_message.get("reasoning_content")
                    or delta_message.get("content")
                    or ""
                )
                if generated:
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - started
                    generated_parts.append(generated)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    elapsed_s = time.perf_counter() - started
    return {
        "request_index": request_index,
        "ttft_s": first_token_s,
        "elapsed_s": elapsed_s,
        "usage": usage,
        "finish_reason": finish_reason,
        "generated_chars": sum(len(part) for part in generated_parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--system-file",
        type=Path,
        default=Path(__file__).with_name("README.md"),
    )
    parser.add_argument("--system-chars", type=int, default=24_000)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.system_chars < 100 or args.requests < 2 or args.max_tokens < 1:
        parser.error("invalid benchmark sizes")

    base_url = args.base_url.rstrip("/")
    model = get_json(f"{base_url}/v1/models")["data"][0]["id"]
    system_prompt = make_system_prompt(args.system_file, args.system_chars)
    metrics_before = cache_metrics(f"{base_url}/metrics")
    requests = [
        stream_completion(
            f"{base_url}/v1/chat/completions",
            model,
            system_prompt,
            index,
            args.max_tokens,
        )
        for index in range(args.requests)
    ]
    metrics_after = cache_metrics(f"{base_url}/metrics")
    cold = requests[0]
    warm = requests[1:]
    warm_ttft = [request["ttft_s"] for request in warm if request["ttft_s"] is not None]
    warm_elapsed = [request["elapsed_s"] for request in warm]
    summary = {
        "cold_ttft_s": cold["ttft_s"],
        "warm_ttft_mean_s": sum(warm_ttft) / len(warm_ttft) if warm_ttft else None,
        "cold_elapsed_s": cold["elapsed_s"],
        "warm_elapsed_mean_s": sum(warm_elapsed) / len(warm_elapsed),
        "prompt_tokens": cold.get("usage", {}).get("prompt_tokens")
        if cold.get("usage")
        else None,
        "ttft_speedup": cold["ttft_s"] / (sum(warm_ttft) / len(warm_ttft))
        if cold["ttft_s"] and warm_ttft
        else None,
    }
    document = {
        "schema_version": 1,
        "label": args.label,
        "model": model,
        "settings": {
            "system_file": str(args.system_file),
            "system_chars": len(system_prompt),
            "requests": args.requests,
            "max_tokens": args.max_tokens,
        },
        "requests": requests,
        "metrics_delta": delta(metrics_before, metrics_after),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary, "metrics_delta": document["metrics_delta"]}, indent=2))


if __name__ == "__main__":
    main()
