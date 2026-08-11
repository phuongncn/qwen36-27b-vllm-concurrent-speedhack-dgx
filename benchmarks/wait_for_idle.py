#!/usr/bin/env python3
"""Poll vLLM at a real cadence and exit only after a stable idle window."""

from __future__ import annotations

import argparse
import json
import time

from metrics_snapshot import fetch


ACTIVITY_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def activity_signature(snapshot: dict) -> tuple:
    metrics = snapshot["metrics"]
    success = sum(
        value
        for key, value in metrics.items()
        if key.startswith("vllm:request_success_total[")
    )
    return tuple(metrics.get(key, 0.0) for key in ACTIVITY_COUNTERS) + (success,)


def is_zero(snapshot: dict) -> bool:
    metrics = snapshot["metrics"]
    return (
        metrics.get("vllm:num_requests_running") == 0
        and metrics.get("vllm:num_requests_waiting") == 0
    )


def compact(snapshot: dict) -> dict:
    metrics = snapshot["metrics"]
    return {
        "time_unix": snapshot["time_unix"],
        "running": metrics.get("vllm:num_requests_running"),
        "waiting": metrics.get("vllm:num_requests_waiting"),
        "prompt_tokens": metrics.get("vllm:prompt_tokens_total"),
        "generation_tokens": metrics.get("vllm:generation_tokens_total"),
        "success_total": activity_signature(snapshot)[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--sample-seconds", type=int, default=5)
    args = parser.parse_args()

    while True:
        try:
            first = fetch(args.url)
        except Exception as error:
            print(json.dumps({"event": "metrics_error", "error": str(error)}), flush=True)
            time.sleep(args.interval)
            continue

        print(json.dumps({"event": "check", **compact(first)}), flush=True)
        if not is_zero(first):
            time.sleep(args.interval)
            continue

        signature = activity_signature(first)
        deadline = time.monotonic() + args.idle_seconds
        stable = True
        while time.monotonic() < deadline:
            time.sleep(min(args.sample_seconds, max(0.0, deadline - time.monotonic())))
            current = fetch(args.url)
            if not is_zero(current) or activity_signature(current) != signature:
                print(json.dumps({"event": "idle_cancelled", **compact(current)}), flush=True)
                stable = False
                break
        if stable:
            final = fetch(args.url)
            print(json.dumps({"event": "IDLE_CONFIRMED", **compact(final)}), flush=True)
            return


if __name__ == "__main__":
    main()
