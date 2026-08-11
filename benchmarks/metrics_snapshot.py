#!/usr/bin/env python3
"""Capture selected vLLM counters, or diff a new snapshot against an old one."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


KEEP = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}

SAMPLE = re.compile(r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+\w.eE]+)$")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def metric_key(name: str, labels_text: str | None) -> str:
    if not labels_text:
        return name
    labels = dict(LABEL.findall(labels_text))
    useful = [f"{key}={labels[key]}" for key in ("position", "finished_reason") if key in labels]
    return name if not useful else f"{name}[{','.join(useful)}]"


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode()
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if not match or match.group("name") not in KEEP:
            continue
        metrics[metric_key(match.group("name"), match.group("labels"))] = float(
            match.group("value")
        )
    return {"time_unix": time.time(), "url": url, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    current = fetch(args.url)
    if args.before:
        before = json.loads(args.before.read_text())
        old = before["metrics"]
        delta = {
            key: value - old.get(key, 0.0)
            for key, value in current["metrics"].items()
            if key.endswith("_total") or "_total[" in key
        }
        drafts = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
        accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
        derived = {
            "elapsed_seconds": current["time_unix"] - before["time_unix"],
            "acceptance": accepted / drafts if drafts else None,
        }
        proposal_steps = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
        for position in range(8):
            key = f"vllm:spec_decode_num_accepted_tokens_per_pos_total[position={position}]"
            if key in delta:
                derived[f"accepted_per_step_pos_{position + 1}"] = (
                    delta[key] / proposal_steps if proposal_steps else None
                )
        result = {"before": str(args.before), "current": current, "delta": delta, "derived": derived}
    else:
        result = current

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
