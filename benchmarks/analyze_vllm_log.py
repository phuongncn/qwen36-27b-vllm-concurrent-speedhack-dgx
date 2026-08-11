#!/usr/bin/env python3
"""Summarize real-world vLLM 10-second telemetry windows from docker logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass


ENGINE = re.compile(
    r"^(?P<iso>\S+).*Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<decode>[0-9.]+) tokens/s, "
    r"Running: (?P<running>[0-9]+) reqs, Waiting: (?P<waiting>[0-9]+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)
SPEC = re.compile(
    r"^(?P<iso>\S+).*Mean acceptance length: (?P<length>[0-9.]+), "
    r"Accepted throughput: (?P<accepted_tps>[0-9.]+) tokens/s, "
    r"Drafted throughput: (?P<drafted_tps>[0-9.]+) tokens/s, "
    r"Accepted: (?P<accepted>[0-9]+) tokens, Drafted: (?P<drafted>[0-9]+) tokens, "
    r"Per-position acceptance rate: (?P<positions>[0-9., ]+), "
    r"Avg Draft acceptance rate: (?P<rate>[0-9.]+)%"
)


@dataclass
class Row:
    iso: str
    prompt: float
    decode: float
    running: int
    waiting: int
    kv: float
    accepted: int = 0
    drafted: int = 0
    positions: tuple[float, ...] = ()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def f(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def summarize(rows: list[Row]) -> dict:
    decodes = [row.decode for row in rows]
    active = [row.decode for row in rows if row.decode > 0]
    productive = [row.decode for row in rows if row.decode >= 20]
    no_prefill = [row.decode for row in rows if row.prompt == 0 and row.decode > 0]
    mixed = [row.decode for row in rows if row.prompt > 0]
    total_accepted = sum(row.accepted for row in rows)
    total_drafted = sum(row.drafted for row in rows)
    proposal_steps = total_drafted / 3 if total_drafted else 0
    npos = max((len(row.positions) for row in rows), default=0)
    positions = []
    for position in range(npos):
        numerator = sum(
            row.positions[position] * (row.drafted / 3)
            for row in rows
            if len(row.positions) > position and row.drafted
        )
        positions.append(numerator / proposal_steps if proposal_steps else None)

    return {
        "windows": len(rows),
        "approx_minutes": round(len(rows) / 6, 1),
        "decode_all_mean": f(statistics.fmean(decodes)) if decodes else None,
        "decode_all_median": f(statistics.median(decodes)) if decodes else None,
        "decode_all_p25": f(percentile(decodes, 0.25)),
        "decode_all_p75": f(percentile(decodes, 0.75)),
        "decode_all_p90": f(percentile(decodes, 0.90)),
        "decode_all_max": f(max(decodes)) if decodes else None,
        "zero_decode_share": f(sum(value == 0 for value in decodes) / len(decodes)) if decodes else None,
        "decode_active_mean": f(statistics.fmean(active)) if active else None,
        "decode_active_median": f(statistics.median(active)) if active else None,
        "decode_ge20_mean": f(statistics.fmean(productive)) if productive else None,
        "decode_no_prefill_mean": f(statistics.fmean(no_prefill)) if no_prefill else None,
        "decode_mixed_prefill_mean": f(statistics.fmean(mixed)) if mixed else None,
        "aggregate_per_running_slot": f(
            sum(row.decode for row in rows) / sum(row.running for row in rows)
        )
        if rows and sum(row.running for row in rows)
        else None,
        "prompt_mean": f(statistics.fmean(row.prompt for row in rows)) if rows else None,
        "kv_mean_pct": f(statistics.fmean(row.kv for row in rows)) if rows else None,
        "mtp_drafted_tokens": total_drafted,
        "mtp_accepted_tokens": total_accepted,
        "mtp_acceptance": f(total_accepted / total_drafted) if total_drafted else None,
        "mtp_implied_mean_length": f(1 + total_accepted / proposal_steps) if proposal_steps else None,
        "mtp_acceptance_by_position": [f(value) for value in positions],
    }


def parse(lines) -> list[Row]:
    rows: list[Row] = []
    for line in lines:
        engine = ENGINE.search(line)
        if engine:
            rows.append(
                Row(
                    iso=engine.group("iso"),
                    prompt=float(engine.group("prompt")),
                    decode=float(engine.group("decode")),
                    running=int(engine.group("running")),
                    waiting=int(engine.group("waiting")),
                    kv=float(engine.group("kv")),
                )
            )
            continue
        spec = SPEC.search(line)
        if spec and rows:
            rows[-1].accepted = int(spec.group("accepted"))
            rows[-1].drafted = int(spec.group("drafted"))
            rows[-1].positions = tuple(
                float(value.strip()) for value in spec.group("positions").split(",")
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-running", type=int, default=4)
    parser.add_argument("--max-running", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = parse(sys.stdin)
    chosen = [row for row in rows if args.min_running <= row.running <= args.max_running]
    grouped: dict[int, list[Row]] = defaultdict(list)
    for row in chosen:
        grouped[row.running].append(row)
    result = {
        "source_windows": len(rows),
        "time_start": rows[0].iso if rows else None,
        "time_end": rows[-1].iso if rows else None,
        "filter": {"min_running": args.min_running, "max_running": args.max_running},
        "combined": summarize(chosen),
        "by_running": {str(key): summarize(grouped[key]) for key in sorted(grouped)},
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
