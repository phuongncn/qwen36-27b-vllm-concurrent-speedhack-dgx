#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4.
"""Correctness smoke test and decode-shape benchmark for the LM-head patch.

Run inside a patched vLLM image. The default performance shape matches the
Qwen3.6-27B LM head on GB10. The script first validates sampled logits on a
smaller matrix, then benchmarks flattened MTP logit-row batches.
"""

from __future__ import annotations

import argparse
import time

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor


class DummyLMHead:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def synchronize() -> None:
    torch.cuda.synchronize()


def timed_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    synchronize()
    return start.elapsed_time(end) / repeats


def initialize_and_check(device: torch.device) -> tuple[LogitsProcessor, DummyLMHead]:
    # The production patch activates only for vocabularies above 100k.
    vocab, hidden, batch = 100_096, 128, 8
    torch.manual_seed(7)
    weight = torch.randn((vocab, hidden), dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn((batch, hidden), dtype=torch.bfloat16, device=device)
    lm_head = DummyLMHead(weight)
    processor = LogitsProcessor(vocab_size=vocab)

    got = processor._get_logits(hidden_states, lm_head, None)
    synchronize()
    if got is None:
        raise RuntimeError("patched LM head returned None")

    # Compare a deterministic sample against the exact dequantized INT8 matrix.
    sample = torch.arange(0, vocab, max(1, vocab // 257), device=device)[:257]
    sampled_weight = (
        lm_head._ww_int8[sample].float()
        * lm_head._ww_scales[sample, None].float()
    )
    reference = hidden_states.to(torch.float16).float() @ sampled_weight.T
    observed = got[:, sample].float()
    max_error = (observed - reference).abs().max().item()
    mean_error = (observed - reference).abs().mean().item()
    tolerance = 0.15
    if max_error > tolerance:
        raise RuntimeError(
            f"correctness check failed: max_error={max_error:.6f} > {tolerance}"
        )
    print(
        f"correctness=ok batch={batch} sampled_logits={sample.numel()} "
        f"max_error={max_error:.6f} mean_error={mean_error:.6f}"
    )
    return processor, lm_head


def benchmark(
    processor: LogitsProcessor,
    lm_head: DummyLMHead,
    device: torch.device,
    vocab: int,
    hidden: int,
    batches: list[int],
    warmup: int,
    repeats: int,
) -> None:
    # No random fill is needed for timing. The warmup faults all unified-memory
    # pages in before measurement, matching steady-state model serving.
    lm_head._ww_int8 = torch.empty((vocab, hidden), dtype=torch.int8, device=device)
    lm_head._ww_scales = torch.ones((vocab,), dtype=torch.float16, device=device)
    weight_gb = lm_head._ww_int8.numel() / 1e9
    print(
        f"shape=vocab:{vocab},hidden:{hidden} int8_weight={weight_gb:.3f}GB "
        f"warmup={warmup} repeats={repeats}"
    )

    for batch in batches:
        hidden_states = torch.randn(
            (batch, hidden), dtype=torch.bfloat16, device=device
        )
        last: torch.Tensor | None = None

        def run() -> None:
            nonlocal last
            last = processor._get_logits(hidden_states, lm_head, None)

        before = time.monotonic()
        elapsed_ms = timed_ms(run, warmup=warmup, repeats=repeats)
        compile_and_bench_seconds = time.monotonic() - before
        assert last is not None
        rows_per_second = batch * 1000.0 / elapsed_ms
        print(
            f"batch={batch:>2} ms={elapsed_ms:>9.3f} "
            f"rows/s={rows_per_second:>9.2f} "
            f"first_shape_seconds={compile_and_bench_seconds:.2f}"
        )
        del hidden_states, last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=248_320)
    parser.add_argument("--hidden-size", type=int, default=5_120)
    parser.add_argument("--batches", default="1,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    print(
        f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )
    # LogitsProcessor is a vLLM CustomOp and normally receives this context
    # from the engine during model construction. Supply the same minimal
    # context when running this standalone regression benchmark.
    with set_current_vllm_config(VllmConfig()):
        processor, lm_head = initialize_and_check(device)
        batches = [int(value) for value in args.batches.split(",") if value]
        benchmark(
            processor,
            lm_head,
            device,
            args.vocab_size,
            args.hidden_size,
            batches,
            args.warmup,
            args.repeats,
        )


if __name__ == "__main__":
    main()
