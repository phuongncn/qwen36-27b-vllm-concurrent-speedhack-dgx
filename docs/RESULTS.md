# ThinkingCap Qwen3.6-27B AutoRound optimization results

Date: 2026-08-11

Hardware: one NVIDIA GB10 / DGX Spark

Model: `ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1`

Runtime: vLLM `0.19.1.dev0`, 131,072-token context, up to 32 active
sequences, FP8 KV cache, Triton attention, MTP, vision, and thinking enabled.

## Final configuration

The production winner on port 8000 uses:

- image `vllm-qwen35-frspec-x16-v1:latest`;
- FR-Spec draft vocabulary: 98,304 rows plus a 512-token tail;
- one-launch x16 INT8 target LM-head kernel;
- MTP depth 3;
- Qwen3.5 prefix caching with Mamba cache mode `align`;
- the original AutoRound W4A16 target weights, full target LM head, and vision
  path.

Without prefix caching, the winning kernel combination improves c1/c4/c8 by
9.2% / 12.6% / 10.9% over the hot V3 baseline. With prefix caching enabled,
c4 reaches 111.58 tok/s, 13.5% above the 98.29 tok/s baseline. A repeated
8k-token system prompt gets a 4.2x warm-TTFT improvement.

## Controlled throughput A/B

Each main result uses temperature 0, 512 output tokens per request, identical
prompts and seeds, three repeats, and a warmup for every concurrency shape.
Prometheus request counters reject runs contaminated by external traffic.
One-time Triton compile passes remain available for audit but are not used as
steady-state numbers.

| Configuration | c1 tok/s | c4 tok/s | c8 tok/s | c4 MTP acceptance | Quality |
|---|---:|---:|---:|---:|---|
| V3 baseline | 29.58 | 98.29 | 154.62 | 66.94% | reference |
| FR-Spec 98k | 32.39 | 106.41 | 165.27 | 66.52% | exact 20/20 |
| FR-Spec 65k | 32.57 | 105.04 | 163.80 | 64.06% | exact 20/20 |
| x16 only | 29.30 | 100.39 | 161.41 | 66.64% | exact 20/20 |
| FR98 + x16, prefix off | 32.32 | **110.68** | **171.45** | 66.86% | exact 20/20 |
| FR98 + x16 + prefix align | **32.62** | **111.58** | see c8 caveat | 69.01% | exact 20/20 |

## FR-Spec subset sweep

The c4/MTP3 LM head performs two full-head target reads and three full-head
draft reads per speculative cycle. At 98,816 selected rows out of 248,320,
FR-Spec cuts traffic from five full-head equivalents to approximately 3.19
without materially reducing draft acceptance.

- **98k won the Pareto comparison:** c4 106.41 tok/s and 66.52% acceptance.
- 65k made the draft head smaller, but c4 acceptance fell to 64.06%; net c4
  throughput was only 105.04 tok/s.
- 49k failed the gate: steady c4 was approximately 99.9 tok/s, aggregate
  acceptance was 57.14%, and position-1 acceptance fell to 75.3%.
- 32k failed the early gate: measured c4 mean was 91.13 tok/s and acceptance
  was 53.93%.

All tested subsets still passed 20/20 deterministic output comparisons because
the unchanged full target head verifies every proposal. Lower acceptance costs
speed; it does not change the verified greedy result.

## x16 target LM head

Standalone full-head microbenchmark for a `248320 x 5120` row-wise INT8 matrix:

| Logit rows | V3 x8 dispatch | x16 kernel | Change |
|---:|---:|---:|---:|
| 1 | 5.606 ms | 5.663 ms | 1.0% slower |
| 4 | 5.564 ms | 5.466 ms | 1.8% faster |
| 8 | 5.880 ms | 6.047 ms | 2.8% slower |
| 16 | 11.607 ms | **9.127 ms** | **21.4% faster** |
| 32 | 23.205 ms | **17.321 ms** | **25.4% faster** |

The sampled-logit correctness check passed with max absolute error `0.013584`,
the same as V3. x16 alone adds only 2.14% at c4, but combining it with FR98
raises c4 from 106.41 to 110.68 tok/s, so the combined image is worthwhile.

## MTP depth sweep

| Depth | c4 tok/s | Result |
|---:|---:|---|
| 2 | 101.20 | insufficient proposal yield; rejected |
| **3** | **110.68** prefix off / **111.58** prefix on | winner |
| 4 | approximately 104.7 steady | position 4 at approximately 40% did not pay for its cost |

MTP1 was early-stopped after MTP2 lost by a wide margin. MTP5 was early-stopped
after MTP4 lost and marginal acceptance continued to fall with depth. Every
depth that was run passed the 20/20 deterministic output gate.

## Repeated-prefix A/B

The dedicated benchmark used an 8,066-token system prompt, six sequential
requests with different user suffixes, and 64 output tokens per request.

| Metric | Prefix off | Prefix align |
|---|---:|---:|
| Warm TTFT | 8.97 s | **2.12 s** |
| Warm total request time | 10.98 s | **4.04 s** |
| Cached-token hits | 0 | 32,000 |

Qwen3.5 in this vLLM build rejects Mamba cache mode `all`. The launcher uses
the supported `--mamba-cache-mode align` path. vLLM automatically increases the
attention block size from 128 to 1,600 tokens and reports a 492,800-token GPU
KV capacity.

### c8 caveat

Prefix-align c4 was highly stable at 111.52 / 111.68 / 111.55 tok/s. c8 reached
175.25 tok/s, and the five-run stability median was 174.49 tok/s, but two
outliers fell to 105.44 and 141.72 tok/s. vLLM explicitly labels Mamba `align`
prefix caching experimental.

The default keeps prefix caching enabled because the target workload repeatedly
sends 5k-10k-token system prefixes and benefits much more from the 4.2x warm
TTFT gain. A pure-decode c8 service can use `PREFIX_CACHING=false` for a steady
171.45 tok/s result.

## Quality gates

- The 20-prompt deterministic suite compares reasoning, answer content,
  completion-token count, and finish reason. The winner passed 20/20 exactly.
- A vision smoke test correctly read `VeloGB10` and `Fast native compute for
  GB10 systems`, then accurately described the V/arrow symbol.
- No measured span had NaNs, OOMs, failed requests, or external traffic.
- Sampling-distribution equivalence was not statistically certified. Users
  should run workload-specific sampling evaluations before high-stakes use.

## Reproduction

Run the default throughput profile:

```bash
./run.sh agentbulk
```

Run the interactive 256k/c1 profile:

```bash
./run.sh agentcode
```

Disable experimental prefix caching for steadier pure-decode c8:

```bash
PREFIX_CACHING=false ./run.sh agentbulk
```

Rollback to V3:

```bash
VLLM_IMAGE=vllm-qwen35-v3:latest \
DGX_SPARK_DRAFT_VOCAB=0 \
PREFIX_CACHING=false \
./run.sh agentbulk
```

Benchmark drivers that regenerate JSON artifacts are in `benchmarks/`. Raw
response text, container inspection data, and local machine paths are not
published, which avoids leaking prompts or host metadata.
