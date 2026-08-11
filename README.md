# Five LM-head reads per step: making Qwen3.6-27B scale on one DGX Spark

[![Hardware](https://img.shields.io/badge/NVIDIA-DGX%20Spark-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![vLLM](https://img.shields.io/badge/vLLM-0.19.1-red)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

Four requests were decoding. MTP3 was proposing three tokens per step. vLLM
was batching. Yet one verification cycle could still stream the same 1.27 GB
LM-head weights through memory **five times**: twice for 16 target rows because
the existing kernel stopped at eight, then once for each of three draft passes.

The concurrency plateau was not just a scheduling problem. It was a
memory-traffic bill.

This repository contains the reproducible patches, launch profiles, and
benchmarks that raised controlled c4 decode throughput for
[ThinkingCap Qwen3.6-27B AutoRound INT4](https://huggingface.co/josefprusa/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1)
from **98.29 to 111.58 tok/s (+13.5%)** on one NVIDIA GB10 / DGX Spark. The
final configuration also preserved **20/20 deterministic outputs**, kept vision
working, and reduced warm TTFT for a repeated 8k-token system prompt from
**8.97 s to 2.12 s**.

## The target was production c4-c8, not a record at c1

The surface goal was simple: make a 27B dense model scale under four to eight
concurrent agent requests.

The harder requirement was to do it without changing the target model,
retraining the checkpoint, dropping vision, or sacrificing the 131k context
profile. The service also sees repeated 5k-10k-token system prompts, so a pure
decode win that made prefill worse would not be a production win.

That left a narrow path: reduce work around speculative decoding while keeping
the full 248,320-row target head as the final authority.

## Three walls blocked the obvious fixes

### 1. The target kernel stopped batching at eight rows

At c4 with MTP3, target verification naturally produces 16 rows. The V3 kernel
handled them as two full LM-head reads. A wider kernel looked promising in
isolation, but x16 alone moved c4 only from 98.29 to 100.39 tok/s.

### 2. A smaller draft head could destroy its own gain

MTP draft passes do not need every target-vocabulary row to propose useful
tokens. Reducing the draft head saves bandwidth, but over-pruning lowers
acceptance and forces the expensive target path to recover more often. The 65k
variant was smaller than 98k, yet slower at c4 because acceptance fell from
66.52% to 64.06%.

### 3. The real workload mixed prefill with decode

Prefix reuse matters when many requests begin with the same long system prompt.
But Qwen3.5's hybrid GDN/Mamba cache path is experimental: `align` mode greatly
improved repeated-prefix TTFT while introducing occasional c8 batch jitter.
Optimizing only a clean decode benchmark would hide that tradeoff.

The darkest point was also the useful clue: every isolated optimization had a
caveat. The speedup had to come from a combination whose parts protected one
another.

## The combination that won

The final path was guided by bytes moved per accepted token, not kernel speed in
isolation:

1. **INT8 LM-head V3** batches up to eight logit rows per full weight read.
2. **FR-Spec 98k** gives MTP draft passes a compact 98,304 + 512-row head while
   target verification retains the full 248,320-row head.
3. **x16 target kernel** handles c4/MTP3's 16 verification rows in one launch.
4. **MTP3** remains the measured optimum after sweeping depths two and four.
5. **Qwen3.5 prefix-cache `align` mode** reuses repeated long system prompts.

FR-Spec changes the proposals, not the final greedy target. Every proposal is
still verified by the unchanged full target head.

> **Move fewer bytes, preserve the verifier, and concurrency starts scaling again.**

## Measured outcome

All main rows use temperature 0, 512 output tokens per request, three repeats,
the same prompts and seeds, per-concurrency warmup, and Prometheus contamination
checks.

| Configuration | c1 tok/s | c4 tok/s | c8 tok/s | c4 MTP accept | Greedy gate |
|---|---:|---:|---:|---:|---|
| V3 baseline | 29.58 | 98.29 | 154.62 | 66.94% | reference |
| FR-Spec 98k | 32.39 | 106.41 | 165.27 | 66.52% | 20/20 exact |
| FR-Spec 65k | 32.57 | 105.04 | 163.80 | 64.06% | 20/20 exact |
| x16 only | 29.30 | 100.39 | 161.41 | 66.64% | 20/20 exact |
| FR98 + x16, prefix off | 32.32 | 110.68 | 171.45 | 66.86% | 20/20 exact |
| **FR98 + x16 + prefix align** | **32.62** | **111.58** | see caveat | 69.01% | **20/20 exact** |

The standalone LM-head microbenchmark (`248320 x 5120`, row-wise INT8) explains
why x16 becomes valuable only when the workload reaches the matching shape:

| Rows | V3 | x16 | Change |
|---:|---:|---:|---:|
| 1 | 5.606 ms | 5.663 ms | -1.0% |
| 4 | 5.564 ms | 5.466 ms | +1.8% |
| 8 | 5.880 ms | 6.047 ms | -2.8% |
| 16 | 11.607 ms | **9.127 ms** | **+21.4%** |
| 32 | 23.205 ms | **17.321 ms** | **+25.4%** |

Full methodology, subset and depth sweeps, prefix-cache data, quality gates,
and the c8 caveat are in [docs/RESULTS.md](docs/RESULTS.md). Production
telemetry analysis is in
[docs/REALWORLD-ANALYSIS.md](docs/REALWORLD-ANALYSIS.md).

## Your turn: reproduce it on a DGX Spark

If your own service stalls around c4, start by measuring LM-head row shapes,
full-weight reads, and accepted tokens per draft. A faster microkernel is useful
only when it removes a shape that the live scheduler actually produces.

### Prerequisites

- NVIDIA DGX Spark / GB10 with Docker and NVIDIA Container Toolkit.
- An ARM64 vLLM 0.19.1-era image compiled for SM121 and supporting Qwen3.5
  AutoRound. We tested `vllm-sm121:latest`, built from
  [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).
- The AutoRound checkpoint above, including its MTP tensors.

The patches are source-layout-specific. Do not assume compatibility with a
different vLLM release without rerunning the microbenchmark, exact-output gate,
and server tests.

### Quick start

Clone the repository and download the model:

```bash
git clone https://github.com/phuongncn/qwen36-27b-vllm-concurrent-speedhack-dgx.git
cd qwen36-27b-vllm-concurrent-speedhack-dgx

hf download josefprusa/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1 \
  --local-dir "$HOME/models/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1"
```

Build V3 and the final winner on top of an existing SM121 image:

```bash
BASE_IMAGE=vllm-sm121:latest ./build.sh
```

If you already have the tested V3 image, build only the two winner deltas:

```bash
SKIP_V3=1 V3_IMAGE=vllm-qwen35-v3:latest ./build.sh
```

Launch the 131k / up-to-c32 throughput profile:

```bash
MODEL_DIR="$HOME/models/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1" \
  ./run.sh agentbulk
```

Launch the 256k / c1 interactive profile:

```bash
MODEL_DIR="$HOME/models/ThinkingCap-Qwen3.6-27B-int4-AutoRound-v1" \
  ./run.sh agentcode
```

The default winner is FR98 + x16 + MTP3 + prefix-cache align. Useful
overrides:

```bash
PREFIX_CACHING=false ./run.sh agentbulk     # Steadier pure-decode c8
DGX_SPARK_DRAFT_VOCAB=65536 ./run.sh agentbulk
MTP_TOKENS=2 ./run.sh agentbulk
VLLM_IMAGE=vllm-qwen35-v3:latest \
  DGX_SPARK_DRAFT_VOCAB=0 PREFIX_CACHING=false ./run.sh agentbulk
```

### Benchmark it

Run the controlled HTTP A/B benchmark with exact outputs and MTP counters:

```bash
python3 benchmarks/ab_benchmark.py \
  --label winner \
  --output results/winner.json \
  --concurrency 1,4,8 \
  --repeats 3 \
  --max-tokens 512
```

Measure repeated-system-prefix behavior:

```bash
python3 benchmarks/prefix_cache_benchmark.py \
  --label prefix-align \
  --output results/prefix-align.json \
  --system-file README.md \
  --system-chars 24000
```

Run the standalone LM-head correctness and performance test inside the image:

```bash
docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v "$PWD/benchmarks/bench_int8_lmhead.py:/bench.py:ro" \
  vllm-qwen35-frspec-x16-v1:latest /bench.py \
  --warmup 3 --repeats 10
```

## Read the c8 result with the prefix-cache caveat

vLLM marks Qwen3.5 Mamba `align` prefix caching experimental. It changes the
attention block from 128 to 1,600 tokens and exposed occasional c8 batch jitter
in our run: stability-pass median 174.49 tok/s and peak 175.25, but outliers at
105.44 and 141.72. c4 was stable at 111.52-111.68.

Prefix caching remains enabled by default because the target workload repeatedly
sends 5k-10k-token system prefixes and gets a 4.2x warm-TTFT gain. Disable it
for a pure-decode c8 service.

## Scope and quality

- Deterministic winner output matched V3 on 20/20 prompts, including reasoning,
  content, completion count, and finish reason.
- Vision smoke testing read the VeloGB10 logo and text correctly.
- Sampling-distribution equivalence was not statistically certified. Target
  verification is retained, but downstream users should run their own sampling
  evaluations before high-stakes deployment.
- No model weights, Hugging Face cache, private prompts, or generated responses
  are stored in this repository.

## Acknowledgements

The V3 batching work builds on the public DGX Spark Qwen3.5 optimization work by
[albond](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4).
FR-Spec design and the byte-traffic-first methodology were informed by
[VeloGB10](https://github.com/sf-stav/veloGB10). vLLM remains licensed by its
respective contributors; this repository is Apache-2.0.
