# ThinkingCap AutoRound V3: real-world c4-c8 telemetry analysis

Date: 2026-08-11

Source: 30 minutes of compact vLLM telemetry from the live
`vllm-thinkingcap-8000` container. This analysis generated no benchmark
requests.

## Summary

The production workload was dominated by c6 and included c4-c7, but no sampled
window reached c8. When c4 was actively decoding, productive throughput averaged
85.9 tok/s and peaked at 93.4 tok/s, close to the earlier 90.1 tok/s synthetic
baseline. The original runtime therefore did not suffer a hidden production
decode collapse.

Wall-clock output was lower because vLLM's `Running` count includes requests in
prefill. The workload continuously ingested 5k-10k-token prompts while other
requests decoded. A displayed concurrency of six or seven did not mean that
every lane generated a token in every 10-second window.

## Observation window

- UTC 07:34:13-08:04:03, approximately 14:34-15:04 local time.
- 180 telemetry windows at approximately 10 seconds each.
- 136 windows had `Running=4..7`, or 22.7 minutes of the sample.
- No `Running=8` window; the report does not infer real c8 from c7.
- The waiting queue remained zero throughout the window.

## Throughput by displayed concurrency

`Wall mean` includes prefill and 10-second windows with little or no generation.
`Productive mean` includes windows with at least 20 output tok/s and is the
closest proxy for sustained decode. It still cannot reveal exactly how many
requests were in decode phase.

| Running | Windows | Wall mean | Median | P75 | Productive mean | Peak | Output/running proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 13 / 2.2 min | 51.5 | 72.0 | 85.3 | **85.9** | 93.4 | 12.9 |
| 5 | 17 / 2.8 min | 40.4 | 45.9 | 64.1 | 55.3 | 70.1 | 8.1 |
| 6 | 78 / 13.0 min | 32.3 | 27.7 | 51.1 | 48.5 | **107.3** | 5.4 |
| 7 | 28 / 4.7 min | 22.4 | 12.8 | 27.8 | 43.4 | 106.0 | 3.2 |
| c4-c7 combined | 136 / 22.7 min | **33.1** | 25.9 | 53.9 | **52.0** | 107.3 | 5.6 |

The last column is not true per-request decode speed. It divides wall-clock
output by all running requests even though some lanes were prefilling.

## MTP behavior

| Running | Draft acceptance | Position 1 | Position 2 | Position 3 | Implied output/step |
|---:|---:|---:|---:|---:|---:|
| 4 | 68% | 84% | 68% | 51% | 3.03 |
| 5 | 68% | 84% | 67% | 52% | 3.03 |
| 6 | 65% | 83% | 64% | 48% | 2.95 |
| 7 | 66% | 83% | 66% | 49% | 2.98 |
| c4-c7 combined | **66%** | **83%** | **65%** | **49%** | **2.98** |

The c4-c7 windows contained 45,510 draft tokens and 29,962 accepted tokens.
Position 3 remained close to 50%, which supported keeping MTP3 as the baseline
before the controlled depth sweep.

## Workload shape

Across c4-c7 windows:

- mean prompt throughput was 648.7 tok/s;
- mean wall-clock generation throughput was 33.1 tok/s;
- mean KV usage was only 8.2%; capacity was not the bottleneck;
- prefix caching was disabled and the hit rate was zero;
- multimodal cache hit rate was approximately 40-42%.

Lifetime metrics at the analysis point included 218 completed requests,
1,202,452 prompt tokens (5,516 per request), and 113,530 generated tokens (521
per request). Average prefill time was 9.0 seconds and average decode time was
44.3 seconds. Of the 218 prompts, 145 were in the 5k-10k-token histogram bucket.

This evidence motivated two decisions: optimize sustained c4 decode through the
LM head, and enable repeated-prefix caching for the actual long-system-prompt
agent workload.

## Why strict prefill/decode separation was rejected

On one GB10, hard phase separation would likely reduce total throughput. The
30-minute sample can be grouped as follows:

| Window type | Windows | Prompt tok/s | Output tok/s |
|---|---:|---:|---:|
| Prefill-dominant (`prompt>0`, output<10) | 18 | 990 | 4.1 |
| Productive mixed (`prompt>0`, output>=20) | 84 | **891** | **51.0** |
| Decode-dominant (`prompt=0`, output>=20) | 7 | 0 | **56.1** |

Mixed windows produced only about 9% less output than decode-dominant windows
while simultaneously processing 891 prompt tok/s. A simple time-slice estimate
needs `650/990 = 66%` of GPU time for prefill and `33/56 = 59%` for decode, or
125% in total. Strict phases would build a queue at the observed arrival rate.

The vLLM scheduler already prioritizes running requests, allows one partial
prefill, and spends only the remaining token budget on prefill. A hard rule that
blocks prefill while any request decodes can starve new agents for minutes; the
opposite rule stalls inter-token latency whenever a new long prompt arrives.

Prefix caching is a better lever for this workload because it removes repeated
system-prefix prefill instead of merely rescheduling the same work.

## Reproduce the log analysis

```bash
docker logs --timestamps --since 30m vllm-thinkingcap-8000 2>&1 \
  | python3 benchmarks/analyze_vllm_log.py \
      --min-running 4 --max-running 8 --json
```
