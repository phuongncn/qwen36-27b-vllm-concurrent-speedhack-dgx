#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4 and modified for
# concurrency-safe x8 batching plus SM121 Triton autotuning.
"""INT8 LM Head v3 — concurrency-safe batched Triton GEMV.

v1 problem: Python `for b in range(batch)` launched kernel once per token.
  228 launches/128 tokens = 5 per step. Each reads 485MB weights.
  Total: 11.34ms/step at 49% BW.

v2 fixed batch sizes up to four, but fell back to one kernel per row above
four. MTP-3 therefore took the fast path at c=1 (four logit rows) and the
pathological fallback at c>=2 (8/16/32 rows).

v3 handles up to eight rows per launch and chunks larger batches by eight.
Each chunk reads the LM-head weights once, reducing LM-head traffic by up to
8x versus v2's per-row fallback while preserving the tuned c=1 fast path.
"""

import os, sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py"


def apply():
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found"); sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if "DGX_SPARK_INT8_LMHEAD_V3" in content:
        print("SKIP: INT8 LM Head v3 already applied"); return

    if "DGX_SPARK_INT8_LMHEAD" in content:
        print("NOTE: Replacing an older INT8 LM Head patch with v3")

    old = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)'''

    # Also handle v1 patch (replace the entire v1 block)
    old_v1_start = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # DGX_SPARK_INT8_LMHEAD: Fused INT8 GEMV via Triton'''

    old_v2_start = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # DGX_SPARK_INT8_LMHEAD_V2: Batched 2D INT8 GEMV — single kernel launch'''

    new = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # DGX_SPARK_INT8_LMHEAD_V3: concurrency-safe batched INT8 GEMV
        if not hasattr(self, '_int8v3_initialized'):
            self._int8v3_initialized = True
            w = lm_head.weight.data
            if w.dtype in (torch.bfloat16, torch.float16) and w.shape[0] > 100000:
                scales = w.float().abs().amax(dim=1) / 127.0
                scales = scales.clamp(min=1e-12)
                w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
                lm_head._ww_int8 = w_int8
                lm_head._ww_scales = scales.to(torch.float16)
                orig_size = w.numel() * w.element_size()
                lm_head.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)
                import sys as _sys
                print(f"DGX_SPARK_V3: LM Head -> INT8 Batched Triton x8 ({list(w_int8.shape)}, saved {orig_size//1024//1024}MB)", file=_sys.stderr, flush=True)
                import triton
                import triton.language as tl
                # Backport from community PR (DGX Spark forum): autotune
                # picks BLOCK_M/BLOCK_K/num_warps/num_stages for SM121 instead
                # of the hardcoded 128/256/4/2. Arithmetic is byte-identical;
                # only the kernel-launch parameters change. ~6s once-per-process
                # autotune cost (per unique NUM_BATCH × 10 configs), then steady
                # state. Measured on Qwen3.5-122B/Spark: +0.4 to +2.6% across
                # prompt classes (avg +1.2%), no quality change.
                _AUTOTUNE_CONFIGS = [
                    triton.Config({'BLOCK_M': 32,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),  # = v2 baseline
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                ]
                @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['M', 'K', 'NUM_BATCH'])
                @triton.jit
                def _k_v3(out_ptr, w_ptr, x_ptr, s_ptr, M, K,
                          stride_ob, stride_xb, NUM_BATCH: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
                    # 1D grid: each block processes ALL batch elements
                    # Weight tile loaded ONCE, reused for all batch inputs
                    pid_m = tl.program_id(0)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    rmask = rows < M
                    # One accumulator per batch element (unrolled by compiler)
                    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc4 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc5 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc6 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc7 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        co = ks + tl.arange(0, BLOCK_K)
                        km = co < K
                        # Load weight tile ONCE
                        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                                    mask=rmask[:, None] & km[None, :], other=0).to(tl.float32)
                        # Reuse weight tile for each batch element
                        x0 = tl.load(x_ptr + 0 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                        acc0 += tl.sum(w * x0[None, :], axis=1)
                        if NUM_BATCH > 1:
                            x1 = tl.load(x_ptr + 1 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc1 += tl.sum(w * x1[None, :], axis=1)
                        if NUM_BATCH > 2:
                            x2 = tl.load(x_ptr + 2 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc2 += tl.sum(w * x2[None, :], axis=1)
                        if NUM_BATCH > 3:
                            x3 = tl.load(x_ptr + 3 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc3 += tl.sum(w * x3[None, :], axis=1)
                        if NUM_BATCH > 4:
                            x4 = tl.load(x_ptr + 4 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc4 += tl.sum(w * x4[None, :], axis=1)
                        if NUM_BATCH > 5:
                            x5 = tl.load(x_ptr + 5 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc5 += tl.sum(w * x5[None, :], axis=1)
                        if NUM_BATCH > 6:
                            x6 = tl.load(x_ptr + 6 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc6 += tl.sum(w * x6[None, :], axis=1)
                        if NUM_BATCH > 7:
                            x7 = tl.load(x_ptr + 7 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc7 += tl.sum(w * x7[None, :], axis=1)
                    # Scale and store
                    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
                    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 1:
                        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 2:
                        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 3:
                        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 4:
                        tl.store(out_ptr + 4 * stride_ob + rows, (acc4 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 5:
                        tl.store(out_ptr + 5 * stride_ob + rows, (acc5 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 6:
                        tl.store(out_ptr + 6 * stride_ob + rows, (acc6 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 7:
                        tl.store(out_ptr + 7 * stride_ob + rows, (acc7 * s).to(tl.float16), mask=rmask)
                lm_head._ww_kernel_v3 = _k_v3

        if hasattr(lm_head, '_ww_int8'):
            M, K = lm_head._ww_int8.shape
            x = hidden_states.view(-1, K)
            batch = x.shape[0]
            out = torch.empty(batch, M, dtype=torch.float16, device=x.device)
            x_fp16 = x.to(torch.float16)
            # Autotune-aware grid: BLOCK_M is chosen by autotuner per (M,K,NUM_BATCH).
            grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
            # MTP-3 produces four logit rows per request. Process two requests
            # per launch, then tile larger concurrent batches in groups of 8.
            # Unlike v2, this never falls back to one full LM-head read per row.
            max_rows_per_launch = 8
            for start in range(0, batch, max_rows_per_launch):
                stop = min(start + max_rows_per_launch, batch)
                nb = stop - start
                out_chunk = out[start:stop]
                x_chunk = x_fp16[start:stop]
                lm_head._ww_kernel_v3[grid](
                    out_chunk, lm_head._ww_int8, x_chunk,
                    lm_head._ww_scales, M, K,
                    out_chunk.stride(0), x_chunk.stride(0), NUM_BATCH=nb)
            logits = out.view(hidden_states.shape[:-1] + (M,))
            if embedding_bias is not None:
                logits = logits + embedding_bias
            return logits

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)'''

    # Upgrade an existing v1/v2 patch in place when layering on a patched image.
    for old_patch_start in (old_v2_start, old_v1_start):
        if old_patch_start not in content:
            continue
        idx_start = content.index(old_patch_start)
        fallback = "        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)"
        remaining = content[idx_start:]
        patch_end = remaining.find(fallback)
        if patch_end >= 0:
            idx_end = idx_start + patch_end + len(fallback)
            content = content[:idx_start] + new + content[idx_end:]
            with open(TARGET, "w") as f:
                f.write(content)
            print("OK: INT8 LM Head v3 patch applied (upgraded old patch)")
            return

    # Try clean apply (no v1)
    if old in content:
        content = content.replace(old, new)
        with open(TARGET, "w") as f:
            f.write(content)
        print("OK: INT8 LM Head v3 patch applied (clean)")
    else:
        print("FAIL: pattern not found (neither old patch nor clean source)")
        sys.exit(1)


if __name__ == "__main__":
    apply()
