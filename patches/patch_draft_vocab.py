#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add a draft-only vocabulary subset to the patched Qwen3.5 MTP path.

The target model keeps the complete V3 INT8 LM head. Only Qwen3_5MTP's logits
processor gets a compact copy of selected rows. The draft logits are scattered
back into a full-vocabulary tensor with -inf elsewhere, so vLLM's sampler and
token IDs continue to see the normal vocabulary layout.

Runtime control:
    DGX_SPARK_DRAFT_VOCAB=65536   # default; 0 disables FR-Spec
    DGX_SPARK_DRAFT_VOCAB_TAIL=512
"""

from __future__ import annotations

import os
from pathlib import Path


LOGITS_TARGET = Path(
    os.environ.get(
        "DGX_SPARK_LOGITS_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py",
    )
)
MTP_TARGET = Path(
    os.environ.get(
        "DGX_SPARK_MTP_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py",
    )
)

MARKER = "DGX_SPARK_FR_SPEC_V1"


def patch_mtp() -> None:
    content = MTP_TARGET.read_text()
    if MARKER in content:
        print(f"SKIP: {MTP_TARGET} already has {MARKER}")
        return

    old = "        self.logits_processor = LogitsProcessor(config.vocab_size)\n"
    if content.count(old) != 1:
        raise SystemExit(
            f"FAIL: expected one Qwen3_5MTP LogitsProcessor constructor, found {content.count(old)}"
        )
    new = old + (
        f"        # {MARKER}: only this processor may use the compact draft head.\n"
        "        self.logits_processor._dgx_mtp_draft = True\n"
    )
    MTP_TARGET.write_text(content.replace(old, new, 1))
    print(f"OK: marked Qwen3_5MTP draft processor in {MTP_TARGET}")


def patch_logits() -> None:
    content = LOGITS_TARGET.read_text()
    if MARKER in content:
        print(f"SKIP: {LOGITS_TARGET} already has {MARKER}")
        return
    if "DGX_SPARK_INT8_LMHEAD_V3" not in content:
        raise SystemExit("FAIL: FR-Spec requires the V3 INT8 LM-head patch")

    start_text = "        if hasattr(lm_head, '_ww_int8'):\n"
    end_text = "        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)"
    start = content.find(start_text)
    if start < 0:
        raise SystemExit("FAIL: V3 INT8 dispatch start not found")
    end = content.find(end_text, start)
    if end < 0:
        raise SystemExit("FAIL: V3 INT8 dispatch end not found")

    replacement = f'''        # {MARKER}: compact LM head for MTP proposals only.
        if hasattr(lm_head, '_ww_int8'):
            full_M, K = lm_head._ww_int8.shape
            use_draft = getattr(self, '_dgx_mtp_draft', False)

            if use_draft and not hasattr(self, '_dgx_draft_initialized'):
                self._dgx_draft_initialized = True
                import os as _os
                draft_top = int(_os.environ.get('DGX_SPARK_DRAFT_VOCAB', '65536'))
                draft_tail = int(_os.environ.get('DGX_SPARK_DRAFT_VOCAB_TAIL', '512'))
                draft_top = max(0, min(draft_top, full_M))
                draft_tail = max(0, min(draft_tail, full_M))
                tail_start = full_M - draft_tail
                draft_top = min(draft_top, tail_start)
                if draft_top > 0 or draft_tail > 0:
                    top_ids = torch.arange(draft_top, dtype=torch.long, device=lm_head._ww_int8.device)
                    tail_ids = torch.arange(tail_start, full_M, dtype=torch.long, device=lm_head._ww_int8.device)
                    ids = torch.cat((top_ids, tail_ids))
                    self._dgx_draft_ids = ids
                    self._dgx_draft_int8 = lm_head._ww_int8.index_select(0, ids).contiguous()
                    self._dgx_draft_scales = lm_head._ww_scales.index_select(0, ids).contiguous()
                    import sys as _sys
                    mib = self._dgx_draft_int8.numel() / 1024 / 1024
                    print(
                        f'DGX_SPARK_FR_SPEC: draft head {{ids.numel()}}/{{full_M}} rows, '
                        f'{{mib:.1f}} MiB INT8 (top={{draft_top}}, tail={{draft_tail}})',
                        file=_sys.stderr,
                        flush=True,
                    )

            draft_active = use_draft and hasattr(self, '_dgx_draft_int8')
            if draft_active:
                active_w = self._dgx_draft_int8
                active_scales = self._dgx_draft_scales
            else:
                active_w = lm_head._ww_int8
                active_scales = lm_head._ww_scales

            M = active_w.shape[0]
            x = hidden_states.view(-1, K)
            batch = x.shape[0]
            out = torch.empty(batch, M, dtype=torch.float16, device=x.device)
            x_fp16 = x.to(torch.float16)
            # Autotune-aware grid: BLOCK_M is chosen by autotuner per (M,K,NUM_BATCH).
            grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
            max_rows_per_launch = 8
            for start in range(0, batch, max_rows_per_launch):
                stop = min(start + max_rows_per_launch, batch)
                nb = stop - start
                out_chunk = out[start:stop]
                x_chunk = x_fp16[start:stop]
                lm_head._ww_kernel_v3[grid](
                    out_chunk, active_w, x_chunk,
                    active_scales, M, K,
                    out_chunk.stride(0), x_chunk.stride(0), NUM_BATCH=nb)

            compact_logits = out.view(hidden_states.shape[:-1] + (M,))
            if draft_active:
                if embedding_bias is not None:
                    compact_logits = compact_logits + embedding_bias.index_select(
                        0, self._dgx_draft_ids
                    )
                logits = torch.full(
                    hidden_states.shape[:-1] + (full_M,),
                    -float('inf'),
                    dtype=compact_logits.dtype,
                    device=compact_logits.device,
                )
                logits.index_copy_(-1, self._dgx_draft_ids, compact_logits)
                return logits

            if embedding_bias is not None:
                compact_logits = compact_logits + embedding_bias
            return compact_logits

'''

    content = content[:start] + replacement + content[end:]
    LOGITS_TARGET.write_text(content)
    print(f"OK: added draft-only compact INT8 head in {LOGITS_TARGET}")


def main() -> None:
    for path in (LOGITS_TARGET, MTP_TARGET):
        if not path.is_file():
            raise SystemExit(f"FAIL: missing target: {path}")
    patch_mtp()
    patch_logits()


if __name__ == "__main__":
    main()
