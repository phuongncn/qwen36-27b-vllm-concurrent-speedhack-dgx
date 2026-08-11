#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extend the V3 weight-reuse LM-head kernel from 8 to 16 rows per launch.

This is deliberately a mechanical extension of the numerically qualified V3
kernel. It keeps FP32 accumulators and row scales; the GPU A/B decides whether
the larger register footprint is faster than two x8 launches on SM121.
"""

from __future__ import annotations

import os
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "DGX_SPARK_LOGITS_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py",
    )
)
MARKER = "DGX_SPARK_INT8_LMHEAD_X16_V1"


def replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count != 1:
        raise SystemExit(f"FAIL: {label}: expected one match, found {count}")
    return content.replace(old, new, 1)


def main() -> None:
    content = TARGET.read_text()
    if MARKER in content:
        print(f"SKIP: {MARKER} already applied")
        return
    if "DGX_SPARK_INT8_LMHEAD_V3" not in content:
        raise SystemExit("FAIL: x16 patch requires V3 INT8 LM head")

    content = replace_once(
        content,
        """                    acc4 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc5 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc6 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc7 = tl.zeros((BLOCK_M,), dtype=tl.float32)
""",
        """                    acc4 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc5 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc6 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc7 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    # DGX_SPARK_INT8_LMHEAD_X16_V1
                    acc8 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc9 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc10 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc11 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc12 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc13 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc14 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc15 = tl.zeros((BLOCK_M,), dtype=tl.float32)
""",
        "accumulators",
    )

    content = replace_once(
        content,
        """                        if NUM_BATCH > 7:
                            x7 = tl.load(x_ptr + 7 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc7 += tl.sum(w * x7[None, :], axis=1)
""",
        """                        if NUM_BATCH > 7:
                            x7 = tl.load(x_ptr + 7 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc7 += tl.sum(w * x7[None, :], axis=1)
                        if NUM_BATCH > 8:
                            x8 = tl.load(x_ptr + 8 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc8 += tl.sum(w * x8[None, :], axis=1)
                        if NUM_BATCH > 9:
                            x9 = tl.load(x_ptr + 9 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc9 += tl.sum(w * x9[None, :], axis=1)
                        if NUM_BATCH > 10:
                            x10 = tl.load(x_ptr + 10 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc10 += tl.sum(w * x10[None, :], axis=1)
                        if NUM_BATCH > 11:
                            x11 = tl.load(x_ptr + 11 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc11 += tl.sum(w * x11[None, :], axis=1)
                        if NUM_BATCH > 12:
                            x12 = tl.load(x_ptr + 12 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc12 += tl.sum(w * x12[None, :], axis=1)
                        if NUM_BATCH > 13:
                            x13 = tl.load(x_ptr + 13 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc13 += tl.sum(w * x13[None, :], axis=1)
                        if NUM_BATCH > 14:
                            x14 = tl.load(x_ptr + 14 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc14 += tl.sum(w * x14[None, :], axis=1)
                        if NUM_BATCH > 15:
                            x15 = tl.load(x_ptr + 15 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc15 += tl.sum(w * x15[None, :], axis=1)
""",
        "loads",
    )

    content = replace_once(
        content,
        """                    if NUM_BATCH > 7:
                        tl.store(out_ptr + 7 * stride_ob + rows, (acc7 * s).to(tl.float16), mask=rmask)
""",
        """                    if NUM_BATCH > 7:
                        tl.store(out_ptr + 7 * stride_ob + rows, (acc7 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 8:
                        tl.store(out_ptr + 8 * stride_ob + rows, (acc8 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 9:
                        tl.store(out_ptr + 9 * stride_ob + rows, (acc9 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 10:
                        tl.store(out_ptr + 10 * stride_ob + rows, (acc10 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 11:
                        tl.store(out_ptr + 11 * stride_ob + rows, (acc11 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 12:
                        tl.store(out_ptr + 12 * stride_ob + rows, (acc12 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 13:
                        tl.store(out_ptr + 13 * stride_ob + rows, (acc13 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 14:
                        tl.store(out_ptr + 14 * stride_ob + rows, (acc14 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 15:
                        tl.store(out_ptr + 15 * stride_ob + rows, (acc15 * s).to(tl.float16), mask=rmask)
""",
        "stores",
    )

    content = replace_once(
        content,
        "            max_rows_per_launch = 8\n",
        "            max_rows_per_launch = 16\n",
        "dispatch width",
    )

    TARGET.write_text(content)
    print(f"OK: {MARKER} applied to {TARGET}")


if __name__ == "__main__":
    main()
