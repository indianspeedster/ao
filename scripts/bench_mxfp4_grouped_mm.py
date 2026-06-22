# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Forward MXFP4 grouped-GEMM benchmark for ROCm gfx950 (MI350/MI355X).
# Mirrors grouped-gemms/bench.py: bf16 baseline (torch._grouped_mm) vs the MXFP4
# Triton grouped GEMM, over Llama4 / DeepSeek-V3 expert shapes with jagged offs.
#
# Three MXFP4 arms:
#   mxfp4_gemm : raw triton_mxfp4_grouped_mm on pre-quantized inputs (kernel
#                throughput, scales already computed).
#   mxfp4_e2e  : dynamic quant of A and W + the GEMM (what the training forward
#                actually pays every step, since activations are re-quantized).

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.mxfp4.rocm_mxfp4_mm import (
    triton_mxfp4_grouped_mm,
)
from torchao.prototype.moe_training.mxfp4_grouped_mm import _to_mxfp4
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.utils import is_MI350

device = torch.device("cuda")
BLOCK_SIZE = 32


def bench_us(fn, *args, **kwargs) -> float:
    """Median runtime of fn in microseconds."""
    return do_bench(lambda: fn(*args, **kwargs), return_mode="median") * 1e3


@dataclass(frozen=True)
class Cfg:
    e: int
    m: int
    n: int
    k: int


# Llama4 shapes (same sweep as the ao MXFP8 bench).
_LLAMA4_M = [16640]
_LLAMA4_K = [2048, 5120, 8192]
_LLAMA4_N = [2048, 5120, 8192]
_LLAMA4_E = [1, 2, 4, 8]

# DeepSeek-V3 671B expert shapes: N=2048, K=7168, E in {4,8}, M in {32768,128000}.
_DSV3_EMNK = [
    (4, 32768, 2048, 7168),
    (8, 32768, 2048, 7168),
    (4, 128000, 2048, 7168),
    (8, 128000, 2048, 7168),
]

SHAPE_SETS = ("llama4", "dsv3")


def get_configs(shape_set: str) -> List[Cfg]:
    if shape_set == "llama4":
        return [
            Cfg(e=e, m=m, n=n, k=k)
            for e, m, n, k in itertools.product(
                _LLAMA4_E, _LLAMA4_M, _LLAMA4_N, _LLAMA4_K
            )
        ]
    if shape_set == "dsv3":
        return [Cfg(e=e, m=m, n=n, k=k) for e, m, n, k in _DSV3_EMNK]
    raise ValueError(f"unknown shape set: {shape_set}")


def _mxfp4_e2e(A, Wt_NK, offs):
    """Quantize A (M,K) and weight (E,N,K) to MXFP4, then run the grouped GEMM.

    Wt_NK is the transposed/contiguous weight (E, N, K) the forward feeds to
    _to_mxfp4; this matches _MXFP4GroupedMM.forward's quantization steps.
    """
    a_packed, a_scale = _to_mxfp4(A, BLOCK_SIZE, None)
    w_packed, w_scale = _to_mxfp4(Wt_NK, BLOCK_SIZE, None)
    return triton_mxfp4_grouped_mm(a_packed, w_packed, a_scale, w_scale, offs)


def run(cfg: Cfg):
    e, m, n, k = cfg.e, cfg.m, cfg.n, cfg.k
    A = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_t = torch.randn((e, n, k), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    offs = generate_jagged_offs(e, m, multiple_of=BLOCK_SIZE)

    # bf16 baseline
    bf16_us = bench_us(torch._grouped_mm, A, B_t, offs, out_dtype=torch.bfloat16)

    # Pre-quantized inputs for the GEMM-only arm.
    Wt_NK = B_t.transpose(-2, -1).contiguous()  # (E, N, K)
    a_packed, a_scale = _to_mxfp4(A, BLOCK_SIZE, None)
    w_packed, w_scale = _to_mxfp4(Wt_NK, BLOCK_SIZE, None)
    gemm_us = bench_us(
        triton_mxfp4_grouped_mm, a_packed, w_packed, a_scale, w_scale, offs
    )

    # End-to-end: dynamic quant of A + W, then GEMM.
    e2e_us = bench_us(_mxfp4_e2e, A, Wt_NK, offs)

    flops = 2 * m * n * k
    return {
        "bf16_us": round(bf16_us, 2),
        "gemm_us": round(gemm_us, 2),
        "e2e_us": round(e2e_us, 2),
        "gemm_x": bf16_us / gemm_us,
        "e2e_x": bf16_us / e2e_us,
        "bf16_TF": (flops / 1e12) / (bf16_us / 1e6),
        "gemm_TF": (flops / 1e12) / (gemm_us / 1e6),
        "e2e_TF": (flops / 1e12) / (e2e_us / 1e6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="llama4", choices=SHAPE_SETS)
    args = ap.parse_args()

    assert torch.cuda.is_available() and is_MI350(), (
        "MXFP4 grouped GEMM requires ROCm gfx950 (MI350/MI355X)"
    )

    torch.manual_seed(123)
    import random

    random.seed(123)

    rows, geo = [], {"gemm_x": 0.0, "e2e_x": 0.0, "gemm_TF": 0.0, "e2e_TF": 0.0}
    cfgs = get_configs(args.shapes)
    for c in tqdm(cfgs):
        r = run(c)
        rows.append([
            c.e, c.m, c.n, c.k,
            r["bf16_us"], r["gemm_us"], r["e2e_us"],
            f"{r['gemm_x']:.2f}x", f"{r['e2e_x']:.2f}x",
            round(r["bf16_TF"], 1), round(r["gemm_TF"], 1), round(r["e2e_TF"], 1),
        ])
        for kk in geo:
            geo[kk] += math.log(r[kk])

    print(tabulate(rows, headers=[
        "E", "M", "N", "K",
        "bf16_us", "mxfp4_gemm_us", "mxfp4_e2e_us",
        "gemm_x", "e2e_x",
        "bf16_TF", "gemm_TF", "e2e_TF",
    ]))
    ngeo = len(cfgs)
    print(
        f"\nGeomean vs bf16 ({ngeo} shapes): "
        f"mxfp4_gemm={math.exp(geo['gemm_x']/ngeo):.3f}x  "
        f"mxfp4_e2e={math.exp(geo['e2e_x']/ngeo):.3f}x"
    )
    print(
        f"Geomean TFLOPS: mxfp4_gemm={math.exp(geo['gemm_TF']/ngeo):.1f}  "
        f"mxfp4_e2e={math.exp(geo['e2e_TF']/ngeo):.1f}"
    )


if __name__ == "__main__":
    main()
