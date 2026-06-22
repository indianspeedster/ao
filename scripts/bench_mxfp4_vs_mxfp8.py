# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Head-to-head forward grouped-GEMM bench: MXFP4 (torchao) vs MXFP8 (tuned ROCm
# triton kernel from the grouped-gemms repo). GEMM-only (operands pre-quantized).
# Goal: MXFP4 should be >= 1.7x faster than MXFP8.

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

# MXFP8 tuned kernel lives in the grouped-gemms repo.
sys.path.insert(0, "/it-share/shekhar/grouped-gemms")
from kernels import triton_mxfp8_grouped_mm  # noqa: E402

from torchao.prototype.moe_training.kernels.mxfp4.rocm_mxfp4_mm import (  # noqa: E402
    triton_mxfp4_grouped_mm,
)
from torchao.prototype.moe_training.kernels.mxfp4.rocm_mxfp4_mm_opt import (  # noqa: E402
    triton_mxfp4_grouped_mm_opt,
)
from torchao.prototype.moe_training.mxfp4_grouped_mm import _to_mxfp4  # noqa: E402
from torchao.prototype.mx_formats.mx_tensor import to_mx  # noqa: E402
from torchao.prototype.moe_training.utils import generate_jagged_offs  # noqa: E402
from torchao.utils import is_MI350  # noqa: E402

device = torch.device("cuda")
BS = 32


def bench_us(fn, *a, **k) -> float:
    return do_bench(lambda: fn(*a, **k), return_mode="median") * 1e3


@dataclass(frozen=True)
class Cfg:
    e: int
    m: int
    n: int
    k: int


_LLAMA4 = list(itertools.product([1, 2, 4, 8], [16640], [2048, 5120, 8192], [2048, 5120, 8192]))
_DSV3 = [(4, 32768, 2048, 7168), (8, 32768, 2048, 7168),
         (4, 128000, 2048, 7168), (8, 128000, 2048, 7168)]


def get_configs(s):
    if s == "llama4":
        return [Cfg(e, m, n, k) for e, m, n, k in _LLAMA4]
    if s == "dsv3":
        return [Cfg(e, m, n, k) for e, m, n, k in _DSV3]
    raise ValueError(s)


def run(c: Cfg):
    e, m, n, k = c.e, c.m, c.n, c.k
    A = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    B_t = torch.randn((e, n, k), dtype=torch.bfloat16, device=device).transpose(-2, -1)
    offs = generate_jagged_offs(e, m, multiple_of=BS)
    bf16_us = bench_us(torch._grouped_mm, A, B_t, offs, out_dtype=torch.bfloat16)

    W_NK = B_t.transpose(-2, -1).contiguous()  # (E, N, K)

    # MXFP4 operands
    a4_p, a4_s = _to_mxfp4(A, BS, None)
    w4_p, w4_s = _to_mxfp4(W_NK, BS, None)
    mxfp4_us = bench_us(triton_mxfp4_grouped_mm, a4_p, w4_p, a4_s, w4_s, offs)
    mxfp4o_us = bench_us(triton_mxfp4_grouped_mm_opt, a4_p, w4_p, a4_s, w4_s, offs)

    # MXFP8 operands: to_mx returns (scale, data)
    a8_s, a8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=BS)
    w8_s, w8 = to_mx(W_NK, elem_dtype=torch.float8_e4m3fn, block_size=BS)
    mxfp8_us = bench_us(triton_mxfp8_grouped_mm, a8, w8, a8_s, w8_s, offs)

    flops = 2 * m * n * k
    return {
        "bf16_us": round(bf16_us, 1),
        "mxfp8_us": round(mxfp8_us, 1),
        "mxfp4_us": round(mxfp4_us, 1),
        "mxfp4o_us": round(mxfp4o_us, 1),
        "old_vs8": mxfp8_us / mxfp4_us,
        "opt_vs8": mxfp8_us / mxfp4o_us,
        "mxfp8_TF": (flops / 1e12) / (mxfp8_us / 1e6),
        "mxfp4o_TF": (flops / 1e12) / (mxfp4o_us / 1e6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="llama4", choices=("llama4", "dsv3"))
    args = ap.parse_args()
    assert torch.cuda.is_available() and is_MI350(), "needs gfx950"
    torch.manual_seed(123)
    import random
    random.seed(123)

    rows, geo_opt8, geo_optbf, ge = [], 0.0, 0.0, 0
    cfgs = get_configs(args.shapes)
    for c in tqdm(cfgs):
        r = run(c)
        opt_vs_bf16 = r["bf16_us"] / r["mxfp4o_us"]
        rows.append([c.e, c.m, c.n, c.k, r["bf16_us"], r["mxfp8_us"], r["mxfp4o_us"],
                     f"{opt_vs_bf16:.2f}x", f"{r['opt_vs8']:.2f}x",
                     round(r["mxfp4o_TF"], 1)])
        geo_opt8 += math.log(r["opt_vs8"]); geo_optbf += math.log(opt_vs_bf16); ge += 1
    print(tabulate(rows, headers=["E", "M", "N", "K", "bf16_us", "mxfp8_us", "mxfp4opt_us",
                                   "opt_vs_bf16", "opt_vs_mxfp8", "mxfp4opt_TF"]))
    print(f"\nGeomean MXFP4-opt ({ge} shapes): vs bf16={math.exp(geo_optbf/ge):.3f}x  "
          f"vs MXFP8={math.exp(geo_opt8/ge):.3f}x")


if __name__ == "__main__":
    main()
