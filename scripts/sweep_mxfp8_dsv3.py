"""Sweep MXFP8 configs on one DSV3 shape (fair tuned-vs-tuned baseline).

Usage: HIP_VISIBLE_DEVICES=<g> python sweep_mxfp8_dsv3.py <E> <M> <N> <K>
"""
import itertools
import json
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, "/it-share/shekhar/grouped-gemms")
from kernels import triton_mxfp8_grouped_mm

from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.moe_training.utils import generate_jagged_offs

dev = "cuda"
BS = 32

# MXFP8 kernel's documented search space.
GRID = list(itertools.product(
    [64, 128, 256],  # BLOCK_M
    [128, 256],      # BLOCK_N
    [128, 256],      # BLOCK_K
    [1, 4, 8],       # GROUP_M
    [4, 8],          # num_warps
    [16, 32],        # nonkdim
    [0, 2],          # waves_per_eu
))


def bench_us(fn, *a, **k):
    return do_bench(lambda: fn(*a, **k), return_mode="median") * 1e3


def main():
    E, M, N, K = (int(x) for x in sys.argv[1:5])
    torch.manual_seed(0)
    import random
    random.seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16)
    offs = generate_jagged_offs(E, M, multiple_of=BS)
    a8s, a8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=BS)
    w8s, w8 = to_mx(W, elem_dtype=torch.float8_e4m3fn, block_size=BS)

    fallback = bench_us(triton_mxfp8_grouped_mm, a8, w8, a8s, w8s, offs)  # default cfg
    best = None
    for (bm, bn, bk, gm, nw, nk, we) in GRID:
        try:
            us = bench_us(
                triton_mxfp8_grouped_mm, a8, w8, a8s, w8s, offs,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                num_warps=nw, matrix_instr_nonkdim=nk, waves_per_eu=we,
            )
        except Exception:
            continue
        if best is None or us < best[0]:
            best = (us, (bm, bn, bk, gm, nw, nk, we))
    print(json.dumps({"E": E, "M": M, "N": N, "K": K,
                      "fallback_us": round(fallback, 2),
                      "best_us": round(best[0], 2), "cfg": best[1]}))


if __name__ == "__main__":
    main()
