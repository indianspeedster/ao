"""Finer MXFP4-opt sweep on one DSV3 shape, incl. kpack/BLOCK_K=512/num_stages.

Usage: HIP_VISIBLE_DEVICES=<g> python sweep_mxfp4_dsv3_fine.py <E> <M> <N> <K> <mxfp8_us>
Prints JSON with best config + ratio vs the given tuned-MXFP8 baseline.
"""
import itertools
import json
import sys

import torch
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.mxfp4.rocm_mxfp4_mm_opt import (
    triton_mxfp4_grouped_mm_opt,
)
from torchao.prototype.moe_training.mxfp4_grouped_mm import _to_mxfp4
from torchao.prototype.moe_training.utils import generate_jagged_offs

dev = "cuda"
BS = 32

GRID = list(itertools.product(
    [128, 256],   # BLOCK_M
    [128, 256],   # BLOCK_N
    [256, 512],   # BLOCK_K
    [4, 8],       # GROUP_M
    [4, 8],       # num_warps
    [1, 2],       # num_stages
    [16, 32],     # nonkdim
    [0, 2],       # waves_per_eu
    [1, 2],       # kpack
))


def b(fn, *a, **k):
    return do_bench(lambda: fn(*a, **k), return_mode="median") * 1e3


def main():
    E, M, N, K = (int(x) for x in sys.argv[1:5])
    mxfp8_us = float(sys.argv[5])
    torch.manual_seed(0)
    import random
    random.seed(0)
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16)
    offs = generate_jagged_offs(E, M, multiple_of=BS)
    a4, a4s = _to_mxfp4(A, BS, None)
    w4, w4s = _to_mxfp4(W, BS, None)

    best = None
    for (bm, bn, bk, gm, nw, ns, nk, we, kp) in GRID:
        if K % bk:
            continue
        try:
            us = b(triton_mxfp4_grouped_mm_opt, a4, w4, a4s, w4s, offs,
                   BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                   num_warps=nw, num_stages=ns, matrix_instr_nonkdim=nk,
                   waves_per_eu=we, kpack=kp)
        except Exception:
            continue
        if best is None or us < best[0]:
            best = (us, dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                             num_warps=nw, num_stages=ns, matrix_instr_nonkdim=nk,
                             waves_per_eu=we, kpack=kp))
    print(json.dumps({"E": E, "M": M, "N": N, "K": K, "mxfp8_us": mxfp8_us,
                      "best_us": round(best[0], 2),
                      "ratio_vs_mxfp8": round(mxfp8_us / best[0], 3),
                      "cfg": best[1]}))


if __name__ == "__main__":
    main()
