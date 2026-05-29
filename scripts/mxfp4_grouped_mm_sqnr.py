# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
SQNR accuracy harness for the MXFP4 grouped GEMM (ROCm gfx950 / MI355X).

SQNR (signal-to-quantization-noise ratio, dB) = 20*log10(||ref|| / ||ref - test||).
Higher is better. For 4-bit (e2m1) expect ~16-22 dB on Gaussian inputs; real,
structured activations usually score higher.

Two references are reported:
  - total error vs a bf16 high-precision grouped matmul (torch._grouped_mm):
    folds in *all* MXFP4 error (quantization + kernel).
  - per-group SQNR: the key diagnostic for this kernel. Uniform per-group SQNR
    means every expert's token rows were actually computed; a single low group
    flags uncomputed/garbage rows (e.g. the old average-grid-bound bug).

Run: python scripts/mxfp4_grouped_mm_sqnr.py
"""

import torch

from torchao.prototype.moe_training.mxfp4_grouped_mm import (
    _to_mxfp4_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.quantization.utils import compute_error  # SQNR in dB


def per_group_sqnr(ref, out, offs):
    starts = [0] + offs[:-1].tolist()
    ends = offs.tolist()
    return [
        float(compute_error(ref[s:e], out[s:e]))
        for s, e in zip(starts, ends)
        if e > s
    ]


def main():
    torch.manual_seed(0)
    dev = "cuda"
    shapes = [
        (1024, 1024, 1024, 8),
        (1024, 2048, 4096, 8),
        (2048, 2048, 2048, 16),
        (4096, 1024, 1024, 8),
    ]
    hdr = f"{'M':>5} {'K':>5} {'N':>5} {'E':>3} | {'overall':>8} | {'min grp':>8} | {'max grp':>8} | nan"
    print(hdr)
    print("-" * len(hdr))
    for (M, K, N, E) in shapes:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        Bt = torch.randn(E, K, N, dtype=torch.bfloat16, device=dev)
        offs = generate_jagged_offs(E, M, multiple_of=32)  # imbalanced groups
        ref = torch._grouped_mm(A, Bt, offs=offs, out_dtype=torch.bfloat16)
        out = _to_mxfp4_then_scaled_grouped_mm(A, Bt, offs=offs)
        overall = float(compute_error(ref, out))
        pg = per_group_sqnr(ref, out, offs)
        nan = int(torch.isnan(out).any())
        print(
            f"{M:>5} {K:>5} {N:>5} {E:>3} | {overall:>8.2f} | "
            f"{min(pg):>8.2f} | {max(pg):>8.2f} | {nan}"
        )


if __name__ == "__main__":
    main()
