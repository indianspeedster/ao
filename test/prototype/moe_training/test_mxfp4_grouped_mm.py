# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_MI350

# MXFP4 training is currently supported only on ROCm gfx950 (MI350/MI355X) via
# torchao's Triton tl.dot_scaled kernels.
if not (torch.cuda.is_available() and is_MI350()):
    pytest.skip(
        "MXFP4 grouped GEMM requires ROCm gfx950 (MI350/MI355X)",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.mxfp4_grouped_mm import (
    _to_mxfp4_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

# fp4 (e2m1, 4-bit) with E8M0 power-of-2 block scales lands around ~15-16 dB SQNR
# on iid-Gaussian inputs; 14 dB leaves margin while still catching regressions.
MIN_SQNR = 14.0


def _per_group_sqnr(ref, out, offs):
    """SQNR (dB) computed independently for each expert's token rows.

    Uniform values mean every group was fully computed. A single low group flags
    uncomputed/garbage rows (e.g. the grouped-GEMM grid covering only the average
    tokens-per-expert instead of the largest group).
    """
    starts = [0] + offs[:-1].tolist()
    ends = offs.tolist()
    return [
        compute_error(ref[s:e], out[s:e]).item()
        for s, e in zip(starts, ends)
        if e > s
    ]


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
def test_mxfp4_grouped_mm_2d_3d(M, K, N, num_experts):
    """Forward grouped GEMM: out = input @ weight^T, per expert group."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    # Jagged (imbalanced) offsets exercise the per-group grid coverage.
    offs = generate_jagged_offs(num_experts, M, multiple_of=32)

    ref_out = torch._grouped_mm(x, w_t, offs=offs, out_dtype=torch.bfloat16)
    out = _to_mxfp4_then_scaled_grouped_mm(x, w_t, offs=offs)

    assert not torch.isnan(out).any(), "MXFP4 grouped GEMM produced NaNs"

    sqnr = compute_error(ref_out, out)
    assert sqnr >= MIN_SQNR, f"overall sqnr {sqnr} < {MIN_SQNR}"

    # Every expert group must be computed to the same fidelity: the spread across
    # groups should be small. A collapsed group would betray uncomputed rows.
    per_group = _per_group_sqnr(ref_out, out, offs)
    assert min(per_group) >= MIN_SQNR, (
        f"min per-group sqnr {min(per_group):.2f} < {MIN_SQNR}; "
        f"per-group sqnr = {[round(v, 2) for v in per_group]}"
    )


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8))
def test_mxfp4_grouped_mm_autograd(M, K, N, num_experts):
    """Backward pass: grad_input (dgrad) and grad_weight (wgrad) GEMMs."""
    x = torch.randn(
        M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    w_t = torch.randn(
        num_experts, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)
    w_t_ref = w_t.detach().clone().requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=32)

    out = _to_mxfp4_then_scaled_grouped_mm(x, w_t, offs=offs)
    ref_out = torch._grouped_mm(x_ref, w_t_ref, offs=offs, out_dtype=torch.bfloat16)

    grad_out = torch.randn_like(ref_out)
    out.backward(grad_out)
    ref_out.backward(grad_out)

    for got, ref, name in (
        (x.grad, x_ref.grad, "grad_input (dgrad)"),
        (w_t.grad, w_t_ref.grad, "grad_weight (wgrad)"),
    ):
        assert not torch.isnan(got).any(), f"{name} produced NaNs"
        sqnr = compute_error(ref, got)
        assert sqnr >= MIN_SQNR, f"{name} sqnr {sqnr} < {MIN_SQNR}"
