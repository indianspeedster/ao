# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
MXFP4 grouped GEMM autograd function for MoE training.

Uses torchao's MXTensor for fp4 quantization (torch.float4_e2m1fn_x2, packed
2-per-byte) with E8M0 block scales (block_size=32). Forward/dgrad compute in
MXFP4 via our ROCm Triton kernels; wgrad can be either MXFP4 or bf16
(wgrad_with_hp=True) for higher precision.
"""

import logging
from typing import Optional

import torch

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx
from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_ROCM

logger: logging.Logger = logging.getLogger(__name__)

# MXFP4 is currently only supported on ROCm gfx950 via our Triton kernels.
_ROCM_MXFP4_KERNELS_AVAILABLE = is_ROCM() and _triton_kernels_available


def _to_mxfp4(tensor: torch.Tensor, block_size: int, scale_calculation_mode: ScaleCalculationMode):
    """Quantize a bf16 tensor to MXFP4 (fp4 packed 2-per-byte + e8m0 scales).

    Uses our fast Triton kernel on ROCm; falls back to eager `to_mx` otherwise.
    """
    if isinstance(tensor, MXTensor):
        return tensor.qdata, tensor.scale
    if _ROCM_MXFP4_KERNELS_AVAILABLE and tensor.dtype == torch.bfloat16 and tensor.is_contiguous():
        from torchao.prototype.moe_training.kernels.mxfp4 import triton_to_mxfp4_dim0
        orig_shape = tensor.shape
        qdata, scale = triton_to_mxfp4_dim0(tensor)
        # Reshape back: qdata is (..., N//2), scale is (..., N//32)
        qdata = qdata.reshape(*orig_shape[:-1], orig_shape[-1] // 2)
        scale = scale.reshape(*orig_shape[:-1], orig_shape[-1] // block_size)
        return qdata, scale
    scale, qdata = to_mx(
        tensor,
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=block_size,
        scaling_mode=scale_calculation_mode,
    )
    return qdata, scale


def _dequantize_if_mxtensor(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    if isinstance(tensor, MXTensor):
        return tensor.dequantize(torch.bfloat16)
    return tensor


class _MXFP4GroupedMM(torch.autograd.Function):
    """
    Differentiable MXFP4 grouped GEMM for MoE training.

    Forward:  output[g] = input_act[group_g] @ weight[g]^T
    Backward: grad_input[g] = grad_output[group_g] @ weight[g]
              grad_weight[g] = grad_output[group_g]^T @ input_act[group_g]

    All three GEMMs run in MXFP4 by default; wgrad falls back to bf16 if
    wgrad_with_hp=True.
    """

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight_t: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        wgrad_with_hp: bool = False,
        scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    ) -> torch.Tensor:
        assert _ROCM_MXFP4_KERNELS_AVAILABLE, (
            "MXFP4 training requires ROCm with gfx950 (MI355X) or later"
        )
        assert input_act.ndim == 2, "input_act must be 2D"
        assert weight_t.ndim == 3, "weight_t must be 3D (E, K, N)"
        assert out_dtype == torch.bfloat16

        from torchao.prototype.moe_training.kernels.mxfp4.rocm_mxfp4_mm import (
            triton_mxfp4_grouped_mm,
        )

        block_size = 32
        ctx.block_size = block_size
        ctx.wgrad_with_hp = wgrad_with_hp
        ctx.scale_calculation_mode = scale_calculation_mode
        ctx.out_dtype = out_dtype

        # Quantize input (M, K) -> (M, K//2) packed fp4
        a_packed, a_scale = _to_mxfp4(input_act, block_size, scale_calculation_mode)
        # Quantize weight_t (E, K, N) -> quantize transposed to (E, N, K) -> packed (E, N, K//2)
        w_packed, w_scale = _to_mxfp4(
            weight_t.transpose(-2, -1).contiguous(), block_size, scale_calculation_mode
        )

        M = a_packed.shape[0]
        E = w_packed.shape[0]
        max_M_per_expert = (M + E - 1) // E

        output = triton_mxfp4_grouped_mm(
            a_packed, w_packed, a_scale, w_scale,
            group_end_offsets,
            out_dtype=out_dtype,
            max_M_per_expert=max_M_per_expert,
        )

        # Save for backward (keep hp inputs for wgrad path)
        ctx.save_for_backward(input_act, weight_t, group_end_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_act, weight_t, group_end_offsets = ctx.saved_tensors
        block_size = ctx.block_size
        scale_calculation_mode = ctx.scale_calculation_mode
        out_dtype = ctx.out_dtype
        wgrad_with_hp = ctx.wgrad_with_hp

        from torchao.prototype.moe_training.kernels.mxfp4.rocm_mxfp4_mm import (
            triton_mxfp4_grouped_mm,
            triton_mxfp4_wgrad,
        )

        # ----- dgrad: grad_input = grad_output @ weight -----
        # weight_t is (E, K, N); for dgrad the kernel wants B as (E, ?, contraction_last).
        # We pass weight_t directly (E, K, N): contraction over N gives (M, K).
        go_packed, go_scale = _to_mxfp4(
            grad_output.contiguous(), block_size, scale_calculation_mode
        )
        wt_packed, wt_scale = _to_mxfp4(
            weight_t.contiguous(), block_size, scale_calculation_mode
        )

        M = go_packed.shape[0]
        E = wt_packed.shape[0]
        max_M_per_expert = (M + E - 1) // E

        grad_input = triton_mxfp4_grouped_mm(
            go_packed, wt_packed, go_scale, wt_scale,
            group_end_offsets,
            out_dtype=out_dtype,
            max_M_per_expert=max_M_per_expert,
        )

        # ----- wgrad: grad_weight = grad_output^T @ input_act -----
        if wgrad_with_hp:
            go_hp = _dequantize_if_mxtensor(grad_output, block_size)
            ia_hp = _dequantize_if_mxtensor(input_act, block_size)
            grad_weight = torch._grouped_mm(
                go_hp.transpose(-2, -1).contiguous(),
                ia_hp,
                offs=group_end_offsets,
                out_dtype=out_dtype,
            )
            grad_weight_t = grad_weight.transpose(-2, -1)
        else:
            # MXFP4 wgrad: quantize both inputs along dim1 (the M/token dimension)
            go_hp = _dequantize_if_mxtensor(grad_output, block_size).contiguous()
            ia_hp = _dequantize_if_mxtensor(input_act, block_size).contiguous()

            # Transpose so dim0 becomes the M dimension, then quantize along dim0
            # dim1-of-(M,N) = dim0-of-(N,M)
            go_dim1_packed, go_dim1_scale = _to_mxfp4(
                go_hp.transpose(-2, -1).contiguous(), block_size, scale_calculation_mode
            )  # (N, M//2), (N, M//32)
            ia_dim1_packed, ia_dim1_scale = _to_mxfp4(
                ia_hp.transpose(-2, -1).contiguous(), block_size, scale_calculation_mode
            )  # (K, M//2), (K, M//32)

            grad_weight = triton_mxfp4_wgrad(
                go_dim1_packed, go_dim1_scale,
                ia_dim1_packed, ia_dim1_scale,
                group_end_offsets,
                out_dtype=out_dtype,
            )  # (E, N, K)
            grad_weight_t = grad_weight.transpose(-2, -1)  # (E, K, N)

        return (
            grad_input,
            grad_weight_t,
            None,  # group_end_offsets
            None,  # out_dtype
            None,  # wgrad_with_hp
            None,  # scale_calculation_mode
        )


def _to_mxfp4_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    wgrad_with_hp: bool = False,
    scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
) -> torch.Tensor:
    """
    Differentiable MXFP4 grouped GEMM with dynamic quantization.

    Args:
        A: Input activations, shape (M, K) bf16
        B_t: Expert weights transposed, shape (E, K, N) bf16
        offs: (E,) int32, cumulative token end indices per expert
        out_dtype: bfloat16
        wgrad_with_hp: If True, compute weight gradient in bf16 instead of MXFP4
        scale_calculation_mode: RCEIL or FLOOR

    Returns:
        output: (M, N) bfloat16
    """
    return _MXFP4GroupedMM.apply(
        A, B_t, offs, out_dtype, wgrad_with_hp, scale_calculation_mode,
    )
