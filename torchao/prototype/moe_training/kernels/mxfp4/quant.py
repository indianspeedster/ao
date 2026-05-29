# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton MXFP4 quantization kernels (dim0 + dim1) for ROCm gfx950.

Quantizes bf16 -> packed fp4 (e2m1) with E8M0 block scales (block_size=32).
Output:
  data:   (M, N//2) uint8 packed (each byte holds 2 fp4 values: low nibble = even idx, high = odd)
  scale:  (M, N//32) uint8 (E8M0)

The eager `torchao.prototype.mx_formats.mx_tensor.to_mx` for fp4 is pure
Python and dominates step time at MoE scale (hundreds of quants per step).
These Triton kernels move that work onto the GPU.
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_available = is_ROCM() and _triton_kernels_available

# OCP MXFP4 (E2M1) constants
F4_E2M1_MAX = 6.0  # max representable magnitude (Python constant)


if _available:
    import triton
    import triton.language as tl

    F4_E2M1_MAX_TL = tl.constexpr(6.0)

    @triton.jit
    def _f32_to_e2m1_unpacked(x):
        """
        Convert float32 values to e2m1 (fp4) encoding as uint8 (low nibble valid).

        E2M1 encoding (4 bits = sign|exp(2)|mantissa(1)):
          0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
          (sign bit set = negative of the same magnitude)
        """
        sign = (x < 0).to(tl.uint8) << 3
        ax = tl.abs(x)
        # Encode magnitude using the 8 representable positive values
        # We compare against midpoints to round to nearest.
        mag = tl.where(ax < 0.25, 0,
              tl.where(ax < 0.75, 1,
              tl.where(ax < 1.25, 2,
              tl.where(ax < 1.75, 3,
              tl.where(ax < 2.5,  4,
              tl.where(ax < 3.5,  5,
              tl.where(ax < 5.0,  6, 7)))))))
        return sign | mag.to(tl.uint8)

    @triton.jit
    def _to_mxfp4_dim0_kernel(
        x_ptr,           # (n_rows, n_cols) bf16
        output_ptr,      # (n_rows, n_cols//2) uint8 packed
        scale_ptr,       # (n_rows, n_cols//32) uint8 e8m0
        n_rows, n_cols,
        ROW_TILE: tl.constexpr,
        COL_TILE: tl.constexpr,         # fp4 elements per col tile
        SCALE_BLOCK: tl.constexpr,      # 32
    ):
        """Quantize a (ROW_TILE, COL_TILE) block to packed fp4 + e8m0 scales."""
        SCALES_PER_TILE: tl.constexpr = COL_TILE // SCALE_BLOCK

        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)

        row_off = pid_r * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        col_off = pid_c * COL_TILE + tl.arange(0, COL_TILE)[None, :]
        mask = (row_off < n_rows) & (col_off < n_cols)

        # Load bf16 block
        x = tl.load(x_ptr + row_off.to(tl.int64) * n_cols + col_off, mask=mask, other=0.0).to(tl.float32)

        # Per-block scale (e8m0)
        x_blk = x.reshape(ROW_TILE * SCALES_PER_TILE, SCALE_BLOCK)
        amax = tl.max(tl.abs(x_blk), axis=1)
        # rceil(log2(amax / max_pos)) -> e8m0 biased exponent
        # Need 2^e * F4_E2M1_MAX >= amax  => e >= log2(amax / F4_E2M1_MAX)
        ratio = amax / F4_E2M1_MAX_TL
        # exponent = ceil(log2(ratio)); clamp ratio>0 to avoid log2(0)
        ratio = tl.where(ratio > 0, ratio, 1e-30)
        exp_unbiased = tl.ceil(tl.log2(ratio)).to(tl.int32)
        # clamp e8m0 range: biased exponent in [0, 254]; 255 reserved for NaN
        exp_unbiased = tl.maximum(exp_unbiased, -127)
        exp_unbiased = tl.minimum(exp_unbiased, 127)
        scale_e8m0 = (exp_unbiased + 127).to(tl.uint8)

        # Reciprocal scale = 2^-exp_unbiased; build via bit pattern: (254 - biased) << 23
        inv_scale_bits = ((254 - scale_e8m0.to(tl.int32)) << 23)
        inv_scale = inv_scale_bits.to(tl.float32, bitcast=True)
        x_scaled = x_blk * inv_scale[:, None]

        # Encode to fp4 (uint8 nibble)
        nibbles = _f32_to_e2m1_unpacked(x_scaled).reshape(ROW_TILE, COL_TILE)

        # Pack pairs of (even, odd) cols into bytes via reshape + reduce.
        # nibbles_3d shape: (ROW_TILE, COL_TILE//2, 2); pack[..., 0] is low nibble, [..., 1] is high.
        # Use shift+sum trick: multiply odd entries by 16 (shift left 4) and sum along the pair axis.
        nibbles_3d = nibbles.reshape(ROW_TILE, COL_TILE // 2, 2).to(tl.uint8)
        # Build a (2,) shift vector: [1, 16]
        shift = tl.arange(0, 2)
        shift = (1 << (shift * 4)).to(tl.uint8)  # [1, 16]
        # Multiply and reduce along last axis
        packed = tl.sum((nibbles_3d * shift[None, None, :]).to(tl.int32), axis=2).to(tl.uint8)

        # Store packed data
        out_col_off = pid_c * (COL_TILE // 2) + tl.arange(0, COL_TILE // 2)[None, :]
        out_mask = (row_off < n_rows) & (out_col_off < (n_cols // 2))
        tl.store(
            output_ptr + row_off.to(tl.int64) * (n_cols // 2) + out_col_off,
            packed,
            mask=out_mask,
        )

        # Store scales
        scales_per_row = n_cols // SCALE_BLOCK
        sc_row = (pid_r * ROW_TILE + tl.arange(0, ROW_TILE))[:, None]
        sc_col = (pid_c * SCALES_PER_TILE + tl.arange(0, SCALES_PER_TILE))[None, :]
        sc_mask = (sc_row < n_rows) & (sc_col < scales_per_row)
        tl.store(
            scale_ptr + sc_row * scales_per_row + sc_col,
            scale_e8m0.reshape(ROW_TILE, SCALES_PER_TILE),
            mask=sc_mask,
        )

    def triton_to_mxfp4_dim0(x: torch.Tensor):
        """
        Quantize bf16 (M, N) -> packed fp4 (M, N//2) uint8 + e8m0 scales (M, N//32).

        Uses RCEIL scaling. N must be a multiple of 32.
        Returns (qdata_packed_uint8, scale_uint8).
        """
        assert x.is_contiguous()
        assert x.dtype == torch.bfloat16
        x = x.reshape(-1, x.shape[-1])
        M, N = x.shape
        assert N % 32 == 0, f"N must be multiple of 32, got {N}"

        qdata = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
        scale = torch.empty((M, N // 32), dtype=torch.uint8, device=x.device)

        ROW_TILE = 64
        COL_TILE = 128 if N >= 128 else N
        grid = (triton.cdiv(M, ROW_TILE), triton.cdiv(N, COL_TILE))
        _to_mxfp4_dim0_kernel[grid](
            x, qdata, scale, M, N,
            ROW_TILE=ROW_TILE, COL_TILE=COL_TILE, SCALE_BLOCK=32,
            num_warps=4, num_stages=2,
        )
        return qdata, scale

    def triton_to_mxfp4_dim1(x: torch.Tensor):
        """
        Quantize bf16 (M, N) along dim1 (the M dim becomes the reduction dim).
        Returns:
          qdata: (N, M//2) uint8 packed (column-major view of original)
          scale: (N, M//32) uint8 e8m0
        """
        assert x.is_contiguous()
        x_t = x.transpose(-2, -1).contiguous()  # (N, M)
        return triton_to_mxfp4_dim0(x_t)

else:
    def triton_to_mxfp4_dim0(*args, **kwargs):
        raise NotImplementedError("requires ROCm + triton")

    def triton_to_mxfp4_dim1(*args, **kwargs):
        raise NotImplementedError("requires ROCm + triton")
