# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton MXFP4 matmul kernels for ROCm (gfx950 / MI355X and later).

MXFP4 uses 4-bit float (e2m1) elements with E8M0 block scales (block_size=32),
giving ~4x memory savings vs bfloat16 and ~2x vs MXFP8. FP4 values are packed
2-per-byte in memory (torch.float4_e2m1fn_x2 dtype).

These kernels use tl.dot_scaled with "e2m1" format, which interprets loaded
uint8 tiles as packed fp4 and feeds E8M0 scales directly into gfx950's native
v_mfma_scale_f32_*_f8f6f4 instructions.

Tile shape convention:
  - Tensor storage: (M, K//2) uint8 packed (logical fp4 shape (M, K))
  - BLOCK_K is in *fp4 elements*; we load BLOCK_K//2 bytes per tile row
  - SCALE_BLOCK=32 fp4 elements per scale (same as MXFP8)

Contents:
  - triton_mxfp4_grouped_mm: Grouped GEMM for MoE forward/dgrad
  - triton_mxfp4_wgrad: Weight gradient grouped GEMM
  - triton_mxfp4_mm: Dense matmul for shared-expert linear paths
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_rocm_mxfp4_available = is_ROCM() and _triton_kernels_available

if _rocm_mxfp4_available:
    import triton
    import triton.language as tl

    # ==================== Grouped GEMM (fwd / dgrad) ====================

    @triton.jit
    def _mxfp4_grouped_mm_kernel(
        # A: packed fp4 (M, K//2) uint8
        A_ptr, A_stride_m, A_stride_kp,
        # B: packed fp4 (E, N, K//2) uint8
        B_ptr, B_stride_e, B_stride_n, B_stride_kp,
        # A scales: (M, K//32) uint8 (e8m0)
        A_scales_ptr, A_scales_stride_m, A_scales_stride_kb,
        # B scales: (E, N, K//32) uint8 (e8m0)
        B_scales_ptr, B_scales_stride_e, B_scales_stride_n, B_scales_stride_kb,
        # C: (M, N) bf16
        C_ptr, C_stride_m, C_stride_n,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,   # in fp4 elements
        SCALE_BLOCK: tl.constexpr,  # 32
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_g = tl.program_id(2)

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)

        m_base = group_start + pid_m * BLOCK_M
        if m_base >= group_end:
            return
        n_base = pid_n * BLOCK_N
        if n_base >= N:
            return

        m_offs = m_base + tl.arange(0, BLOCK_M)
        n_offs = n_base + tl.arange(0, BLOCK_N)
        m_mask = m_offs < group_end
        n_mask = n_offs < N

        # Packed byte dimension: BLOCK_K fp4 elems -> BLOCK_K//2 bytes
        BLOCK_KP: tl.constexpr = BLOCK_K // 2
        SUB_PER_BLOCK_K: tl.constexpr = BLOCK_K // SCALE_BLOCK
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_outer = K // BLOCK_K
        for k_outer in range(0, num_outer):
            kp_offs = k_outer * BLOCK_KP + tl.arange(0, BLOCK_KP)

            # Load A: (BLOCK_M, BLOCK_KP) packed uint8
            a = tl.load(
                A_ptr + m_offs[:, None] * A_stride_m + kp_offs[None, :] * A_stride_kp,
                mask=m_mask[:, None], other=0,
            )
            # Load B: (BLOCK_KP, BLOCK_N) packed uint8
            b = tl.load(
                B_ptr + pid_g * B_stride_e + n_offs[None, :] * B_stride_n + kp_offs[:, None] * B_stride_kp,
                mask=n_mask[None, :], other=0,
            )

            kb_offs = k_outer * SUB_PER_BLOCK_K + tl.arange(0, SUB_PER_BLOCK_K)
            a_scale = tl.load(
                A_scales_ptr + m_offs[:, None] * A_scales_stride_m + kb_offs[None, :] * A_scales_stride_kb,
                mask=m_mask[:, None], other=127,
            )
            b_scale = tl.load(
                B_scales_ptr + pid_g * B_scales_stride_e + n_offs[:, None] * B_scales_stride_n + kb_offs[None, :] * B_scales_stride_kb,
                mask=n_mask[:, None], other=127,
            )

            # tl.dot_scaled with "e2m1" interprets uint8 tiles as packed fp4
            # of shape (BLOCK_M, BLOCK_K) and (BLOCK_K, BLOCK_N).
            acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc, out_dtype=tl.float32)

        c_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            C_ptr + m_offs[:, None] * C_stride_m + n_offs[None, :] * C_stride_n,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp4_grouped_mm(
        input_act: torch.Tensor,      # (M, K//2) uint8 packed
        weight: torch.Tensor,         # (E, N, K//2) uint8 packed
        input_act_scales: torch.Tensor,   # (M, K//32) e8m0
        weight_scales: torch.Tensor,      # (E, N, K//32) e8m0
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        num_warps: int = 8,
        num_stages: int = 2,
        max_M_per_expert: int = 0,
    ) -> torch.Tensor:
        """
        MXFP4 grouped GEMM: output[g] = input_act[group_g] @ weight[g]^T

        Inputs are fp4 packed 2-per-byte. BLOCK_K counts fp4 elements.
        """
        M, Kp = input_act.shape
        K = Kp * 2
        E, N, Kp2 = weight.shape
        assert Kp == Kp2
        SCALE_BLOCK = 32

        output = torch.empty((M, N), dtype=out_dtype, device=input_act.device)

        grid_m_bound = max_M_per_expert if max_M_per_expert > 0 else M
        grid = (
            triton.cdiv(grid_m_bound, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
            E,
        )

        _mxfp4_grouped_mm_kernel[grid](
            input_act, input_act.stride(0), input_act.stride(1),
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            input_act_scales.view(torch.uint8),
            input_act_scales.stride(0), input_act_scales.stride(1),
            weight_scales.view(torch.uint8),
            weight_scales.stride(0), weight_scales.stride(1), weight_scales.stride(2),
            output, output.stride(0), output.stride(1),
            group_end_offsets,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

    # ==================== Weight gradient grouped GEMM ====================

    @triton.jit
    def _mxfp4_wgrad_kernel(
        # grad_output dim1-quantized: (N, M//2) packed uint8
        GO_ptr, GO_stride_n, GO_stride_mp,
        # go scales: (N, M//32) uint8
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        # input_act dim1-quantized: (K, M//2) packed uint8
        IA_ptr, IA_stride_k, IA_stride_mp,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        # output: (E, N, K) bf16
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,   # in fp4 elements
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base = pid_k * BLOCK_K
        if n_base >= N or k_base >= K:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs = k_base + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K

        BLOCK_MP: tl.constexpr = BLOCK_M // 2
        SUB_PER_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        num_m_iters = (M_g + BLOCK_M - 1) // BLOCK_M
        for m_iter in range(0, num_m_iters):
            m_base = group_start + m_iter * BLOCK_M
            mp_base = m_base // 2
            mp_offs = mp_base + tl.arange(0, BLOCK_MP)
            # Byte-level mask: include bytes whose fp4 elements fall in [m_base, group_end)
            mp_valid_end = (group_end + 1) // 2
            mp_mask = (mp_offs >= mp_base) & (mp_offs < mp_valid_end)

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + mp_offs[None, :] * GO_stride_mp,
                mask=n_mask[:, None] & mp_mask[None, :], other=0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + mp_offs[:, None] * IA_stride_mp,
                mask=mp_mask[:, None] & k_mask[None, :], other=0,
            )

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, SUB_PER_BLOCK_M)

            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None], other=127,
            )
            ia_scale = tl.load(
                IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask[:, None], other=127,
            )

            acc = tl.dot_scaled(go_tile, go_scale, "e2m1", ia_tile, ia_scale, "e2m1", acc=acc, out_dtype=tl.float32)

        c_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp4_wgrad(
        go_t: torch.Tensor,          # (N, M//2) packed uint8
        go_scale: torch.Tensor,      # (N, M//32) e8m0
        ia_t: torch.Tensor,          # (K, M//2) packed uint8
        ia_scale: torch.Tensor,      # (K, M//32) e8m0
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        BLOCK_M: int = 128,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """
        MXFP4 weight gradient: grad_W[g] = grad_output[group_g]^T @ input_act[group_g]

        Both inputs must be dim1-quantized to fp4 (packed 2-per-byte).
        Returns (E, N, K) bf16.
        """
        N, Mp = go_t.shape
        M = Mp * 2
        K, Mp2 = ia_t.shape
        assert Mp == Mp2
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        grid = (
            triton.cdiv(N, BLOCK_N),
            triton.cdiv(K, BLOCK_K),
            E,
        )

        _mxfp4_wgrad_kernel[grid](
            go_t, go_t.stride(0), go_t.stride(1),
            go_scale.view(torch.uint8),
            go_scale.stride(0), go_scale.stride(1),
            ia_t, ia_t.stride(0), ia_t.stride(1),
            ia_scale.view(torch.uint8),
            ia_scale.stride(0), ia_scale.stride(1),
            output, output.stride(0), output.stride(1), output.stride(2),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

    # ==================== Dense matmul (shared experts) ====================

    @triton.jit
    def _mxfp4_mm_kernel(
        A_ptr, A_stride_m, A_stride_kp,
        B_ptr, B_stride_kp, B_stride_n,
        A_scales_ptr, A_scales_stride_m, A_scales_stride_kb,
        B_scales_ptr, B_scales_stride_n, B_scales_stride_kb,
        C_ptr, C_stride_m, C_stride_n,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_base = pid_m * BLOCK_M
        n_base = pid_n * BLOCK_N

        m_offs = m_base + tl.arange(0, BLOCK_M)
        n_offs = n_base + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N

        BLOCK_KP: tl.constexpr = BLOCK_K // 2
        SUB_PER_BLOCK_K: tl.constexpr = BLOCK_K // SCALE_BLOCK
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_outer = K // BLOCK_K
        for k_outer in range(0, num_outer):
            kp_offs = k_outer * BLOCK_KP + tl.arange(0, BLOCK_KP)

            a = tl.load(
                A_ptr + m_offs[:, None] * A_stride_m + kp_offs[None, :] * A_stride_kp,
                mask=m_mask[:, None], other=0,
            )
            b = tl.load(
                B_ptr + kp_offs[:, None] * B_stride_kp + n_offs[None, :] * B_stride_n,
                mask=n_mask[None, :], other=0,
            )

            kb_offs = k_outer * SUB_PER_BLOCK_K + tl.arange(0, SUB_PER_BLOCK_K)
            a_scale = tl.load(
                A_scales_ptr + m_offs[:, None] * A_scales_stride_m + kb_offs[None, :] * A_scales_stride_kb,
                mask=m_mask[:, None], other=127,
            )
            b_scale = tl.load(
                B_scales_ptr + n_offs[:, None] * B_scales_stride_n + kb_offs[None, :] * B_scales_stride_kb,
                mask=n_mask[:, None], other=127,
            )

            acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc=acc, out_dtype=tl.float32)

        c_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            C_ptr + m_offs[:, None] * C_stride_m + n_offs[None, :] * C_stride_n,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp4_mm(
        a_fp4: torch.Tensor,    # (M, K//2) packed uint8
        b_fp4: torch.Tensor,    # (K//2, N) packed uint8
        a_scale: torch.Tensor,  # (M, K//32) e8m0
        b_scale: torch.Tensor,  # (N, K//32) e8m0
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """
        Dense MXFP4 matmul: C = A @ B (fp4 packed 2-per-byte).
        """
        M, Kp = a_fp4.shape
        Kp2, N = b_fp4.shape
        assert Kp == Kp2
        K = Kp * 2

        C = torch.empty((M, N), dtype=out_dtype, device=a_fp4.device)

        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        _mxfp4_mm_kernel[grid](
            a_fp4, a_fp4.stride(0), a_fp4.stride(1),
            b_fp4, b_fp4.stride(0), b_fp4.stride(1),
            a_scale.view(torch.uint8),
            a_scale.stride(0), a_scale.stride(1),
            b_scale.view(torch.uint8),
            b_scale.stride(0), b_scale.stride(1),
            C, C.stride(0), C.stride(1),
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SCALE_BLOCK=32,
            num_warps=num_warps, num_stages=num_stages,
        )
        return C

else:
    def triton_mxfp4_grouped_mm(*args, **kwargs):
        raise NotImplementedError(
            "ROCm MXFP4 kernels require gfx950 (MI355X) or later"
        )

    def triton_mxfp4_wgrad(*args, **kwargs):
        raise NotImplementedError(
            "ROCm MXFP4 kernels require gfx950 (MI355X) or later"
        )

    def triton_mxfp4_mm(*args, **kwargs):
        raise NotImplementedError(
            "ROCm MXFP4 kernels require gfx950 (MI355X) or later"
        )
