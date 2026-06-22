# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Optimized forward/dgrad MXFP4 grouped-GEMM for ROCm gfx950 (MI355X).

Ports the scheduling + scale-handling optimizations from the tuned ROCm MXFP8
grouped-GEMM (forward + dgrad both compute A @ B^T per expert group):

  - histogram-built exact-tile 1D grid (``_build_expt_data``): launch exactly
    sum_e cdiv(hist[e], BLOCK_M) * grid_n tiles instead of E * cdiv(M, BLOCK_M)
    * grid_n with most blocks early-returning.
  - XCD swizzle (8-XCD L2 locality) + GROUP_M pid reordering.
  - CDNA4-native pre-shuffled e8m0 scale layout: the MX scale block layout is
    identical for fp8 and fp4 (both feed v_mfma_scale_f32_16x16x128_f8f6f4), so
    the host-side shuffle + in-kernel unshuffle are reused verbatim. Removes the
    ds_read_u8 + v_perm chain the plain lowering emits for MFMA scale loads.

Only the data path is fp4-specific: A is (M, K//2) and W is (E, N, K//2) packed
2-per-byte (e2m1), BLOCK_K counts fp4 elements (BLOCK_K//2 bytes loaded), and
tl.dot_scaled uses the "e2m1" format.
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_available = is_ROCM() and _triton_kernels_available

if _available:
    import triton
    import triton.language as tl

    @triton.jit
    def _xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
        pids_per_group = domain_size // XCD_SWIZZLE
        extra_pid_groups = domain_size % XCD_SWIZZLE
        group = pid % XCD_SWIZZLE
        local_pid = pid // XCD_SWIZZLE
        return group * pids_per_group + min(group, extra_pid_groups) + local_pid

    @triton.jit
    def _pid_grid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr = 1):
        if GROUP_M == 1:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n
        else:
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            tl.assume(group_size_m >= 0)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m
        return pid_m, pid_n

    @triton.jit
    def _unswizzle_mx_scale_cdna4(
        x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
        N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
    ):
        """Inverse of host-side shuffle for nonkdim=16 MFMA (e8m0 scales)."""
        x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
        x = x.permute(0, 5, 3, 1, 4, 2, 6)
        return x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)

    @triton.jit
    def _unswizzle_mx_scale_cdna4_nonkdim32(
        x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
        N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
    ):
        """Inverse of host-side shuffle for nonkdim=32 MFMA (e8m0 scales)."""
        x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 2, 32, 4, 1)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)

    @triton.jit
    def _mxfp4_grouped_mm_opt_kernel(
        Y, stride_y_m, stride_y_n,
        X, stride_x_m, stride_x_kp,            # X (M, K//2) packed fp4
        XMxScale, stride_x_mx_m, stride_x_mx_k,
        W, stride_w_e, stride_w_n, stride_w_kp,  # W (E, N, K//2) packed fp4
        WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
        N, K,                                  # K in fp4 elements
        ExptHist, ExptOffs, ExptOffsSum, ExptData,
        grid_m, grid_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,                 # fp4 elements
        GROUP_M: tl.constexpr,
        XCD_SWIZZLE: tl.constexpr,
        SWIZZLE_MX_SCALE: tl.constexpr,
        SCALE_NONKDIM: tl.constexpr,
        UPCAST_INDICES: tl.constexpr = False,
    ):
        MX_PACK_DIVISOR: tl.constexpr = 32
        BLOCK_KP: tl.constexpr = BLOCK_K // 2  # packed bytes per K tile

        pid = tl.program_id(0)
        if ExptOffsSum is not None and XCD_SWIZZLE > 1:
            padding_m = grid_m - tl.load(ExptOffsSum)
        else:
            padding_m: tl.constexpr = 0

        index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32
        unpadded_m = grid_m - padding_m
        tl.assume(unpadded_m >= 0)
        total_actual_tiles = unpadded_m * grid_n
        if padding_m > 0 and pid >= total_actual_tiles:
            return

        pid_emn = pid
        if XCD_SWIZZLE != 1:
            pid_emn = _xcd_swizzle(pid_emn, total_actual_tiles, XCD_SWIZZLE)
        pid_m, pid_n = _pid_grid(pid_emn, unpadded_m, grid_n, GROUP_M)

        expt_data = tl.load(ExptData + pid_m)
        if expt_data == -1:
            return
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        M = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        expt_id = expt_id.to(index_type)
        block_id = block_id.to(index_type)
        start_m = start_m.to(index_type)
        pid_n = pid_n.to(index_type)

        # ---- X (activation) data pointers, packed fp4 ----
        offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
        offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
        X += start_m * stride_x_m
        offs_x_kp = tl.arange(0, BLOCK_KP)
        XPtrs = (
            X
            + offs_x_m.to(index_type)[:, None] * stride_x_m
            + offs_x_kp.to(index_type)[None, :] * stride_x_kp
        )

        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

        # ---- W scale pointers ----
        WMxScale += expt_id * stride_w_mx_e
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE
        else:
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M
        offs_w_n_scale = (pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)) % N
        offs_w_n_scale = tl.max_contiguous(
            tl.multiple_of(offs_w_n_scale, SCALE_BLOCK_N), SCALE_BLOCK_N
        )
        offs_w_k_scale = tl.arange(0, PACKED_MX_BLOCK)
        WMxScalePtrs = (
            WMxScale
            + offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k
            + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n
        )

        # ---- W (weight) data pointers, packed fp4 ----
        offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
        offs_w_kp = tl.arange(0, BLOCK_KP)
        W += expt_id * stride_w_e
        WPtrs = W + (
            offs_w_kp.to(index_type)[:, None] * stride_w_kp
            + offs_w_n.to(index_type)[None, :] * stride_w_n
        )

        # ---- X scale pointers ----
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            XMxScale += (start_m // 32) * stride_x_mx_m
            offs_x_m_scale = BLOCK_M // NON_K_PRESHUFFLE_BLOCK_SIZE * block_id + tl.arange(0, SCALE_BLOCK_M)
            offs_x_k_scale = tl.arange(0, PACKED_MX_BLOCK)
        else:
            XMxScale += start_m * stride_x_mx_m
            offs_x_m_scale = offs_x_m
            offs_x_k_scale = tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = (
            XMxScale
            + offs_x_m_scale.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

        num_k_iter = K // BLOCK_K  # EVEN_K enforced by launcher
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(num_k_iter):
            x = tl.load(XPtrs)
            w = tl.load(WPtrs)
            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                if SCALE_NONKDIM == 32:
                    x_scales = _unswizzle_mx_scale_cdna4_nonkdim32(
                        tl.load(XMxScalePtrs), BLOCK_M, MX_SCALE_BLOCK_K)
                    w_scales = _unswizzle_mx_scale_cdna4_nonkdim32(
                        tl.load(WMxScalePtrs), BLOCK_N, MX_SCALE_BLOCK_K)
                else:
                    x_scales = _unswizzle_mx_scale_cdna4(
                        tl.load(XMxScalePtrs), BLOCK_M, MX_SCALE_BLOCK_K)
                    w_scales = _unswizzle_mx_scale_cdna4(
                        tl.load(WMxScalePtrs), BLOCK_N, MX_SCALE_BLOCK_K)
            else:
                x_scales = tl.load(XMxScalePtrs)
                w_scales = tl.load(WMxScalePtrs)

            acc = tl.dot_scaled(
                x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc, fast_math=True
            )

            WMxScalePtrs += PACKED_MX_BLOCK * stride_w_mx_k
            XMxScalePtrs += PACKED_MX_BLOCK * stride_x_mx_k
            XPtrs += BLOCK_KP * stride_x_kp
            WPtrs += BLOCK_KP * stride_w_kp

        # ---- write-back ----
        offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
        offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_y_n < N
        Y += start_m * stride_y_m
        YPtrs = (
            Y
            + offs_m.to(index_type)[:, None] * stride_y_m
            + offs_y_n.to(index_type)[None, :] * stride_y_n
        )
        tl.store(YPtrs, acc.to(Y.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _expt_data_kernel(
        OffsetsPtr, HistPtr, OffsRawPtr, OffsPadSumPtr, BlockPidMapPtr,
        E: tl.constexpr, BLOCK_M: tl.constexpr, GRID_M_UB: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_prev = tl.zeros((), dtype=tl.int32)
        cum = tl.zeros((), dtype=tl.int32)
        target_e = tl.zeros((), dtype=tl.int32)
        target_block = tl.zeros((), dtype=tl.int32)
        valid = tl.full((), 0, dtype=tl.int1)
        for e in tl.static_range(E):
            off = tl.load(OffsetsPtr + e).to(tl.int32)
            h = off - offs_prev
            if pid == e:
                tl.store(HistPtr + e, h)
                tl.store(OffsRawPtr + e, offs_prev)
            blocks_e = (h + BLOCK_M - 1) // BLOCK_M
            cum_next = cum + blocks_e
            owned = (pid >= cum) & (pid < cum_next)
            target_e = tl.where(owned, e, target_e)
            target_block = tl.where(owned, pid - cum, target_block)
            valid = valid | owned
            cum = cum_next
            offs_prev = off
        value = tl.where(valid, (target_block << 16) | target_e, tl.full((), -1, dtype=tl.int32))
        tl.store(BlockPidMapPtr + pid, value)
        if pid == 0:
            tl.store(OffsPadSumPtr, cum)

    def _build_expt_data(group_end_offsets, M, E, block_m):
        device = group_end_offsets.device
        grid_m_ub = triton.cdiv(M, block_m) + max(E - 1, 0)
        total = 2 * E + 1 + grid_m_ub
        buf = torch.empty(total, dtype=torch.int32, device=device)
        hist = buf[:E]
        offs_raw = buf[E:2 * E]
        offs_pad_sum = buf[2 * E]
        block_pid_map = buf[2 * E + 1:]
        offs_i32 = group_end_offsets if group_end_offsets.dtype == torch.int32 \
            else group_end_offsets.to(torch.int32)
        _expt_data_kernel[(grid_m_ub,)](
            offs_i32, hist, offs_raw, offs_pad_sum, block_pid_map,
            E=E, BLOCK_M=block_m, GRID_M_UB=grid_m_ub, num_warps=1,
        )
        return hist, offs_raw, offs_pad_sum, block_pid_map, grid_m_ub

    # ---- host-side e8m0 scale shuffles (identical to MXFP8; dtype-independent) ----
    def _shuffle_w_scales_cdna4_nonkdim16(w_scales):
        E, N, Ks = w_scales.shape
        x = w_scales.reshape(E, N // 32, 2, 16, Ks // 8, 2, 4, 1)
        x = x.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
        return x.reshape(E, N // 32, Ks * 32)

    def _shuffle_x_scales_cdna4_nonkdim16(x_scales):
        M, Ks = x_scales.shape
        x = x_scales.reshape(M // 32, 2, 16, Ks // 8, 2, 4, 1)
        x = x.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
        return x.reshape(M // 32, Ks * 32)

    def _shuffle_w_scales_cdna4_nonkdim32(w_scales):
        E, N, Ks = w_scales.shape
        x = w_scales.reshape(E, N // 32, 32, Ks // 8, 4, 2, 1)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        return x.reshape(E, N // 32, Ks * 32)

    def _shuffle_x_scales_cdna4_nonkdim32(x_scales):
        M, Ks = x_scales.shape
        x = x_scales.reshape(M // 32, 32, Ks // 8, 4, 2, 1)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(M // 32, Ks * 32)

    # Per-shape best configs from an 8-GPU parallel sweep on MI355X
    # (scripts/sweep_mxfp4_driver.py; search space BLOCK_M{128,256} x
    # BLOCK_N{128,256} x BLOCK_K{128,256} x GROUP_M{4,8,16} x num_warps{4,8} x
    # nonkdim{16,32} x waves{0,2}). Keyed (E, N, K); Llama4 shapes (M=16640).
    # nonkdim=32 + BLOCK_K=256 (CDNA4 scale path) win almost universally.
    # Comments show the swept speedup vs the tuned MXFP8 kernel.
    _BEST_CFGS = {
        (1, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.18x
        (1, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.57x
        (1, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=16, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 1.65x
        (1, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.29x
        (1, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.59x
        (1, 5120, 8192): dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 1.66x
        (1, 8192, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.36x
        (1, 8192, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.59x
        (1, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 1.65x
        (2, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.18x
        (2, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.55x
        (2, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.69x
        (2, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.32x
        (2, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.58x
        (2, 5120, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.65x
        (2, 8192, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.38x
        (2, 8192, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.56x
        (2, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 1.61x
        (4, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.20x
        (4, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.58x
        (4, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.66x
        (4, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.34x
        (4, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.59x
        (4, 5120, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.59x
        (4, 8192, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.38x
        (4, 8192, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.57x
        (4, 8192, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.58x
        (8, 2048, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.20x
        (8, 2048, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=16, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.49x
        (8, 2048, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.63x
        (8, 5120, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.34x
        (8, 5120, 5120): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.52x
        (8, 5120, 8192): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.58x
        (8, 8192, 2048): dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.33x
        (8, 8192, 5120): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32),  # 1.49x
        (8, 8192, 8192): dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=256, GROUP_M=8, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32),  # 1.56x
    }

    # Fallback for unseen (E, N, K): nonkdim=32 + BLOCK_K=256 (CDNA4) is the
    # universal winner. BLOCK_M scales with token count M; large M (e.g. DSV3
    # EP shapes) prefers 256/256.
    def _pick_config(E, N, K, M):
        cfg = _BEST_CFGS.get((E, N, K))
        if cfg is not None:
            return cfg
        if M >= 65536:
            return dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, GROUP_M=4,
                        num_warps=4, num_stages=2, waves_per_eu=0,
                        matrix_instr_nonkdim=16 if M >= 100000 else 32)
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=256, GROUP_M=8,
                    num_warps=4, num_stages=2, waves_per_eu=0,
                    matrix_instr_nonkdim=32)

    def triton_mxfp4_grouped_mm_opt(
        input_act, weight, input_act_scales, weight_scales, group_end_offsets,
        out_dtype=torch.bfloat16,
        BLOCK_M=None, BLOCK_N=None, BLOCK_K=None, GROUP_M=None, XCD_SWIZZLE=None,
        num_warps=None, num_stages=None, matrix_instr_nonkdim=None,
        waves_per_eu=None, kpack=1,
    ):
        """Optimized MXFP4 grouped GEMM: output[g] = A[group_g] @ W[g]^T.

        Args:
            input_act: (M, K//2) uint8 packed fp4, row-major.
            weight:    (E, N, K//2) uint8 packed fp4.
            input_act_scales: (M, K//32) e8m0 (uint8).
            weight_scales:    (E, N, K//32) e8m0 (uint8).
            group_end_offsets: (E,) int32 cumulative token counts per expert.
        """
        M, Kp = input_act.shape
        K = Kp * 2
        E, N, Kp2 = weight.shape
        assert Kp == Kp2, f"K mismatch: A={Kp}, W={Kp2}"

        _cfg = _pick_config(E, N, K, M)
        BLOCK_M = BLOCK_M or _cfg["BLOCK_M"]
        BLOCK_N = BLOCK_N or _cfg["BLOCK_N"]
        BLOCK_K = BLOCK_K or _cfg["BLOCK_K"]
        GROUP_M = GROUP_M or _cfg["GROUP_M"]
        num_warps = num_warps or _cfg["num_warps"]
        num_stages = num_stages or _cfg["num_stages"]
        waves_per_eu = _cfg["waves_per_eu"] if waves_per_eu is None else waves_per_eu
        matrix_instr_nonkdim = matrix_instr_nonkdim or _cfg["matrix_instr_nonkdim"]
        XCD_SWIZZLE = 8 if XCD_SWIZZLE is None else XCD_SWIZZLE

        # BLOCK_K must divide the contraction dim K (in fp4 elements). The tuned
        # configs assume K % 256 == 0 (CDNA4); for other K (e.g. DeepSeek-16B
        # down-proj K=1408) fall back to the largest power-of-2 divisor, which
        # also drops to the plain (non-CDNA4) scale path below.
        if K % BLOCK_K != 0:
            for _cand in (256, 128, 64, 32):
                if K % _cand == 0:
                    BLOCK_K = _cand
                    break
            else:
                raise ValueError(f"K={K} not divisible by any supported BLOCK_K")

        x_scales_u8 = input_act_scales.view(torch.uint8)
        w_scales_u8 = weight_scales.view(torch.uint8)

        # CDNA4 scale path: same requirements as MXFP8 (scales are dtype-agnostic).
        # MXFP4_FORCE_PLAIN=1 disables it (diagnostic: isolate per-call shuffle cost).
        import os
        use_cdna4_scale = (
            BLOCK_K >= 256 and K % 256 == 0 and N % 32 == 0 and M % 32 == 0
            and os.environ.get("MXFP4_FORCE_PLAIN") != "1"
        )

        if use_cdna4_scale:
            if matrix_instr_nonkdim == 32:
                w_scales_shuf = _shuffle_w_scales_cdna4_nonkdim32(w_scales_u8)
                x_scales_shuf = _shuffle_x_scales_cdna4_nonkdim32(x_scales_u8)
                nonkdim = 32
            else:
                w_scales_shuf = _shuffle_w_scales_cdna4_nonkdim16(w_scales_u8)
                x_scales_shuf = _shuffle_x_scales_cdna4_nonkdim16(x_scales_u8)
                nonkdim = 16
            w_scales_arg = w_scales_shuf
            w_sc_e, w_sc_k, w_sc_n = (w_scales_shuf.stride(0), w_scales_shuf.stride(2),
                                      w_scales_shuf.stride(1))
            x_scales_arg = x_scales_shuf
            x_sc_m, x_sc_k = x_scales_shuf.stride(0), x_scales_shuf.stride(1)
            swizzle = "CDNA4_SCALE"
        else:
            # Plain path: W scales (E, K//32, N), X scales (M, K//32).
            w_scales_kn = w_scales_u8.permute(0, 2, 1)
            w_scales_arg = w_scales_kn
            w_sc_e, w_sc_k, w_sc_n = (w_scales_kn.stride(0), w_scales_kn.stride(1),
                                      w_scales_kn.stride(2))
            x_scales_arg = x_scales_u8
            x_sc_m, x_sc_k = x_scales_u8.stride(0), x_scales_u8.stride(1)
            swizzle = None
            nonkdim = matrix_instr_nonkdim

        hist, offs_raw, offs_pad_sum, block_pid_map, grid_m = _build_expt_data(
            group_end_offsets, M, E, BLOCK_M
        )
        grid_n = triton.cdiv(N, BLOCK_N)
        grid = (grid_m * grid_n,)

        output = torch.zeros((M, N), dtype=out_dtype, device=input_act.device)

        _mxfp4_grouped_mm_opt_kernel[grid](
            output, output.stride(0), output.stride(1),
            input_act, input_act.stride(0), input_act.stride(1),
            x_scales_arg, x_sc_m, x_sc_k,
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            w_scales_arg, w_sc_e, w_sc_k, w_sc_n,
            N, K,
            hist, offs_raw, offs_pad_sum, block_pid_map,
            grid_m, grid_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, XCD_SWIZZLE=XCD_SWIZZLE,
            SWIZZLE_MX_SCALE=swizzle, SCALE_NONKDIM=nonkdim,
            UPCAST_INDICES=False,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=nonkdim, kpack=kpack, waves_per_eu=waves_per_eu,
        )
        return output

else:
    def triton_mxfp4_grouped_mm_opt(*args, **kwargs):
        raise NotImplementedError("requires ROCm gfx950 + triton")
