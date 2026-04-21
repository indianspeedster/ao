# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Triton MXFP8 grouped-GEMM kernels for ROCm gfx950+.

Use ``tl.dot_scaled`` to consume per-block e8m0 scales directly as a stand-in
for ``torch._scaled_grouped_mm``'s MXFP8 path until that ships on ROCm.

Contents:
  - ``triton_mxfp8_grouped_mm``: persistent grouped GEMM for MoE fwd / dgrad.
  - ``triton_mxfp8_wgrad``: weight-gradient grouped GEMM.
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_rocm_mxfp8_available = is_ROCM() and _triton_kernels_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    # ==================== Grouped GEMM (fwd / dgrad) ====================
    #
    # Non-persistent MoE GEMM adapted from AMD's aiter
    # (aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py), which in
    # turn derives from triton-lang's triton_kernels matmul_ogs.
    #
    # Scheduling uses:
    #   - XCD swizzle (ordered per-XCD launch for MI300/MI350 8-XCD parts),
    #   - Group_M pid reordering for L2 reuse,
    #   - Per-tile expert lookup via a packed (block_id << 16) | expt_id map.
    #
    # Requires:
    #   - ``weight`` column-major on the last two dims (``w.stride(-2) == 1``).
    #   - Scales are e8m0 viewed as uint8 with shape matching the data.
    #
    # The swiglu / bias / static-scale / gather-scatter plumbing from aiter is
    # dropped here; torchao's dispatcher handles those concerns upstream.

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
        x,
        BLOCK_N: tl.constexpr,
        MX_SCALE_BLOCK_K: tl.constexpr,
        N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
    ):
        x = x.reshape(
            BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1
        )
        x = x.permute(0, 5, 3, 1, 4, 2, 6)
        return x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)

    @triton.jit
    def _mxfp8_grouped_mm_kernel(
        Y, stride_y_m, stride_y_n,
        X, stride_x_m, stride_x_k,
        XMxScale, stride_x_mx_m, stride_x_mx_k,
        W, stride_w_e, stride_w_k, stride_w_n,
        WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
        N, K,
        ExptHist,       # (E,) int32 — tokens per expert
        ExptOffs,       # (E,) int32 — start offset per expert (expert-sorted)
        ExptOffsSum,    # 0-d int32 — total tile blocks (sum of cdiv(hist[e], BLOCK_M))
        ExptData,       # (grid_m_max,) int32 — packed (block_id<<16)|expt_id, -1 pad
        grid_m, grid_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        XCD_SWIZZLE: tl.constexpr,
        # "CDNA4_SCALE" enables pre-shuffled MX scale layout; None = plain.
        SWIZZLE_MX_SCALE: tl.constexpr,
        EVEN_K: tl.constexpr,
        MASK_K_LIMIT: tl.constexpr,
        W_CACHE_MODIFIER: tl.constexpr,
        UPCAST_INDICES: tl.constexpr = False,
    ):
        MX_PACK_DIVISOR: tl.constexpr = 32

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

        # X pointers (A, per-expert slice)
        offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
        offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
        X += start_m * stride_x_m
        offs_x_k = tl.arange(0, BLOCK_K)
        XPtrs = (
            X
            + offs_x_m.to(index_type)[:, None] * stride_x_m
            + offs_x_k.to(index_type)[None, :] * stride_x_k
        )

        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

        # W scale pointers
        WMxScale += expt_id * stride_w_mx_e
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
        else:
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
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

        # W pointers
        offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
        offs_w_k = tl.arange(0, BLOCK_K)
        W += expt_id * stride_w_e
        WPtrs = W + (
            offs_w_k.to(index_type)[:, None] * stride_w_k
            + offs_w_n.to(index_type)[None, :] * stride_w_n
        )

        # X scale pointers
        XMxScale += start_m * stride_x_mx_m
        offs_x_k_scale = tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = (
            XMxScale
            + offs_x_m.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

        num_k_iter = tl.cdiv(K, BLOCK_K)
        if not EVEN_K:
            num_k_iter -= 1

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(num_k_iter):
            x = tl.load(XPtrs)
            w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
            x_scales = tl.load(XMxScalePtrs)
            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                w_scales = _unswizzle_mx_scale_cdna4(
                    tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                    BLOCK_N, MX_SCALE_BLOCK_K,
                )
            else:
                w_scales = tl.load(WMxScalePtrs)

            acc = tl.dot_scaled(
                x, x_scales, "e4m3", w, w_scales, "e4m3", acc=acc, fast_math=True
            )

            WMxScalePtrs += PACKED_MX_BLOCK * stride_w_mx_k
            XMxScalePtrs += MX_SCALE_BLOCK_K * stride_x_mx_k
            XPtrs += BLOCK_K * stride_x_k
            WPtrs += BLOCK_K * stride_w_k

        if not EVEN_K:
            mask_x_k = offs_x_k < MASK_K_LIMIT
            mask_w_k = offs_w_k < MASK_K_LIMIT
            if SWIZZLE_MX_SCALE is None:
                mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT
            mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

            x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
            w = tl.load(WPtrs, mask=mask_w_k[:, None], other=0.0,
                        cache_modifier=W_CACHE_MODIFIER)
            x_scales = tl.load(XMxScalePtrs, mask=mask_x_k_scale[None, :])
            if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                w_scales = _unswizzle_mx_scale_cdna4(
                    tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                    BLOCK_N, MX_SCALE_BLOCK_K,
                )
            else:
                w_scales = tl.load(WMxScalePtrs, mask=mask_w_k_scale[None, :])

            acc = tl.dot_scaled(
                x, x_scales, "e4m3", w, w_scales, "e4m3", acc=acc, fast_math=True
            )

        # Write-back
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
        OffsetsPtr,             # (E,) int32 — group_end_offsets
        HistPtr,                # (E,) int32 — output: tokens per expert
        OffsRawPtr,             # (E,) int32 — output: prefix start offset per expert
        OffsPadSumPtr,          # 0-d int32 — output: sum of cdiv(hist[e], BLOCK_M)
        BlockPidMapPtr,         # (GRID_M_UB,) int32 — output: packed (block<<16)|e or -1
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GRID_M_UB: tl.constexpr,
    ):
        """One-launch build of all routing tensors the MM kernel needs.

        Grid is ``(GRID_M_UB,)``. Each program:
          - recomputes cum_blocks[] in registers from OffsetsPtr (O(E) loads),
          - writes ``HistPtr[pid]`` and ``OffsRawPtr[pid]`` when ``pid < E``,
          - writes ``BlockPidMapPtr[pid]`` for its global block index,
          - program 0 also writes ``OffsPadSumPtr`` (total block count).

        Replaces a chain of cat/diff/cumsum/arange/searchsorted/clamp/shift/where
        (~30-40 us of host-visible launch overhead) with a single ~2-3 us launch.
        """
        pid = tl.program_id(0)

        # Walk the E offsets in lockstep on every program. This is cheap:
        # with E<=8 and a coalesced load per step, it's ~E SLC loads.
        offs_prev = tl.zeros((), dtype=tl.int32)
        cum = tl.zeros((), dtype=tl.int32)
        target_e = tl.zeros((), dtype=tl.int32)
        target_block = tl.zeros((), dtype=tl.int32)
        valid = tl.full((), 0, dtype=tl.int1)

        for e in tl.static_range(E):
            off = tl.load(OffsetsPtr + e).to(tl.int32)
            h = off - offs_prev
            # First E programs also persist hist[e] and offs_raw[e].
            if pid == e:
                tl.store(HistPtr + e, h)
                tl.store(OffsRawPtr + e, offs_prev)
            blocks_e = (h + BLOCK_M - 1) // BLOCK_M
            cum_next = cum + blocks_e
            # Does this pid belong to expert e?
            owned = (pid >= cum) & (pid < cum_next)
            target_e = tl.where(owned, e, target_e)
            target_block = tl.where(owned, pid - cum, target_block)
            valid = valid | owned
            cum = cum_next
            offs_prev = off

        # Write block_pid_map[pid] and (once) offs_pad_sum.
        value = tl.where(valid, (target_block << 16) | target_e, tl.full((), -1, dtype=tl.int32))
        tl.store(BlockPidMapPtr + pid, value)
        if pid == 0:
            tl.store(OffsPadSumPtr, cum)

    def _build_expt_data(
        group_end_offsets: torch.Tensor, M: int, E: int, block_m: int
    ):
        """Single-kernel build of (hist, offs_raw, offs_pad_sum, block_pid_map,
        grid_m_ub) from ``group_end_offsets``. Sync-free, torch.compile-clean.

        Allocates one combined int32 buffer for all four outputs so we only
        pay one ``torch.empty`` Python call plus one kernel launch.
        """
        device = group_end_offsets.device
        grid_m_ub = triton.cdiv(M, block_m) + max(E - 1, 0)

        # Single backing allocation: [hist (E) | offs_raw (E) | pad_sum (1) | bpm (GRID_M_UB)]
        total = 2 * E + 1 + grid_m_ub
        buf = torch.empty(total, dtype=torch.int32, device=device)
        hist = buf[:E]
        offs_raw = buf[E:2 * E]
        offs_pad_sum = buf[2 * E]                      # 0-d view
        block_pid_map = buf[2 * E + 1:]

        offs_i32 = group_end_offsets if group_end_offsets.dtype == torch.int32 \
            else group_end_offsets.to(torch.int32)

        _expt_data_kernel[(grid_m_ub,)](
            offs_i32,
            hist, offs_raw, offs_pad_sum, block_pid_map,
            E=E, BLOCK_M=block_m, GRID_M_UB=grid_m_ub,
            num_warps=1,
        )
        return hist, offs_raw, offs_pad_sum, block_pid_map, grid_m_ub

    def _pick_block_nk(N: int, K: int) -> tuple:
        """Shape-aware (BLOCK_N, BLOCK_K) pick, derived from an 864-run
        per-shape sweep on gfx950/MI355X across (E, M=16640, N, K) with
        N,K in {2048, 5120, 8192}.

        - (N=2048, K=2048): 128/128 (best by 7.6-8.7% over 256/256 in sweep;
          +0.02 to +0.15x speedup-vs-bf16 in full bench, consistent wins)
        - else: 256/256

        The ``(K=2048, N>=5120) -> 256/128`` arm was dropped: the sweep
        showed only 1-5% gains there, which got swallowed by the >=5-15%
        run-to-run noise of the 36-shape bench (bf16 baseline itself swings
        that much between runs on our CI runner).
        """
        if N <= 2048 and K <= 2048:
            return 128, 128
        return 256, 256

    def triton_mxfp8_grouped_mm(
        input_act: torch.Tensor,
        weight: torch.Tensor,
        input_act_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_M: int = 128,
        BLOCK_N: int = None,   # None = shape-aware default
        BLOCK_K: int = None,   # None = shape-aware default
        GROUP_M: int = 8,
        XCD_SWIZZLE: int = 8,
        num_warps: int = 8,
        num_stages: int = 2,
        matrix_instr_nonkdim: int = 32,
    ) -> torch.Tensor:
        """MXFP8 grouped GEMM: ``output[g] = input_act[group_g] @ weight[g]^T``.

        Args:
            input_act: ``(M, K)`` fp8, row-major.
            weight: ``(E, N, K)`` fp8 row-major; internally viewed as column-major
                (``E, K, N``) before launch (``stride(-2)`` must become 1).
            input_act_scales: ``(M, K//32)`` e8m0-viewed-as-uint8.
            weight_scales: ``(E, N, K//32)`` e8m0-viewed-as-uint8.
            group_end_offsets: ``(E,)`` int32, cumulative token counts per expert.
        """
        M, K = input_act.shape
        E, N, K2 = weight.shape
        assert K == K2, f"K mismatch: A={K}, B={K2}"

        # Shape-aware default tile size: see _pick_block_nk docstring.
        if BLOCK_N is None or BLOCK_K is None:
            _bn, _bk = _pick_block_nk(N, K)
            if BLOCK_N is None:
                BLOCK_N = _bn
            if BLOCK_K is None:
                BLOCK_K = _bk

        # Column-major view of W: (E, N, K) row-major -> (E, K, N) with stride(-2)==1.
        # kernel expects W shape (E, K, N) and its W_scales shape (E, K//32, N).
        w_kn = weight.permute(0, 2, 1)
        w_scales_kn = weight_scales.view(torch.uint8).permute(0, 2, 1)
        x_scales_u8 = input_act_scales.view(torch.uint8)

        hist, offs_raw, offs_pad_sum, block_pid_map, grid_m = _build_expt_data(
            group_end_offsets, M, E, BLOCK_M
        )
        grid_n = triton.cdiv(N, BLOCK_N)
        grid = (grid_m * grid_n,)

        output = torch.empty((M, N), dtype=out_dtype, device=input_act.device)

        _mxfp8_grouped_mm_kernel[grid](
            output, output.stride(0), output.stride(1),
            input_act, input_act.stride(0), input_act.stride(1),
            x_scales_u8, x_scales_u8.stride(0), x_scales_u8.stride(1),
            w_kn, w_kn.stride(0), w_kn.stride(1), w_kn.stride(2),
            w_scales_kn, w_scales_kn.stride(0), w_scales_kn.stride(1), w_scales_kn.stride(2),
            N, K,
            hist, offs_raw, offs_pad_sum, block_pid_map,
            grid_m, grid_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, XCD_SWIZZLE=XCD_SWIZZLE,
            SWIZZLE_MX_SCALE=None,
            EVEN_K=(K % BLOCK_K == 0), MASK_K_LIMIT=(K % BLOCK_K),
            W_CACHE_MODIFIER=None,
            UPCAST_INDICES=False,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim, kpack=1,
            waves_per_eu=0,
        )
        return output

    # ==================== Weight-gradient grouped GEMM ====================

    @triton.jit
    def _mxfp8_wgrad_direct_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
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

        SUB_PER_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m_iter in range(0, tl.cdiv(M_g, BLOCK_M)):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, SUB_PER_BLOCK_M)
            mb_mask = mb_offs < M_SCALES
            # other=127: e8m0 bias 127 = 2^0 = 1.0 (neutral).
            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None] & mb_mask[None, :], other=127,
            )
            ia_scale = tl.load(
                IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask[:, None] & mb_mask[None, :], other=127,
            )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32,
            )

        c_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    # Partial-sum wgrad: partitions the per-group M-loop across SPLIT_M CTAs
    # and writes fp32 partials to an (E, SPLIT_M, N, K) buffer; a reduce
    # kernel sums across SPLIT_M into the (E, N, K) bf16 output. Worth it
    # when a single group leaves too few CTAs to saturate the device.

    @triton.jit
    def _mxfp8_wgrad_partial_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        P_ptr, P_stride_e, P_stride_s, P_stride_n, P_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SPLIT_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_eg = tl.program_id(2)
        pid_split = pid_eg % SPLIT_M
        pid_g = pid_eg // SPLIT_M

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

        SUB_PER_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        M_SCALES = M // SCALE_BLOCK
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        num_m_iters = tl.cdiv(M_g, BLOCK_M)
        for m_iter in range(pid_split, num_m_iters, SPLIT_M):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, SUB_PER_BLOCK_M)
            mb_mask = mb_offs < M_SCALES
            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None] & mb_mask[None, :], other=127,
            )
            ia_scale = tl.load(
                IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask[:, None] & mb_mask[None, :], other=127,
            )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32,
            )

        p_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            P_ptr + pid_g * P_stride_e + pid_split * P_stride_s
                  + n_offs[:, None] * P_stride_n + k_offs[None, :] * P_stride_k,
            acc, mask=p_mask,
        )

    @triton.jit
    def _mxfp8_wgrad_reduce_kernel(
        P_ptr, P_stride_e, P_stride_s, P_stride_n, P_stride_k,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K
        c_mask = n_mask[:, None] & k_mask[None, :]

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for s in range(SPLIT_M):
            acc += tl.load(
                P_ptr + pid_g * P_stride_e + s * P_stride_s
                      + n_offs[:, None] * P_stride_n + k_offs[None, :] * P_stride_k,
                mask=c_mask, other=0.0,
            )

        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp8_wgrad(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_N: int = 256,
        BLOCK_K: int = 256,
        BLOCK_M: int = 64,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """MXFP8 weight gradient: ``grad_W[g] = grad_output[group_g]^T @ input_act[group_g]``.

        Both inputs must be dim1-quantized (scales along the M / token dim).
        For a single group (``E == 1``) the (N/BN, K/BK) grid may not saturate
        the device, so we partition the per-group M-loop across SPLIT_M CTAs
        and reduce their fp32 partials in a second pass; for E >= 2 the natural
        grid is enough and we write bf16 directly.

        Args:
            go_t: ``(N, M)`` fp8.
            ia_t: ``(K, M)`` fp8.

        Returns:
            ``(E, N, K)`` bf16.
        """
        N, M = go_t.shape
        K, _ = ia_t.shape
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32
        SPLIT_M = 2 if E == 1 else 1

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        if SPLIT_M == 1:
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
            _mxfp8_wgrad_direct_kernel[grid](
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
                matrix_instr_nonkdim=0, kpack=1,
            )
            return output

        partials = torch.empty(
            (E, SPLIT_M, N, K), dtype=torch.float32, device=go_t.device
        )
        partial_grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E * SPLIT_M)
        _mxfp8_wgrad_partial_kernel[partial_grid](
            go_t, go_t.stride(0), go_t.stride(1),
            go_scale.view(torch.uint8),
            go_scale.stride(0), go_scale.stride(1),
            ia_t, ia_t.stride(0), ia_t.stride(1),
            ia_scale.view(torch.uint8),
            ia_scale.stride(0), ia_scale.stride(1),
            partials,
            partials.stride(0), partials.stride(1),
            partials.stride(2), partials.stride(3),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SPLIT_M=SPLIT_M, SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=0, kpack=1,
        )

        reduce_grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
        _mxfp8_wgrad_reduce_kernel[reduce_grid](
            partials,
            partials.stride(0), partials.stride(1),
            partials.stride(2), partials.stride(3),
            output,
            output.stride(0), output.stride(1), output.stride(2),
            N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_M=SPLIT_M,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

else:
    _UNAVAILABLE_MSG = "ROCm MXFP8 kernels require gfx950 or later"

    def triton_mxfp8_grouped_mm(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)

    def triton_mxfp8_wgrad(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)
