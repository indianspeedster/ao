"""
Grouped MXFP8 GEMM v6: same kernel body as v3 but autotuned over tile sizes.

Findings from rocm_mm_v4b on the dense path:
  - num_warps=4 (not 8) wins: more registers per warp → better MFMA back-to-back.
  - BLOCK_K=128 or 256 wins (v3 default 128 is OK, 256 sometimes better).
  - Larger BLOCK_M/N (256) wins on big shapes; smaller on small shapes.
  - Persistent loop / super-grouping does NOT help on AMD Triton.

We keep v3's per-program-per-(g, m, n) launch pattern and only sweep tiles.
"""
import torch
from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

if is_ROCM() and _triton_kernels_available:
    import triton
    import triton.language as tl

    _CFGS = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=nw, num_stages=ns,
        )
        for bm in (64, 128, 256)
        for bn in (64, 128, 256)
        for bk in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3)
        if bm * bn <= 256 * 256
    ]

    @triton.autotune(configs=_CFGS, key=["M", "N", "K", "E"])
    @triton.jit
    def _kernel_v6(
        A_ptr, A_stride_m, A_stride_k,
        B_ptr, B_stride_e, B_stride_n, B_stride_k,
        A_scales_ptr, A_scales_stride_m, A_scales_stride_kb,
        B_scales_ptr, B_scales_stride_e, B_scales_stride_n, B_scales_stride_kb,
        C_ptr, C_stride_m, C_stride_n,
        group_end_offsets_ptr,
        M, N, K, E,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
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
        # Bound by group_end AND global M (defensive against bad offsets).
        m_mask = (m_offs < group_end) & (m_offs < M)
        n_mask = n_offs < N

        SUB: tl.constexpr = BLOCK_K // SCALE_BLOCK
        K_SCALES = K // SCALE_BLOCK     # MXFP8 invariant: K % 32 == 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # tl.cdiv handles K-tail; k_mask + kb_mask zero the partial tile.
        for k_outer in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_outer * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            a = tl.load(A_ptr + m_offs[:, None] * A_stride_m + k_offs[None, :] * A_stride_k,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(B_ptr + pid_g * B_stride_e + n_offs[None, :] * B_stride_n + k_offs[:, None] * B_stride_k,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            kb_offs = k_outer * SUB + tl.arange(0, SUB)
            kb_mask = kb_offs < K_SCALES
            # other=127: e8m0 bias-encoded 2^0 = 1.0 (neutral). Combined with
            # data masked to 0.0, tail contribution to dot_scaled is 0.
            a_s = tl.load(A_scales_ptr + m_offs[:, None] * A_scales_stride_m + kb_offs[None, :] * A_scales_stride_kb,
                          mask=m_mask[:, None] & kb_mask[None, :], other=127)
            b_s = tl.load(B_scales_ptr + pid_g * B_scales_stride_e + n_offs[:, None] * B_scales_stride_n + kb_offs[None, :] * B_scales_stride_kb,
                          mask=n_mask[:, None] & kb_mask[None, :], other=127)
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, out_dtype=tl.float32)

        tl.store(C_ptr + m_offs[:, None] * C_stride_m + n_offs[None, :] * C_stride_n,
                 acc.to(tl.bfloat16),
                 mask=m_mask[:, None] & n_mask[None, :])

    def triton_mxfp8_grouped_mm_v6(
        input_act: torch.Tensor,    # (M, K) fp8
        weight: torch.Tensor,       # (E, N, K) fp8
        input_act_scales: torch.Tensor,  # (M, K//32) e8m0
        weight_scales: torch.Tensor,     # (E, N, K//32) e8m0
        group_end_offsets: torch.Tensor, # (E,) int32 cumulative
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        assert input_act.dtype == torch.float8_e4m3fn
        assert weight.dtype == torch.float8_e4m3fn
        M, K = input_act.shape
        E, N, K2 = weight.shape
        assert K == K2 and K % 32 == 0
        C = torch.empty((M, N), dtype=out_dtype, device=input_act.device)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),    # over-launch; early-return guards
            triton.cdiv(N, META["BLOCK_N"]),
            E,
        )
        _kernel_v6[grid](
            input_act, input_act.stride(0), input_act.stride(1),
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            input_act_scales.view(torch.uint8),
            input_act_scales.stride(0), input_act_scales.stride(1),
            weight_scales.view(torch.uint8),
            weight_scales.stride(0), weight_scales.stride(1), weight_scales.stride(2),
            C, C.stride(0), C.stride(1),
            group_end_offsets,
            M, N, K, E,
            SCALE_BLOCK=32,
        )
        return C
else:
    def triton_mxfp8_grouped_mm_v6(*a, **kw):
        raise NotImplementedError
