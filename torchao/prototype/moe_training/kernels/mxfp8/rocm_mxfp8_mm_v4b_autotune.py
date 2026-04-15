"""v4b: drop persistent loop. Same body as v1 but autotune over tile sizes."""
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
        for ns in (2, 3, 4)
        if bm * bn <= 256 * 256
    ]

    @triton.autotune(configs=_CFGS, key=["M", "N", "K"])
    @triton.jit
    def _kernel(
        A_ptr, A_stride_m, A_stride_k,
        B_ptr, B_stride_k, B_stride_n,
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
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N
        SUB: tl.constexpr = BLOCK_K // SCALE_BLOCK
        K_SCALES = K // SCALE_BLOCK
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # tl.cdiv covers K-tail; k_mask + kb_mask zero the partial tile.
        for k_outer in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_outer * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            a = tl.load(A_ptr + m_offs[:, None] * A_stride_m + k_offs[None, :] * A_stride_k,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(B_ptr + k_offs[:, None] * B_stride_k + n_offs[None, :] * B_stride_n,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            kb_offs = k_outer * SUB + tl.arange(0, SUB)
            kb_mask = kb_offs < K_SCALES
            # e8m0 bias=127 → 2^0=1.0; data masked to 0.0 makes tail zero.
            a_s = tl.load(A_scales_ptr + m_offs[:, None] * A_scales_stride_m + kb_offs[None, :] * A_scales_stride_kb,
                          mask=m_mask[:, None] & kb_mask[None, :], other=127)
            b_s = tl.load(B_scales_ptr + n_offs[:, None] * B_scales_stride_n + kb_offs[None, :] * B_scales_stride_kb,
                          mask=n_mask[:, None] & kb_mask[None, :], other=127)
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, out_dtype=tl.float32)
        tl.store(C_ptr + m_offs[:, None] * C_stride_m + n_offs[None, :] * C_stride_n,
                 acc.to(tl.bfloat16),
                 mask=m_mask[:, None] & n_mask[None, :])

    def triton_mxfp8_mm_v4b(a_fp8, b_fp8, a_scale, b_scale, out_dtype=torch.bfloat16):
        M, K = a_fp8.shape
        K2, N = b_fp8.shape
        assert K == K2 and K % 32 == 0
        C = torch.empty((M, N), dtype=out_dtype, device=a_fp8.device)
        # Need to pass full grid that covers ALL tile-size choices the autotuner may pick.
        # Use a worst-case grid for the smallest BLOCK_M/N combo (64).
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        _kernel[grid](
            a_fp8, a_fp8.stride(0), a_fp8.stride(1),
            b_fp8, b_fp8.stride(0), b_fp8.stride(1),
            a_scale.view(torch.uint8), a_scale.stride(0), a_scale.stride(1),
            b_scale.view(torch.uint8), b_scale.stride(0), b_scale.stride(1),
            C, C.stride(0), C.stride(1),
            M, N, K,
            SCALE_BLOCK=32,
        )
        return C
else:
    def triton_mxfp8_mm_v4b(*a, **kw):
        raise NotImplementedError
