"""
Flash Attention v2 Implementation in Triton
Based on: https://arxiv.org/abs/2307.08691
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
    ],
    key=['HEAD_DIM', 'IS_CAUSAL'],
)
@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V,
    O,
    L,
    M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z,
    H,
    H_KV,
    N_CTX,
    HEAD_DIM,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    STORE_STATS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    off_h_kv = off_h // (H // H_KV)

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h_kv * stride_kh
    v_offset = off_z * stride_vz + off_h_kv * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    lo = 0
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k_mask = ((start_n + offs_n[:, None]) < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale

        if IS_CAUSAL:
            is_diagonal_block = start_n + BLOCK_N > start_m * BLOCK_M
            if is_diagonal_block:
                causal_mask = (start_n + offs_n[None, :]) <= offs_m[:, None]
                qk = tl.where(causal_mask, qk, float("-inf"))
        else:
            is_last_block = start_n + BLOCK_N > N_CTX
            if is_last_block:
                pad_mask = (start_n + offs_n[None, :]) < N_CTX
                qk = tl.where(pad_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)

        m_i_new = tl.maximum(m_i, m_ij)

        p = tl.exp(qk - m_i_new[:, None])

        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v_mask = ((start_n + offs_n[:, None]) < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = acc * alpha[:, None]

        acc = tl.dot(p.to(v.dtype), v, acc, out_dtype=tl.float32)

        m_i = m_i_new

    acc = acc / l_i[:, None]

    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)

    if STORE_STATS:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
        tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
    store_stats: bool = False,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda, "All inputs must be on CUDA"
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expected 4D tensors (batch, heads, seq_len, head_dim)"

    batch, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, seq_len_k, head_dim_k = k.shape

    assert seq_len == seq_len_k, f"Q and K sequence lengths must match: {seq_len} vs {seq_len_k}"
    assert head_dim == head_dim_k, f"Q and K head dimensions must match: {head_dim} vs {head_dim_k}"
    assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)

    l = torch.empty((batch * num_heads, seq_len), device=q.device, dtype=torch.float32)
    m = torch.empty((batch * num_heads, seq_len), device=q.device, dtype=torch.float32)

    BLOCK_DMODEL = head_dim

    def grid(META):
        return (triton.cdiv(seq_len, META['BLOCK_M']), batch * num_heads)

    _flash_attention_fwd_kernel[grid](
        q, k, v, o,
        l, m,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        sm_scale,
        IS_CAUSAL=causal,
        STORE_STATS=store_stats,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )

    return o