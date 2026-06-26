import torch
import triton
import triton.language as tl


@triton.jit
def fused_ffn_swiglu_bias_kernel(
    X_ptr, W_gate_ptr, b_gate_ptr, W_up_ptr, b_up_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs  = X_ptr      + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wg_ptrs = W_gate_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    wu_ptrs = W_up_ptr   + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask   = offs_k[None, :] + k * BLOCK_K < K
        k_mask_w = offs_k[:, None] + k * BLOCK_K < K

        x  = tl.load(x_ptrs,  mask=(offs_m[:, None] < M) & k_mask,  other=0.0)
        wg = tl.load(wg_ptrs, mask=k_mask_w & (offs_n[None, :] < N), other=0.0)
        wu = tl.load(wu_ptrs, mask=k_mask_w & (offs_n[None, :] < N), other=0.0)

        acc_gate = tl.dot(x, wg, acc=acc_gate, out_dtype=tl.float32)
        acc_up   = tl.dot(x, wu, acc=acc_up, out_dtype=tl.float32)

        x_ptrs  += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        offs_bias = offs_n
        mask_bias = offs_bias < N

        b_gate = tl.load(b_gate_ptr + offs_bias, mask=mask_bias, other=0.0)
        b_up   = tl.load(b_up_ptr + offs_bias, mask=mask_bias, other=0.0)

        acc_gate = acc_gate + b_gate[None, :].to(tl.float32)
        acc_up   = acc_up + b_up[None, :].to(tl.float32)

    out = acc_gate * tl.sigmoid(acc_gate) * acc_up

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_ffn_swiglu_bias(x, W_gate, b_gate, W_up, b_up):
    orig_shape = x.shape
    K = orig_shape[-1]
    x_2d = x.reshape(-1, K)
    M = x_2d.shape[0]
    N = W_gate.shape[1]

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    has_bias = (b_gate is not None) and (b_up is not None)
    if has_bias:
        assert b_gate.shape == (N,), f"b_gate shape {b_gate.shape} != (N={N},)"
        assert b_up.shape == (N,), f"b_up shape {b_up.shape} != (N={N},)"

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fused_ffn_swiglu_bias_kernel[grid](
        x_2d, W_gate,
        b_gate if has_bias else x_2d,
        W_up,
        b_up if has_bias else x_2d,
        output,
        M, N, K,
        x_2d.stride(0),   x_2d.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        HAS_BIAS=has_bias,
    )

    return output.reshape(*orig_shape[:-1], N)
