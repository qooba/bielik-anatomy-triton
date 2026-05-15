import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_ffn_swiglu_kernel(
    X_ptr, W_gate_ptr, W_up_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

        acc_gate = tl.dot(x, wg, acc=acc_gate)
        acc_up   = tl.dot(x, wu, acc=acc_up)

        x_ptrs  += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    out = acc_gate * tl.sigmoid(acc_gate) * acc_up

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_ffn_swiglu(x, W_gate, W_up):
    orig_shape = x.shape
    K = orig_shape[-1]
    x_2d = x.reshape(-1, K)
    M = x_2d.shape[0]
    N = W_gate.shape[1]

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fused_ffn_swiglu_kernel[grid](
        x_2d, W_gate, W_up, output,
        M, N, K,
        x_2d.stride(0),   x_2d.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )

    return output.reshape(*orig_shape[:-1], N)