import torch
import triton
import triton.language as tl
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))

from kernels.common.gpu_configs import get_tensor_core_config
tc_config = get_tensor_core_config()

autotune_configs = [
    triton.Config(
        {
            'BLOCK_SIZE_M': m,
            'BLOCK_SIZE_N': n,
            'BLOCK_SIZE_K': k,
            'GROUP_SIZE_M': 8
        },
        num_stages=tc_config['num_stages'],
        num_warps=tc_config['num_warps']
    )
    for m, n, k in tc_config['block_sizes']
]


@triton.autotune(
    configs=autotune_configs,
    key=[],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0)
        accumulator = accumulator + bias[None, :].to(tl.float32)

    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_bias_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} @ {b.shape}"

    if bias is not None:
        assert bias.is_cuda, "Bias must be on CUDA"
        assert bias.ndim == 1, f"Bias must be 1D, got shape {bias.shape}"
        assert bias.shape[0] == b.shape[1], f"Bias shape {bias.shape} doesn't match output dim {b.shape[1]}"

    M, K = a.shape
    K2, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_bias_kernel[grid](
        a, b,
        bias if bias is not None else a,
        c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return c
