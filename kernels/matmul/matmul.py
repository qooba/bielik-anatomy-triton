"""
Tensor Core-optimized matrix multiplication kernel.

This is an optimized variant of matmul_tiled.py specifically tuned for tensor cores:
- Larger block sizes (256x256 vs 128x128 for xl tier)
- More pipeline stages (5-6 vs 2 for xl tier)
- More warps (8 vs 4 for better occupancy)
- Auto-detects GPU tier and uses optimal configuration

Expected speedup vs original: 1.1-1.8x depending on GPU tier
Expected speedup vs PyTorch: 0.9-1.1x (targeting cuBLAS parity)
"""

import torch
import triton
import triton.language as tl
import sys
import os

# Add common directory to path for GPU config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.gpu_configs import get_tensor_core_config, get_gpu_tier


# Get tensor core-optimized configurations for current GPU
tc_config = get_tensor_core_config()

# Build autotune configs from tensor core settings
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
    key=['M', 'N', 'K']  # Tune separately for each shape for best performance
    #key=[],  # Autotune once, use same config for all sizes
)
@triton.jit
def matmul(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters (auto-tuned by decorator)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Tensor core-optimized matrix multiplication kernel using block tiling.

    Optimizations for tensor cores:
    - Larger block sizes (up to 256x256) for better tensor core utilization
    - More pipeline stages (5-6) for better memory latency hiding
    - More warps (8) for higher occupancy on modern GPUs
    - fp32 accumulation for numerical stability
    - Block dimensions guaranteed to be multiples of 16 (tensor core tile size for bf16)

    C = A @ B
    - A has shape (M, K)
    - B has shape (K, N)
    - C has shape (M, N)

    Best performance with:
    - Input dtype: bfloat16 or float16
    - Matrix dimensions: multiples of 16
    - GPU: sm_70+ (Volta and newer with tensor cores)
    """
    # -----------------------------------------------------------
    # Map program ids to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # NOTE: Do NOT use modulo wrapping - it causes incorrect reads at boundaries.
    # Instead, use proper masking when loading.
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # CRITICAL: Use fp32 accumulator for numerical stability
    # Tensor cores will still be used for bf16 x bf16 -> fp32 dot product
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B with proper boundary masking.
        # Mask both the M/N dimensions AND the K dimension for correctness.
        # Inputs stay in bf16/fp16 - tensor cores operate on these dtypes
        k_offset = k * BLOCK_SIZE_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + k_offset < K)
        b_mask = (offs_k[:, None] + k_offset < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Tensor core operation: bf16 x bf16 -> fp32
        # tl.dot automatically uses tensor cores when:
        # - Input dtype is fp16/bf16
        # - Output dtype is fp32
        # - Block dimensions are multiples of tensor core tile size (16 for bf16)
        # - GPU has tensor cores (sm_70+)
        accumulator += tl.dot(a, b, out_dtype=tl.float32)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Test and compile kernel to PTX
if __name__ == "__main__":
    from common.gpu_configs import print_gpu_info

    print("="*80)
    print("Tensor Core-Optimized Matmul Kernel")
    print("="*80)
    print()
    print_gpu_info()
    print()

    # Test dimensions
    M, N, K = 1024, 1024, 1024

    # Create test tensors (bf16 for tensor cores)
    print(f"Testing with {M}x{K} @ {K}x{N} matmul (bfloat16)")
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
    c = torch.empty((M, N), device='cuda', dtype=torch.bfloat16)

    # Grid configuration
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

    # Compile kernel with auto-tuning
    print("\nCompiling kernel with auto-tuning...")
    print(f"Testing {len(autotune_configs)} tensor core configurations:")
    tier = get_gpu_tier()
    for i, config in enumerate(autotune_configs):
        block_m = config.kwargs['BLOCK_SIZE_M']
        block_n = config.kwargs['BLOCK_SIZE_N']
        block_k = config.kwargs['BLOCK_SIZE_K']
        stages = config.num_stages
        warps = config.num_warps
        print(f"  Config {i+1}: {block_m}x{block_n}x{block_k}, stages={stages}, warps={warps}")

    compiled_kernel = matmul[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    # Get output path from command line argument or use default
    output_path = sys.argv[1] if len(sys.argv) > 1 else "./compiled/matmul/matmul_tiled_tensorcore.ptx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write PTX
    with open(output_path, "w") as f:
        f.write(compiled_kernel.asm["ptx"])

    print(f"\n✓ Compiled tensor core kernel to {output_path}")

    # Verify correctness
    print("\nVerifying correctness...")
    c_torch = torch.matmul(a, b)
    max_diff = torch.max(torch.abs(c - c_torch))
    mean_diff = torch.mean(torch.abs(c - c_torch))
    relative_diff = max_diff / torch.max(torch.abs(c_torch))

    print(f"  Maximum absolute difference: {max_diff.item():.6f}")
    print(f"  Mean absolute difference: {mean_diff.item():.6f}")
    print(f"  Maximum relative difference: {relative_diff.item():.6f}")

    # For bf16, 0.1% relative error is acceptable
    if relative_diff < 1e-3 or max_diff < 0.01:
        print("  ✓ Kernel is correct!")
    else:
        print("  ✗ Kernel has numerical errors")
        print(f"  Sample outputs:")
        print(f"    Triton: {c[0, :5]}")
        print(f"    PyTorch: {c_torch[0, :5]}")

    # Benchmark
    print("\nBenchmarking...")
    import time
    warmup = 10
    iterations = 100

    # Warmup
    for _ in range(warmup):
        matmul[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    torch.cuda.synchronize()

    # Benchmark Triton tensorcore kernel
    start = time.time()
    for _ in range(iterations):
        matmul[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    torch.cuda.synchronize()
    end = time.time()

    ms_per_matmul = (end - start) / iterations * 1000
    tflops = (2.0 * M * N * K) / (ms_per_matmul * 1e-3) / 1e12

    print(f"\nTensor Core Kernel Performance:")
    print(f"  Time: {ms_per_matmul:.2f} ms")
    print(f"  Performance: {tflops:.2f} TFLOPS")

    # Compare with PyTorch cuBLAS
    for _ in range(warmup):
        c_torch = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        c_torch = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    ms_pytorch = (end - start) / iterations * 1000
    tflops_pytorch = (2.0 * M * N * K) / (ms_pytorch * 1e-3) / 1e12

    print(f"\nPyTorch cuBLAS:")
    print(f"  Time: {ms_pytorch:.2f} ms")
    print(f"  Performance: {tflops_pytorch:.2f} TFLOPS")

    speedup = ms_pytorch / ms_per_matmul
    print(f"\nSpeedup vs PyTorch: {speedup:.2f}x")

    # Expected performance based on GPU tier
    expected_speedup = {
        'xl': (1.3, 1.8),
        'large': (1.2, 1.5),
        'medium': (1.15, 1.3),
        'small': (1.1, 1.2),
    }

