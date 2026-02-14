import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing matrix multiplication C = A @ B.

    - A has shape (M, K)
    - B has shape (K, N)
    - C has shape (M, N)

    This is a basic implementation based on Triton tutorials.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers to A and B for this block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        # Create masks to handle out-of-bounds accesses
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Multiply and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers for next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    #c = accumulator.to(tl.float32)

    # Create output pointers and mask
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# Compile kernel to PTX
if __name__ == "__main__":
    import sys
    import os

    # Test dimensions
    M, N, K = 512, 512, 512

    # Create test tensors
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # Grid configuration
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

    # Compile kernel with specific block sizes
    compiled_kernel = matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )

    # Get output path from command line argument or use default
    output_path = sys.argv[1] if len(sys.argv) > 1 else "./compiled/matmul/matmul_basic.ptx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write PTX
    with open(output_path, "w") as f:
        f.write(compiled_kernel.asm["ptx"])

    print(f"Compiled matmul_basic kernel to {output_path}")

    # Verify correctness
    c_torch = torch.matmul(a, b)
    max_diff = torch.max(torch.abs(c - c_torch))
    mean_diff = torch.mean(torch.abs(c - c_torch))
    relative_diff = max_diff / torch.max(torch.abs(c_torch))

    print(f"Maximum absolute difference: {max_diff.item():.6f}")
    print(f"Mean absolute difference: {mean_diff.item():.6f}")
    print(f"Maximum relative difference: {relative_diff.item():.6f}")

    # For FP32, 0.1% relative error or 0.1 absolute error is acceptable
    if relative_diff < 1e-3 or max_diff < 0.1:
        print("✓ Kernel is correct!")
    else:
        print("✗ Kernel has numerical errors")
        print(f"Sample outputs:")
        print(f"  Triton: {c[0, :5]}")
        print(f"  PyTorch: {c_torch[0, :5]}")
