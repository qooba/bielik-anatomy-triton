#!/usr/bin/env python3
"""
Focused benchmark: Triton Basic (FP32) vs PyTorch (BF16).

Compares our basic Triton FP32 kernel against PyTorch's cuBLAS BF16 implementation.
"""

import sys
import time
from pathlib import Path

import torch
import triton

# Add kernels directory to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))

from matmul.matmul_basic import matmul_kernel as matmul_basic_kernel


class FP32vsBF16Benchmark:
    """Compare Triton Basic FP32 vs PyTorch BF16."""

    def __init__(self, device: str = 'cuda', warmup: int = 10, iterations: int = 100):
        self.device = device
        self.warmup = warmup
        self.iterations = iterations
        self.results = []

        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.gpu_memory_gb = props.total_memory / 1e9
            self.gpu_compute_capability = f"{props.major}.{props.minor}"
        else:
            raise RuntimeError("CUDA not available!")

    def benchmark_pytorch_bf16(self, M: int, N: int, K: int) -> tuple[float, float]:
        """Benchmark PyTorch matmul with BF16."""
        a = torch.randn((M, K), device=self.device, dtype=torch.bfloat16)
        b = torch.randn((K, N), device=self.device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(self.warmup):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(self.iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()

        time_ms = (end - start) / self.iterations * 1000
        ops = 2 * M * N * K
        tflops = ops / (time_ms * 1e-3) / 1e12

        return time_ms, tflops

    def benchmark_triton_fp32(self, M: int, N: int, K: int) -> tuple[float, float]:
        """Benchmark Triton Basic kernel with FP32."""
        a = torch.randn((M, K), device=self.device, dtype=torch.float32)
        b = torch.randn((K, N), device=self.device, dtype=torch.float32)
        c = torch.empty((M, N), device=self.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        def grid(META):
            return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        # Warmup
        for _ in range(self.warmup):
            matmul_basic_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_K=BLOCK_K,
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(self.iterations):
            matmul_basic_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_K=BLOCK_K,
            )
        torch.cuda.synchronize()
        end = time.time()

        time_ms = (end - start) / self.iterations * 1000
        ops = 2 * M * N * K
        tflops = ops / (time_ms * 1e-3) / 1e12

        return time_ms, tflops

    def run_benchmark(self, M: int, N: int, K: int, description: str = ""):
        """Run comparison for given shape."""
        print(f"\n{'='*70}")
        print(f"Shape: {M} x {N} x {K}" + (f" ({description})" if description else ""))
        print(f"{'='*70}")

        # Triton Basic FP32
        print("Triton Basic (FP32)...", end=" ", flush=True)
        triton_time, triton_tflops = self.benchmark_triton_fp32(M, N, K)
        print(f"{triton_time:.3f} ms, {triton_tflops:.2f} TFLOPS")

        # PyTorch BF16
        print("PyTorch (BF16).......", end=" ", flush=True)
        pytorch_time, pytorch_tflops = self.benchmark_pytorch_bf16(M, N, K)
        print(f"{pytorch_time:.3f} ms, {pytorch_tflops:.2f} TFLOPS")

        # Comparison
        speedup = pytorch_time / triton_time
        tflops_ratio = triton_tflops / pytorch_tflops * 100

        print(f"\nTriton FP32 vs PyTorch BF16: {speedup:.2f}x ({tflops_ratio:.1f}% of BF16 TFLOPS)")

        self.results.append({
            'M': M, 'N': N, 'K': K,
            'description': description,
            'triton_fp32_ms': triton_time,
            'triton_fp32_tflops': triton_tflops,
            'pytorch_bf16_ms': pytorch_time,
            'pytorch_bf16_tflops': pytorch_tflops,
            'speedup': speedup,
        })

    def print_summary(self):
        """Print summary table."""
        print(f"\n{'='*70}")
        print("SUMMARY: Triton Basic (FP32) vs PyTorch (BF16)")
        print(f"{'='*70}")
        print(f"GPU: {self.gpu_name}")
        print(f"{'='*70}\n")

        print(f"{'Shape':<20} {'Triton FP32':>14} {'PyTorch BF16':>14} {'Ratio':>10}")
        print(f"{'':<20} {'(ms / TFLOPS)':>14} {'(ms / TFLOPS)':>14} {'':>10}")
        print(f"{'-'*70}")

        for r in self.results:
            shape = f"{r['M']}x{r['N']}x{r['K']}"
            triton_str = f"{r['triton_fp32_ms']:.2f} / {r['triton_fp32_tflops']:.1f}"
            pytorch_str = f"{r['pytorch_bf16_ms']:.2f} / {r['pytorch_bf16_tflops']:.1f}"
            ratio_str = f"{r['speedup']:.2f}x"
            print(f"{shape:<20} {triton_str:>14} {pytorch_str:>14} {ratio_str:>10}")


def main():
    print("="*70)
    print("BENCHMARK: Triton Basic (FP32) vs PyTorch (BF16)")
    print("="*70)

    benchmark = FP32vsBF16Benchmark(warmup=5, iterations=50)

    test_cases = [
        (512, 512, 512, "Small square"),
        (1024, 1024, 1024, "Medium square"),
        (2048, 2048, 2048, "Large square"),
        (128, 1536, 1536, "Bielik Q proj"),
        (128, 1536, 8960, "Bielik gate proj"),
    ]

    for M, N, K, desc in test_cases:
        benchmark.run_benchmark(M, N, K, desc)

    benchmark.print_summary()

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
