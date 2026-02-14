#!/usr/bin/env python3
"""
Comprehensive matmul benchmarking script.

Compares:
1. Basic Triton kernel (matmul_basic.py) - FP32
2. Advanced Triton kernel (matmul_tiled_tensorcore.py) - BF16
3. PyTorch (torch.matmul - uses cuBLAS)
4. cuBLAS direct (via torch.matmul, same as PyTorch)

Tests various matrix sizes relevant to Bielik-11B architecture.
"""

import sys
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import triton

# Add kernels directory to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))

from matmul.matmul_basic import matmul_kernel as matmul_basic_kernel
from matmul.matmul import matmul as matmul_tiled_kernel


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    implementation: str
    M: int
    N: int
    K: int
    dtype: str
    time_ms: float
    tflops: float
    bandwidth_gb_s: float
    vs_pytorch_speedup: float


class MatmulBenchmark:
    """Matmul benchmarking suite."""

    def __init__(self, device: str = 'cuda', warmup: int = 10, iterations: int = 100):
        self.device = device
        self.warmup = warmup
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []

        # Get GPU info
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.gpu_memory_gb = props.total_memory / 1e9
            self.gpu_compute_capability = f"{props.major}.{props.minor}"
        else:
            raise RuntimeError("CUDA not available!")

    def benchmark_pytorch(self, a: torch.Tensor, b: torch.Tensor, name: str = "PyTorch") -> float:
        """Benchmark PyTorch matmul (cuBLAS)."""
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

        return (end - start) / self.iterations * 1000  # ms

    def benchmark_triton_basic(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Benchmark basic Triton kernel."""
        M, K = a.shape
        K_, N = b.shape
        assert K == K_, f"Inner dimensions must match: {K} != {K_}"

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # Fixed block sizes for basic kernel
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        def grid(META):
            return (
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            )

        # Warmup
        for _ in range(self.warmup):
            # Compute grid size
            num_pid_m = triton.cdiv(M, BLOCK_M)
            num_pid_n = triton.cdiv(N, BLOCK_N)

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

        return (end - start) / self.iterations * 1000  # ms

    def benchmark_triton_advanced(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Benchmark advanced Triton kernel (auto-tuned, tensor cores)."""
        M, K = a.shape
        K_, N = b.shape
        assert K == K_, f"Inner dimensions must match: {K} != {K_}"

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

        # Warmup (will auto-tune on first call)
        for _ in range(self.warmup):
            matmul_tiled_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(self.iterations):
            matmul_tiled_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
        torch.cuda.synchronize()
        end = time.time()

        return (end - start) / self.iterations * 1000  # ms

    def run_benchmark(self, M: int, N: int, K: int, dtype: torch.dtype):
        """Run all benchmarks for given shape and dtype."""
        dtype_name = str(dtype).split('.')[-1]
        print(f"\n{'='*80}")
        print(f"Benchmarking: M={M}, N={N}, K={K}, dtype={dtype_name}")
        print(f"{'='*80}")

        # Create test matrices
        a = torch.randn((M, K), device=self.device, dtype=dtype)
        b = torch.randn((K, N), device=self.device, dtype=dtype)

        # Calculate theoretical values
        ops = 2 * M * N * K  # multiply + add
        bytes_accessed = (M * K + K * N + M * N) * a.element_size()

        results = {}

        # 1. PyTorch (cuBLAS)
        print(f"\n[1/4] PyTorch (cuBLAS)...", end=" ", flush=True)
        time_pytorch = self.benchmark_pytorch(a, b)
        tflops_pytorch = ops / (time_pytorch * 1e-3) / 1e12
        bandwidth_pytorch = bytes_accessed / (time_pytorch * 1e-3) / 1e9
        print(f"{time_pytorch:.3f} ms, {tflops_pytorch:.2f} TFLOPS")
        results['pytorch'] = {
            'time_ms': time_pytorch,
            'tflops': tflops_pytorch,
            'bandwidth_gb_s': bandwidth_pytorch,
        }

        # 2. Basic Triton (only for FP32)
        if dtype == torch.float32:
            print(f"[2/4] Triton Basic (FP32)...", end=" ", flush=True)
            try:
                time_basic = self.benchmark_triton_basic(a, b)
                tflops_basic = ops / (time_basic * 1e-3) / 1e12
                bandwidth_basic = bytes_accessed / (time_basic * 1e-3) / 1e9
                speedup_basic = time_pytorch / time_basic
                print(f"{time_basic:.3f} ms, {tflops_basic:.2f} TFLOPS, {speedup_basic:.2f}× vs PyTorch")
                results['triton_basic'] = {
                    'time_ms': time_basic,
                    'tflops': tflops_basic,
                    'bandwidth_gb_s': bandwidth_basic,
                    'speedup': speedup_basic,
                }

                self.results.append(BenchmarkResult(
                    implementation="Triton Basic (FP32)",
                    M=M, N=N, K=K,
                    dtype=dtype_name,
                    time_ms=time_basic,
                    tflops=tflops_basic,
                    bandwidth_gb_s=bandwidth_basic,
                    vs_pytorch_speedup=speedup_basic,
                ))
            except Exception as e:
                print(f"FAILED: {e}")
                results['triton_basic'] = None
        else:
            print(f"[2/4] Triton Basic - SKIPPED (only FP32 supported)")
            results['triton_basic'] = None

        # 3. Advanced Triton (BF16/FP16)
        if dtype in [torch.bfloat16, torch.float16]:
            print(f"[3/4] Triton Advanced ({dtype_name})...", end=" ", flush=True)
            try:
                time_advanced = self.benchmark_triton_advanced(a, b)
                tflops_advanced = ops / (time_advanced * 1e-3) / 1e12
                bandwidth_advanced = bytes_accessed / (time_advanced * 1e-3) / 1e9
                speedup_advanced = time_pytorch / time_advanced
                print(f"{time_advanced:.3f} ms, {tflops_advanced:.2f} TFLOPS, {speedup_advanced:.2f}× vs PyTorch")
                results['triton_advanced'] = {
                    'time_ms': time_advanced,
                    'tflops': tflops_advanced,
                    'bandwidth_gb_s': bandwidth_advanced,
                    'speedup': speedup_advanced,
                }

                self.results.append(BenchmarkResult(
                    implementation=f"Triton Advanced ({dtype_name})",
                    M=M, N=N, K=K,
                    dtype=dtype_name,
                    time_ms=time_advanced,
                    tflops=tflops_advanced,
                    bandwidth_gb_s=bandwidth_advanced,
                    vs_pytorch_speedup=speedup_advanced,
                ))
            except Exception as e:
                print(f"FAILED: {e}")
                results['triton_advanced'] = None
        else:
            print(f"[3/4] Triton Advanced - SKIPPED (requires BF16/FP16)")
            results['triton_advanced'] = None

        # 4. PyTorch reference again (for consistency)
        print(f"[4/4] PyTorch Reference...", end=" ", flush=True)
        time_ref = self.benchmark_pytorch(a, b, "PyTorch Ref")
        print(f"{time_ref:.3f} ms")

        # Store PyTorch result
        self.results.append(BenchmarkResult(
            implementation=f"PyTorch ({dtype_name})",
            M=M, N=N, K=K,
            dtype=dtype_name,
            time_ms=time_pytorch,
            tflops=tflops_pytorch,
            bandwidth_gb_s=bandwidth_pytorch,
            vs_pytorch_speedup=1.0,
        ))

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY:")
        print(f"  PyTorch:         {time_pytorch:.3f} ms ({tflops_pytorch:.2f} TFLOPS) [baseline]")
        if results['triton_basic']:
            pct = (results['triton_basic']['tflops'] / tflops_pytorch) * 100
            print(f"  Triton Basic:    {results['triton_basic']['time_ms']:.3f} ms ({results['triton_basic']['tflops']:.2f} TFLOPS) [{pct:.1f}% of PyTorch]")
        if results['triton_advanced']:
            pct = (results['triton_advanced']['tflops'] / tflops_pytorch) * 100
            print(f"  Triton Advanced: {results['triton_advanced']['time_ms']:.3f} ms ({results['triton_advanced']['tflops']:.2f} TFLOPS) [{pct:.1f}% of PyTorch]")
        print(f"{'='*80}")

    def print_summary_table(self):
        """Print summary table of all results."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY - ALL RESULTS")
        print(f"{'='*80}")
        print(f"GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB, sm_{self.gpu_compute_capability})")
        print(f"Warmup: {self.warmup} iterations, Benchmark: {self.iterations} iterations")
        print(f"{'='*80}\n")

        # Group by shape
        shapes = {}
        for result in self.results:
            key = f"{result.M}×{result.N}×{result.K}"
            if key not in shapes:
                shapes[key] = []
            shapes[key].append(result)

        for shape, results in shapes.items():
            print(f"\nShape: {shape}")
            print(f"{'-'*80}")
            print(f"{'Implementation':<30} {'Time (ms)':<12} {'TFLOPS':<10} {'vs PyTorch':<12}")
            print(f"{'-'*80}")
            for r in results:
                speedup_str = f"{r.vs_pytorch_speedup:.2f}×" if r.vs_pytorch_speedup != 1.0 else "baseline"
                print(f"{r.implementation:<30} {r.time_ms:>10.3f}   {r.tflops:>8.2f}   {speedup_str:>10}")

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename

        data = {
            'gpu_info': {
                'name': self.gpu_name,
                'memory_gb': self.gpu_memory_gb,
                'compute_capability': self.gpu_compute_capability,
            },
            'benchmark_config': {
                'warmup': self.warmup,
                'iterations': self.iterations,
            },
            'results': [asdict(r) for r in self.results],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Run comprehensive matmul benchmarks."""
    print("="*80)
    print("MATMUL BENCHMARK SUITE")
    print("="*80)
    print("\nComparing:")
    print("  1. PyTorch (cuBLAS) - FP32 and BF16")
    print("  2. Triton Basic - FP32 only, fixed blocks (64×64×32)")
    print("  3. Triton Advanced - BF16, auto-tuned, grouped ordering, Tensor Cores")
    print("="*80)

    benchmark = MatmulBenchmark(warmup=10, iterations=100)

    # Test cases: Bielik architecture sizes
    test_cases = [
        # (M, N, K, dtype, description)
        # Square matrices
        (512, 512, 512, torch.float32, "Small square (FP32)"),
        (1024, 1024, 1024, torch.float32, "Medium square (FP32)"),
        (1024, 1024, 1024, torch.bfloat16, "Medium square (BF16)"),
        (2048, 2048, 2048, torch.bfloat16, "Large square (BF16)"),

        # Bielik Q/K/V projections
        (1, 1536, 1536, torch.bfloat16, "Bielik Q proj (single token)"),
        (128, 1536, 1536, torch.bfloat16, "Bielik Q proj (seq=128)"),
        (1, 1536, 256, torch.bfloat16, "Bielik K proj (single token, GQA)"),
        (128, 1536, 256, torch.bfloat16, "Bielik K proj (seq=128, GQA)"),

        # Bielik FFN projections (largest!)
        (1, 1536, 8960, torch.bfloat16, "Bielik gate proj (single token)"),
        (128, 1536, 8960, torch.bfloat16, "Bielik gate proj (seq=128)"),
        (1, 8960, 1536, torch.bfloat16, "Bielik down proj (single token)"),
        (128, 8960, 1536, torch.bfloat16, "Bielik down proj (seq=128)"),
    ]

    print(f"\nRunning {len(test_cases)} test cases...\n")

    for M, N, K, dtype, description in test_cases:
        print(f"\n{'#'*80}")
        print(f"# {description}")
        print(f"{'#'*80}")
        benchmark.run_benchmark(M, N, K, dtype)

    # Print summary
    benchmark.print_summary_table()

    # Save results
    benchmark.save_results()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
