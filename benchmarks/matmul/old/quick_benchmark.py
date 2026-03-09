#!/usr/bin/env python3
"""
Quick matmul benchmark - fewer test cases, faster execution.

Use this for quick validation during development.
"""

import os
import sys
from pathlib import Path

# Add kernels directory to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "matmul"))

import torch
from benchmark_matmul import MatmulBenchmark


def main():
    """Run quick benchmarks on key test cases."""
    print("=" * 80)
    print("QUICK MATMUL BENCHMARK")
    print("=" * 80)
    print("\nRunning reduced test suite (warmup=5, iterations=50)")
    print("For full benchmark, run: python benchmark_matmul.py")
    print("=" * 80)

    # Quick benchmark with fewer iterations
    benchmark = MatmulBenchmark(warmup=5, iterations=50)

    # Key test cases only
    test_cases = [
        # (M, N, K, dtype, description)
        (1024, 1024, 1024, torch.float32, "Square 1024 (FP32) - Basic vs PyTorch"),
        (1024, 1024, 1024, torch.bfloat16, "Square 1024 (BF16) - Advanced vs PyTorch"),
        (128, 1536, 1536, torch.bfloat16, "Bielik Q proj (seq=128)"),
        (128, 1536, 8960, torch.bfloat16, "Bielik gate proj (seq=128)"),
    ]

    print(f"\nRunning {len(test_cases)} quick test cases...\n")

    for M, N, K, dtype, description in test_cases:
        print(f"\n{'#'*80}")
        print(f"# {description}")
        print(f"{'#'*80}")
        benchmark.run_benchmark(M, N, K, dtype)

    # Print summary
    benchmark.print_summary_table()

    print("\n" + "=" * 80)
    print("QUICK BENCHMARK COMPLETE!")
    print("=" * 80)
    print("\nFor comprehensive results, run: python benchmark_matmul.py")


if __name__ == "__main__":
    main()
