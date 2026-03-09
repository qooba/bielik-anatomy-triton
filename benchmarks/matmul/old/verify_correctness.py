#!/usr/bin/env python3
"""
Correctness verification script for matmul kernels.

Verifies that:
1. matmul_basic.py (FP32) matches PyTorch output
2. matmul_tiled_tensorcore.py (BF16) matches PyTorch output

Tests various matrix sizes and reports pass/fail with detailed error metrics.
Returns exit code 0 if all tests pass, 1 otherwise.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import triton

# Add kernels directory to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))

from matmul.matmul import matmul as matmul_tiled_kernel
from matmul.matmul_basic import matmul_kernel as matmul_basic_kernel


@dataclass
class TestResult:
    """Single test result."""

    name: str
    M: int
    N: int
    K: int
    dtype: str
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    error_msg: Optional[str] = None


class MatmulVerifier:
    """Verifies matmul kernel correctness against PyTorch."""

    # Base tolerance thresholds per dtype
    # Note: Actual tolerance scales with sqrt(K) due to accumulation order differences.
    # These values are empirically determined from comparing Triton vs cuBLAS.
    # BF16 has only 7 bits mantissa (vs 23 for FP32), so larger errors are expected.
    BASE_TOLERANCES = {
        "float32": {"max_rel": 1e-2, "base_abs": 5e-3},  # FP32: accumulation order
        "bfloat16": {"max_rel": 5e-2, "base_abs": 4e-2},  # BF16: 7-bit mantissa
        "float16": {"max_rel": 2e-2, "base_abs": 5e-3},  # FP16: 10-bit mantissa
    }

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results: List[TestResult] = []

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")

        self.gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        self.compute_capability = f"{props.major}.{props.minor}"

    def run_basic_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run matmul_basic kernel."""
        M, K = a.shape
        K_, N = b.shape
        assert K == K_

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        def grid(META):
            return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        matmul_basic_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
        )
        return c

    def run_tensorcore_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run matmul_tiled_tensorcore kernel."""
        M, K = a.shape
        K_, N = b.shape
        assert K == K_

        # Output in same dtype as input, but converted from fp32 accumulator
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

        matmul_tiled_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
        return c

    def compare_tensors(
        self, result: torch.Tensor, expected: torch.Tensor, dtype_name: str, K: int
    ) -> Tuple[bool, float, float, float]:
        """Compare two tensors and return (passed, max_abs, mean_abs, max_rel).

        Tolerance scales with sqrt(K) because matmul accumulates K products,
        and different accumulation orders cause O(sqrt(K) * eps) differences.
        """
        # Convert to float32 for comparison
        result_f32 = result.float()
        expected_f32 = expected.float()

        abs_diff = torch.abs(result_f32 - expected_f32)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        # Relative difference (avoid div by zero)
        rel_diff = abs_diff / (torch.abs(expected_f32) + 1e-10)
        max_rel_diff = rel_diff.max().item()

        # Scale tolerance with sqrt(K) for accumulation error
        import math

        tol = self.BASE_TOLERANCES[dtype_name]
        scale_factor = math.sqrt(K)
        max_abs_tol = tol["base_abs"] * scale_factor

        passed = (max_rel_diff < tol["max_rel"]) or (max_abs_diff < max_abs_tol)

        return passed, max_abs_diff, mean_abs_diff, max_rel_diff

    def verify_basic_kernel(self, M: int, N: int, K: int) -> TestResult:
        """Verify matmul_basic kernel against PyTorch."""
        dtype = torch.float32
        dtype_name = "float32"
        name = f"matmul_basic ({M}x{K} @ {K}x{N})"

        try:
            # Create test matrices
            a = torch.randn((M, K), device=self.device, dtype=dtype)
            b = torch.randn((K, N), device=self.device, dtype=dtype)

            # Run Triton kernel
            c_triton = self.run_basic_kernel(a, b)

            # Run PyTorch reference
            c_pytorch = torch.matmul(a, b)

            # Compare
            passed, max_abs, mean_abs, max_rel = self.compare_tensors(
                c_triton, c_pytorch, dtype_name, K
            )

            return TestResult(
                name=name,
                M=M,
                N=N,
                K=K,
                dtype=dtype_name,
                passed=passed,
                max_abs_diff=max_abs,
                mean_abs_diff=mean_abs,
                max_rel_diff=max_rel,
            )

        except Exception as e:
            return TestResult(
                name=name,
                M=M,
                N=N,
                K=K,
                dtype=dtype_name,
                passed=False,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                error_msg=str(e),
            )

    def verify_tensorcore_kernel(
        self, M: int, N: int, K: int, dtype: torch.dtype = torch.bfloat16
    ) -> TestResult:
        """Verify matmul_tiled_tensorcore kernel against PyTorch."""
        dtype_name = str(dtype).split(".")[-1]
        name = f"matmul_tensorcore ({M}x{K} @ {K}x{N}, {dtype_name})"

        try:
            # Create test matrices
            a = torch.randn((M, K), device=self.device, dtype=dtype)
            b = torch.randn((K, N), device=self.device, dtype=dtype)

            # Run Triton kernel
            c_triton = self.run_tensorcore_kernel(a, b)

            # Run PyTorch reference
            c_pytorch = torch.matmul(a, b)

            # Compare
            passed, max_abs, mean_abs, max_rel = self.compare_tensors(
                c_triton, c_pytorch, dtype_name, K
            )

            return TestResult(
                name=name,
                M=M,
                N=N,
                K=K,
                dtype=dtype_name,
                passed=passed,
                max_abs_diff=max_abs,
                mean_abs_diff=mean_abs,
                max_rel_diff=max_rel,
            )

        except Exception as e:
            return TestResult(
                name=name,
                M=M,
                N=N,
                K=K,
                dtype=dtype_name,
                passed=False,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                max_rel_diff=float("inf"),
                error_msg=str(e),
            )

    def run_all_tests(self) -> bool:
        """Run all verification tests. Returns True if all pass."""
        print("=" * 80)
        print("MATMUL KERNEL CORRECTNESS VERIFICATION")
        print("=" * 80)
        print(f"GPU: {self.gpu_name} (sm_{self.compute_capability})")
        print("=" * 80)

        # Test cases: (M, N, K)
        test_sizes = [
            # Small matrices
            (64, 64, 64),
            (128, 128, 128),
            # Medium matrices
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            # Non-square matrices
            (128, 256, 64),
            (256, 128, 512),
            (1, 1024, 1024),  # Single row
            (1024, 1, 1024),  # Single column
            # Non-power-of-2 (edge cases)
            (127, 127, 127),
            (257, 257, 257),
            (100, 200, 300),
            # Bielik-like shapes
            (1, 1536, 1536),  # Single token Q proj
            (128, 1536, 1536),  # Batch Q proj
            (128, 1536, 8960),  # FFN gate proj
        ]

        # Test matmul_basic (FP32)
        print("\n" + "-" * 80)
        print("Testing: matmul_basic.py (FP32)")
        print("-" * 80)

        for M, N, K in test_sizes:
            result = self.verify_basic_kernel(M, N, K)
            self.results.append(result)
            self._print_result(result)

        # Test matmul_tiled_tensorcore (BF16)
        print("\n" + "-" * 80)
        print("Testing: matmul_tiled_tensorcore.py (BF16)")
        print("-" * 80)

        for M, N, K in test_sizes:
            result = self.verify_tensorcore_kernel(M, N, K, torch.bfloat16)
            self.results.append(result)
            self._print_result(result)

        # Test matmul_tiled_tensorcore (FP16)
        print("\n" + "-" * 80)
        print("Testing: matmul_tiled_tensorcore.py (FP16)")
        print("-" * 80)

        for M, N, K in test_sizes:
            result = self.verify_tensorcore_kernel(M, N, K, torch.float16)
            self.results.append(result)
            self._print_result(result)

        # Summary
        return self._print_summary()

    def _print_result(self, result: TestResult):
        """Print a single test result."""
        status = "PASS" if result.passed else "FAIL"
        symbol = "[OK]" if result.passed else "[X]"

        if result.error_msg:
            print(f"  {symbol} {result.name}: ERROR - {result.error_msg}")
        else:
            print(f"  {symbol} {result.name}")
            print(
                f"       max_abs={result.max_abs_diff:.2e}, "
                f"mean_abs={result.mean_abs_diff:.2e}, "
                f"max_rel={result.max_rel_diff:.2e}"
            )

    def _print_summary(self) -> bool:
        """Print test summary. Returns True if all tests passed."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(f"Total: {total} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    if r.error_msg:
                        print(f"  - {r.name}: {r.error_msg}")
                    else:
                        print(f"  - {r.name}: max_rel={r.max_rel_diff:.2e}")

        print("=" * 80)

        if failed == 0:
            print("ALL TESTS PASSED")
            return True
        else:
            print(f"TESTS FAILED: {failed}/{total}")
            return False


def main():
    """Run correctness verification."""
    verifier = MatmulVerifier()
    all_passed = verifier.run_all_tests()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
