#!/usr/bin/env python3
"""
Matrix Multiplication Benchmark: Triton vs PyTorch cuBLAS.
Usage:
  python benchmark_matmul_guide.py
  python benchmark_matmul_guide.py --save-plots  # save PNG files
"""

import argparse
from pathlib import Path

import torch
import triton
import triton.testing

from benchmarks.common.plotting import plot_summary_comparison
from kernels.matmul.matmul import matmul as matmul_tensorcore_kernel
from kernels.matmul.matmul_basic import matmul_kernel as matmul_basic_kernel


def pytorch_native_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch native matmul using cuBLAS (BF16)."""
    return torch.matmul(a, b)


def pytorch_native_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch native matmul using cuBLAS (FP32)."""
    return torch.matmul(a, b)


def _warmup_triton_tensorcore(a: torch.Tensor, b: torch.Tensor) -> None:
    """Pre-compile Triton tensor core kernel (exclude compilation from timing)."""
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Inner dimensions must match: {K} != {K_}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    matmul_tensorcore_kernel[grid](
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


def _warmup_triton_basic(a: torch.Tensor, b: torch.Tensor) -> None:
    """Pre-compile Triton basic kernel (exclude compilation from timing)."""
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Inner dimensions must match: {K} != {K_}"

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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton_tensorcore", "pytorch_bf16", "triton_basic"],
        line_names=["Triton Tensor Core (BF16)", "PyTorch cuBLAS (BF16)", "Triton Basic (FP32)"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOPS",
        plot_name="matmul-tflops-vs-size",
        args={},
    )
)
def bench_square_matmul(size, provider):
    """
    Sweep 1: vary matrix size (square matrices)
    """
    M = N = K = size

    if provider == "triton_basic":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    def grid_tensorcore(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    def grid_basic(META):
        BLOCK_M, BLOCK_N = 64, 64
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    if provider == "triton_tensorcore":
        _warmup_triton_tensorcore(a, b)
        c = torch.empty((M, N), device="cuda", dtype=dtype)
        fn = lambda: matmul_tensorcore_kernel[grid_tensorcore](
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
    elif provider == "pytorch_bf16":
        fn = lambda: pytorch_native_bf16(a, b)
    else:  # triton_basic
        _warmup_triton_basic(a, b)
        c = torch.empty((M, N), device="cuda", dtype=dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        fn = lambda: matmul_basic_kernel[grid_basic](
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

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

    ops = 2 * M * N * K
    tflops = lambda t: ops / (t * 1e9)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape"],
        x_vals=[
            "Q_proj_1",  # (1, 1536, 1536) - single token
            "Q_proj_128",  # (128, 1536, 1536) - seq=128
            "K_proj_1",  # (1, 1536, 256) - GQA
            "K_proj_128",  # (128, 1536, 256) - GQA
            "FFN_up_1",  # (1, 1536, 8960) - FFN gate/up
            "FFN_up_128",  # (128, 1536, 8960)
            "FFN_down_1",  # (1, 8960, 1536) - FFN down
            "FFN_down_128",  # (128, 8960, 1536)
        ],
        line_arg="provider",
        line_vals=["triton_tensorcore", "pytorch_bf16", "triton_basic"],
        line_names=["Triton Tensor Core (BF16)", "PyTorch cuBLAS (BF16)", "Triton Basic (FP32)"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOPS",
        plot_name="matmul-tflops-vs-shape-bielik",
        args={},
    )
)
def bench_bielik_shapes(shape, provider):
    """
    Sweep 2: vary shape (Bielik architecture shapes)
    """
    shapes = {
        "Q_proj_1": (1, 1536, 1536),
        "Q_proj_128": (128, 1536, 1536),
        "K_proj_1": (1, 1536, 256),
        "K_proj_128": (128, 1536, 256),
        "FFN_up_1": (1, 1536, 8960),
        "FFN_up_128": (128, 1536, 8960),
        "FFN_down_1": (1, 8960, 1536),
        "FFN_down_128": (128, 8960, 1536),
    }

    M, K, N = shapes[shape]

    if provider == "triton_basic":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    def grid_tensorcore(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    def grid_basic(META):
        BLOCK_M, BLOCK_N = 64, 64
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    if provider == "triton_tensorcore":
        _warmup_triton_tensorcore(a, b)
        c = torch.empty((M, N), device="cuda", dtype=dtype)
        fn = lambda: matmul_tensorcore_kernel[grid_tensorcore](
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
    elif provider == "pytorch_bf16":
        fn = lambda: pytorch_native_bf16(a, b)
    else:  # triton_basic
        _warmup_triton_basic(a, b)
        c = torch.empty((M, N), device="cuda", dtype=dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        fn = lambda: matmul_basic_kernel[grid_basic](
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

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    ops = 2 * M * N * K
    tflops = lambda t: ops / (t * 1e9)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def print_summary(save_path=""):
    """Print summary table of results and optionally save plots."""
    print()
    print("=" * 90)
    print(f"  Bielik-1.5B Matmul Shapes")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 90)
    print(f"  {'Shape':<20} {'M×K×N':<20} {'Provider':<25} {'ms':>8} {'TFLOPS':>10}")
    print("-" * 90)

    test_cases = [
        ("Q proj (single)", (1, 1536, 1536)),
        ("Q proj (seq=128)", (128, 1536, 1536)),
        ("K proj GQA (single)", (1, 1536, 256)),
        ("K proj GQA (seq=128)", (128, 1536, 256)),
        ("FFN up (single)", (1, 1536, 8960)),
        ("FFN up (seq=128)", (128, 1536, 8960)),
        ("FFN down (single)", (1, 8960, 1536)),
        ("FFN down (seq=128)", (128, 8960, 1536)),
    ]

    data_dict = {
        "Triton Tensor Core (BF16)": [],
        "PyTorch cuBLAS (BF16)": [],
        "Triton Basic (FP32)": [],
    }
    shapes = []

    for name, (M, K, N) in test_cases:
        shape_str = f"{M}x{K}x{N}"
        shapes.append(name)

        a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
        c_bf16 = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

        a_fp32 = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b_fp32 = torch.randn((K, N), device="cuda", dtype=torch.float32)
        c_fp32 = torch.empty((M, N), device="cuda", dtype=torch.float32)

        _warmup_triton_tensorcore(a_bf16, b_bf16)
        _warmup_triton_basic(a_fp32, b_fp32)

        def grid_tensorcore(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

        def grid_basic(META):
            BLOCK_M, BLOCK_N = 64, 64
            return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        cases = [
            (
                "Triton Tensor Core (BF16)",
                lambda: matmul_tensorcore_kernel[grid_tensorcore](
                    a_bf16,
                    b_bf16,
                    c_bf16,
                    M,
                    N,
                    K,
                    a_bf16.stride(0),
                    a_bf16.stride(1),
                    b_bf16.stride(0),
                    b_bf16.stride(1),
                    c_bf16.stride(0),
                    c_bf16.stride(1),
                ),
            ),
            ("PyTorch cuBLAS (BF16)", lambda: pytorch_native_bf16(a_bf16, b_bf16)),
            (
                "Triton Basic (FP32)",
                lambda: matmul_basic_kernel[grid_basic](
                    a_fp32,
                    b_fp32,
                    c_fp32,
                    M,
                    N,
                    K,
                    a_fp32.stride(0),
                    a_fp32.stride(1),
                    b_fp32.stride(0),
                    b_fp32.stride(1),
                    c_fp32.stride(0),
                    c_fp32.stride(1),
                    BLOCK_SIZE_M=64,
                    BLOCK_SIZE_N=64,
                    BLOCK_SIZE_K=32,
                ),
            ),
        ]

        for provider_name, fn in cases:
            ms = triton.testing.do_bench(fn)
            ops = 2 * M * N * K
            tflops = ops / (ms * 1e9)
            data_dict[provider_name].append([ms, tflops])
            print(f"  {name:<20} {shape_str:<20} {provider_name:<25} {ms:>8.3f} {tflops:>10.2f}")
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=shapes,
            metrics=["Latency (ms)", "TFLOPS"],
            title="Matmul Performance - Bielik-1.5B Shapes",
            xlabel="Operation Shape",
            save_path=save_path,
            filename_prefix="matmul-summary-bielik-shapes",
            gpu_name=torch.cuda.get_device_name(0),
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matmul benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument(
        "--plot-dir", default=".", help="Directory to save plots (default: current dir)"
    )
    args = parser.parse_args()

    print("=" * 90)
    print("Matrix Multiplication Benchmark: Triton vs PyTorch cuBLAS")
    print("=" * 90)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.save_plots:
        from pathlib import Path

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path_1 = str(plot_dir)
        save_path_2 = str(plot_dir)
        print(f"Saving plots to: {plot_dir.absolute()}/")
        print()
    else:
        save_path_1 = ""
        save_path_2 = ""

    print("--- Sweep 1: vary matrix size (square matrices, M=N=K) ---")
    bench_square_matmul.run(show_plots=not args.save_plots, print_data=True, save_path=save_path_1)

    print()
    print("--- Sweep 2: vary shape (Bielik-1.5B architecture) ---")
    bench_bielik_shapes.run(show_plots=not args.save_plots, print_data=True, save_path=save_path_2)

    print_summary(save_path=save_path_1 if args.save_plots else "")

    if args.save_plots:
        print()
        print(f"Plots saved to {plot_dir.absolute()}/")
        print(f"  - matmul-tflops-vs-size.png")
        print(f"  - matmul-tflops-vs-shape-bielik.png")
        print(f"  - matmul-summary-bielik-shapes-latency_ms.png")
        print(f"  - matmul-summary-bielik-shapes-tflops.png")
