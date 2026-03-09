#!/usr/bin/env python3
"""
RMSNorm Benchmark: Triton fused vs PyTorch native vs PyTorch unfused.

Usage:
  python benchmark_rms_norm.py
  python benchmark_rms_norm.py --save-plots  # save PNG files
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.testing

from benchmarks.common.plotting import plot_summary_comparison
from kernels.normalization.rms_norm_simple import rms_norm_simple, rms_norm_simple_kernel

EPS = 1e-6


def pytorch_unfused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Unfused RMSNorm - separate PyTorch ops, 5 kernel launches."""
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x / rms) * weight


def _warmup_triton(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Pre-compile Triton kernel so compilation is excluded from timing."""
    hidden_size = x.shape[-1]
    n_rows = x.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    output = torch.empty_like(x)
    rms_norm_simple_kernel.warmup(
        x,
        weight,
        output,
        x.stride(0),
        output.stride(0),
        hidden_size,
        EPS,
        BLOCK_SIZE=BLOCK_SIZE,
        grid=(n_rows,),
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size"],
        x_vals=[128, 256, 512, 768, 1024, 1536, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "pytorch_native", "pytorch_unfused"],
        line_names=["Triton (fused)", "PyTorch RMSNorm", "PyTorch Unfused"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Bandwidth (GB/s)",
        plot_name="rmsnorm-bandwidth-vs-hidden-size",
        args={"n_rows": 512, "dtype": torch.float32},
    )
)
def bench_hidden_size(hidden_size, provider, n_rows, dtype):
    """Sweep 1: vary hidden_size (n_rows=512 fixed)"""
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.ones(hidden_size, device="cuda", dtype=dtype)

    if provider == "triton":
        _warmup_triton(x, weight)
        fn = lambda: rms_norm_simple(x, weight, eps=EPS)
    elif provider == "pytorch_native":
        layer = nn.RMSNorm(hidden_size, eps=EPS, device="cuda", dtype=dtype)
        fn = lambda: layer(x)
    else:  # pytorch_unfused
        fn = lambda: pytorch_unfused(x, weight)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    gbps = lambda t: (2 * x.numel() * x.element_size()) / (t * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows"],
        x_vals=[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["triton", "pytorch_native", "pytorch_unfused"],
        line_names=["Triton (fused)", "PyTorch RMSNorm", "PyTorch Unfused"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Bandwidth (GB/s)",
        plot_name="rmsnorm-bandwidth-vs-rows",
        args={"hidden_size": 1536, "dtype": torch.float32},
    )
)
def bench_n_rows(n_rows, provider, hidden_size, dtype):
    """Sweep 2: vary n_rows (batch*seq_len), hidden_size=1536 fixed)"""
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.ones(hidden_size, device="cuda", dtype=dtype)

    if provider == "triton":
        _warmup_triton(x, weight)
        fn = lambda: rms_norm_simple(x, weight, eps=EPS)
    elif provider == "pytorch_native":
        layer = nn.RMSNorm(hidden_size, eps=EPS, device="cuda", dtype=dtype)
        fn = lambda: layer(x)
    else:  # pytorch_unfused
        fn = lambda: pytorch_unfused(x, weight)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    gbps = lambda t: (2 * x.numel() * x.element_size()) / (t * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def print_summary(hidden_sizes=None, n_rows_list=None, save_path=""):
    """Manual summary table with latency + GB/s + TFLOPS"""
    if hidden_sizes is None:
        hidden_sizes = [512, 1024, 1536, 2048, 4096]
    if n_rows_list is None:
        n_rows_list = [1, 32, 128, 512]

    dtype = torch.float32

    print()
    print("=" * 75)
    print("  Bielik config: hidden_size=1536, dtype=float32, GPU:", torch.cuda.get_device_name(0))
    print("=" * 75)
    print(f"  {'n_rows':<8} {'Provider':<22} {'ms':>8} {'GB/s':>10} {'TFLOPS':>10}")
    print("-" * 75)

    hidden_size = 1536
    weight = torch.ones(hidden_size, device="cuda", dtype=dtype)
    layer = nn.RMSNorm(hidden_size, eps=EPS, device="cuda", dtype=dtype)

    data_dict = {
        "Triton (fused)": [],
        "PyTorch RMSNorm": [],
        "PyTorch Unfused": [],
    }
    labels = [str(n) for n in n_rows_list]

    for n_rows in n_rows_list:
        x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
        _warmup_triton(x, weight)

        cases = [
            ("Triton (fused)  ", "Triton (fused)", lambda: rms_norm_simple(x, weight, eps=EPS)),
            ("PyTorch RMSNorm ", "PyTorch RMSNorm", lambda: layer(x)),
            ("PyTorch Unfused ", "PyTorch Unfused", lambda: pytorch_unfused(x, weight)),
        ]

        for display_name, dict_key, fn in cases:
            ms = triton.testing.do_bench(fn)
            gbps = (2 * x.numel() * x.element_size()) / (ms * 1e6)
            tflops = (4 * n_rows * hidden_size) / (ms * 1e9)
            data_dict[dict_key].append([gbps, tflops])
            print(f"  {n_rows:<8} {display_name:<22} {ms:>8.3f} {gbps:>10.1f} {tflops:>10.4f}")
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["GB/s", "TFLOPS"],
            title="RMSNorm Performance - Bielik Config (hidden_size=1536)",
            xlabel="Batch Size (n_rows)",
            save_path=save_path,
            filename_prefix="rms_norm-summary-bielik-config",
            gpu_name=torch.cuda.get_device_name(0),
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMSNorm benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    print("=" * 75)
    print("RMSNorm Benchmark: Triton vs PyTorch Native vs PyTorch Unfused")
    print("=" * 75)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.save_plots:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plot_dir)
        print(f"Saving plots to: {plot_dir.absolute()}/")
        print()
    else:
        save_path = ""

    print("--- Sweep 1: vary hidden_size (n_rows=512, fp32) ---")
    bench_hidden_size.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print()
    print("--- Sweep 2: vary n_rows / batch*seq_len (hidden_size=1536, fp32) ---")
    bench_n_rows.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print_summary(save_path=save_path if args.save_plots else "")

    if args.save_plots:
        print()
        print(f"Plots saved to {plot_dir.absolute()}/")
        print(f"  - rms_norm-bandwidth-vs-hidden_size.png")
        print(f"  - rms_norm-bandwidth-vs-n_rows.png")
        print(f"  - rms_norm-summary-bielik-config-gb_s.png")
        print(f"  - rms_norm-summary-bielik-config-tflops.png")
