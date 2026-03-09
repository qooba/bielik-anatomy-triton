#!/usr/bin/env python3
"""
Causal Softmax Benchmark: Triton fused vs PyTorch native vs PyTorch unfused.

Usage:
  python benchmark_softmax_causal.py
  python benchmark_softmax_causal.py --save-plots  # save PNG files
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing

from benchmarks.common.plotting import plot_summary_comparison
from kernels.attention.softmax_causal_simple import (
    softmax_causal_simple,
    softmax_causal_simple_kernel,
)


def pytorch_native(x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """PyTorch native: F.softmax with pre-computed causal mask."""
    return F.softmax(x + causal_mask, dim=-1)


def pytorch_unfused(x: torch.Tensor) -> torch.Tensor:
    """Unfused softmax: each op is a separate kernel launch (mask rebuilt every call)."""
    seq_len = x.shape[-1]
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
        diagonal=1,
    )
    x_masked = x + mask
    row_max = x_masked.max(dim=-1, keepdim=True).values
    shifted = x_masked - row_max
    exp_vals = shifted.exp()
    denom = exp_vals.sum(dim=-1, keepdim=True)
    return exp_vals / denom


def _warmup_triton(x: torch.Tensor) -> None:
    """Pre-compile Triton kernel so compilation is excluded from timing."""
    batch, num_heads, seq_len, _ = x.shape
    x_2d = x.reshape(-1, seq_len)
    n_rows = x_2d.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    output = torch.empty_like(x_2d)
    softmax_causal_simple_kernel.warmup(
        x_2d,
        output,
        n_rows,
        seq_len,
        seq_len,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        grid=(n_rows,),
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "pytorch_native", "pytorch_unfused"],
        line_names=["Triton (fused)", "PyTorch Softmax", "PyTorch Unfused"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Bandwidth (GB/s)",
        plot_name="softmax-causal-bandwidth-vs-seq-len",
        args={"batch": 2, "num_heads": 12, "dtype": torch.float32},
    )
)
def bench_seq_len(seq_len, provider, batch, num_heads, dtype):
    """
    Sweep 1: vary sequence length (batch=2, heads=12 fixed).
    """
    x = torch.randn(batch, num_heads, seq_len, seq_len, device="cuda", dtype=dtype)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device="cuda", dtype=dtype),
        diagonal=1,
    )

    if provider == "triton":
        _warmup_triton(x)
        fn = lambda: softmax_causal_simple(x)
    elif provider == "pytorch_native":
        fn = lambda: pytorch_native(x, causal_mask)
    else:  # pytorch_unfused
        fn = lambda: pytorch_unfused(x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    gbps = lambda t: (2 * x.numel() * x.element_size()) / (t * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[1, 2, 4, 6, 8, 12, 16, 32],
        line_arg="provider",
        line_vals=["triton", "pytorch_native", "pytorch_unfused"],
        line_names=["Triton (fused)", "PyTorch Softmax", "PyTorch Unfused"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Bandwidth (GB/s)",
        plot_name="softmax-causal-bandwidth-vs-heads",
        args={"batch": 2, "seq_len": 128, "dtype": torch.float32},
    )
)
def bench_num_heads(num_heads, provider, batch, seq_len, dtype):
    """Sweep 2: vary number of heads (batch=2, seq_len=128 fixed)."""
    x = torch.randn(batch, num_heads, seq_len, seq_len, device="cuda", dtype=dtype)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device="cuda", dtype=dtype),
        diagonal=1,
    )

    if provider == "triton":
        _warmup_triton(x)
        fn = lambda: softmax_causal_simple(x)
    elif provider == "pytorch_native":
        fn = lambda: pytorch_native(x, causal_mask)
    else:  # pytorch_unfused
        fn = lambda: pytorch_unfused(x)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    gbps = lambda t: (2 * x.numel() * x.element_size()) / (t * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def print_summary(save_path=""):
    """Print summary of all results and save summary plots."""
    dtype = torch.float32
    batch = 2
    num_heads = 12  # Bielik Q heads

    seq_lens = [64, 128, 256, 512, 1024]

    print()
    print("=" * 75)
    print(f"  Bielik config: batch={batch}, heads={num_heads}, dtype=float32")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 75)
    print(f"  {'seq_len':<10} {'Provider':<22} {'ms':>8} {'GB/s':>10} {'TFLOPS':>10}")
    print("-" * 75)

    data_dict = {
        "Triton (fused)": [],
        "PyTorch Softmax": [],
        "PyTorch Unfused": [],
    }
    labels = [str(s) for s in seq_lens]

    for seq_len in seq_lens:
        x = torch.randn(batch, num_heads, seq_len, seq_len, device="cuda", dtype=dtype)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device="cuda", dtype=dtype),
            diagonal=1,
        )
        _warmup_triton(x)

        cases = [
            ("Triton (fused)  ", "Triton (fused)", lambda: softmax_causal_simple(x)),
            ("PyTorch Softmax ", "PyTorch Softmax", lambda: pytorch_native(x, causal_mask)),
            ("PyTorch Unfused ", "PyTorch Unfused", lambda: pytorch_unfused(x)),
        ]

        for display_name, dict_key, fn in cases:
            ms = triton.testing.do_bench(fn)
            gbps = (2 * x.numel() * x.element_size()) / (ms * 1e6)
            tflops = (5 * x.numel()) / (ms * 1e9)
            data_dict[dict_key].append([gbps, tflops])
            print(f"  {seq_len:<10} {display_name:<22} {ms:>8.3f} {gbps:>10.1f} {tflops:>10.4f}")
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["GB/s", "TFLOPS"],
            title=f"Causal Softmax Performance - Bielik Config (batch={batch}, heads={num_heads})",
            xlabel="Sequence Length",
            save_path=save_path,
            filename_prefix="softmax_causal-summary-bielik-config",
            gpu_name=torch.cuda.get_device_name(0),
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal softmax benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    print("=" * 75)
    print("Causal Softmax Benchmark: Triton vs PyTorch Native vs PyTorch Unfused")
    print("=" * 75)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Determine save path
    if args.save_plots:
        from pathlib import Path

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plot_dir)
        print(f"Saving plots to: {plot_dir.absolute()}/")
        print()
    else:
        save_path = ""

    print("--- Sweep 1: vary seq_len (batch=2, heads=12, fp32) ---")
    bench_seq_len.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print()
    print("--- Sweep 2: vary num_heads (batch=2, seq_len=128, fp32) ---")
    bench_num_heads.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print_summary(save_path=save_path if args.save_plots else "")

    if args.save_plots:
        print()
        print(f"Plots saved to {plot_dir.absolute()}/")
        print(f"  - softmax_causal-bandwidth-vs-seq_len.png")
        print(f"  - softmax_causal-bandwidth-vs-num_heads.png")
        print(f"  - softmax_causal-summary-bielik-config-gb_s.png")
        print(f"  - softmax_causal-summary-bielik-config-tflops.png")
