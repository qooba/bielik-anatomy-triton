#!/usr/bin/env python3
"""
RoPE Cached Benchmark: Triton kernel vs PyTorch naive vs PyTorch naive + torch.compile.

Usage:
  python benchmark_rope_cached.py
  python benchmark_rope_cached.py --save-plots  # save PNG files
"""

import sys
import argparse
from pathlib import Path

import torch
import triton
import triton.testing

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from benchmarks.common.plotting import plot_summary_comparison

from kernels.attention.rope_cached import (
    rope_cached_kernel,
    apply_rope_cached_,
    build_rope_cache,
)

HEAD_DIM    = 128
THETA_BASE  = 1_000_000.0
MAX_SEQ_LEN = 8192

_cos_cache, _sin_cache = build_rope_cache(MAX_SEQ_LEN, HEAD_DIM, THETA_BASE, device="cuda")

def pytorch_naive(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Out-of-place split-scheme RoPE in plain PyTorch."""
    half = x.shape[-1] // 2
    cos = cos_cache[positions][None, :, None, :]
    sin = sin_cache[positions][None, :, None, :]
    x_first  = x[..., :half]
    x_second = x[..., half:]
    out_first  = x_first * cos - x_second * sin
    out_second = x_first * sin + x_second * cos
    return torch.cat([out_first, out_second], dim=-1)


_pytorch_compiled = torch.compile(pytorch_naive)

def _warmup_triton(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> None:
    """Pre-compile Triton kernel so compilation is excluded from timing."""
    batch, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    x_2d = x.view(-1, head_dim)
    n_tokens = x_2d.shape[0]
    stride_token = x_2d.stride(0)

    if positions.ndim == 1:
        pos_stride_batch = 0
        pos_stride_seq = positions.stride(0)
    else:
        pos_stride_batch = positions.stride(0)
        pos_stride_seq = positions.stride(1)

    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    rope_cached_kernel.warmup(
        x_2d, positions, cos_cache, sin_cache,
        stride_token, pos_stride_batch, pos_stride_seq, seq_len, num_heads, half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        grid=(n_tokens,),
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["triton", "pytorch_naive", "pytorch_compile"],
        line_names=["Triton (cached)", "PyTorch Naive", "PyTorch + compile"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Bandwidth (GB/s)",
        plot_name="rope-cached-bandwidth-vs-seq-len",
        args={"batch": 2, "num_heads": 12, "dtype": torch.float32},
    )
)
def bench_seq_len(seq_len, provider, batch, num_heads, dtype):
    x = torch.randn(batch, seq_len, num_heads, HEAD_DIM, device="cuda", dtype=dtype)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32)

    if provider == "triton":
        _warmup_triton(x, positions, _cos_cache, _sin_cache)
        x_triton = x.clone()
        fn = lambda: apply_rope_cached_(x_triton, positions, _cos_cache, _sin_cache)
    elif provider == "pytorch_naive":
        fn = lambda: pytorch_naive(x, positions, _cos_cache, _sin_cache)
    else:  # pytorch_compile
        # warmup compile
        _pytorch_compiled(x, positions, _cos_cache, _sin_cache)
        fn = lambda: _pytorch_compiled(x, positions, _cos_cache, _sin_cache)

    fn()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    gbps = lambda t: (3 * x.numel() * x.element_size()) / (t * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[1, 2, 4, 6, 8, 12, 16, 32],
        line_arg="provider",
        line_vals=["triton", "pytorch_naive", "pytorch_compile"],
        line_names=["Triton (cached)", "PyTorch Naive", "PyTorch + compile"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="Bandwidth (GB/s)",
        plot_name="rope-cached-bandwidth-vs-heads",
        args={"batch": 2, "seq_len": 1024, "dtype": torch.float32},
    )
)
def bench_num_heads(num_heads, provider, batch, seq_len, dtype):
    x = torch.randn(batch, seq_len, num_heads, HEAD_DIM, device="cuda", dtype=dtype)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32)

    if provider == "triton":
        _warmup_triton(x, positions, _cos_cache, _sin_cache)
        x_triton = x.clone()
        fn = lambda: apply_rope_cached_(x_triton, positions, _cos_cache, _sin_cache)
    elif provider == "pytorch_naive":
        fn = lambda: pytorch_naive(x, positions, _cos_cache, _sin_cache)
    else:  # pytorch_compile
        _pytorch_compiled(x, positions, _cos_cache, _sin_cache)
        fn = lambda: _pytorch_compiled(x, positions, _cos_cache, _sin_cache)

    fn()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    gbps = lambda t: (3 * x.numel() * x.element_size()) / (t * 1e6)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def print_summary(save_path=""):
    dtype = torch.float32
    batch = 2

    test_cases = [
        (64,   12, "seq=64,  Q-heads"),
        (128,  12, "seq=128, Q-heads"),
        (512,  12, "seq=512, Q-heads"),
        (1024, 12, "seq=1024,Q-heads"),
        (512,   2, "seq=512, KV-heads"),
        (1024,  2, "seq=1024,KV-heads"),
    ]

    print()
    print("=" * 80)
    print(f"  RoPE Cached — Bielik config  head_dim={HEAD_DIM}  theta={THETA_BASE:.0e}  fp32")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    print(f"  {'config':<22} {'Provider':<22} {'ms':>8} {'GB/s':>10} {'TFLOPS':>10}")
    print("-" * 80)

    data_dict = {
        "Triton (cached)": [],
        "PyTorch Naive": [],
        "PyTorch + compile": [],
    }
    labels = []

    for seq_len, num_heads, label in test_cases:
        labels.append(label)
        x = torch.randn(batch, seq_len, num_heads, HEAD_DIM, device="cuda", dtype=dtype)
        positions = torch.arange(seq_len, device="cuda", dtype=torch.int32)
        _warmup_triton(x, positions, _cos_cache, _sin_cache)
        # warmup compile for this shape
        _pytorch_compiled(x, positions, _cos_cache, _sin_cache)

        x_triton = x.clone()

        cases = [
            ("Triton (cached)  ", "Triton (cached)",
             lambda: apply_rope_cached_(x_triton, positions, _cos_cache, _sin_cache)),
            ("PyTorch Naive    ", "PyTorch Naive",
             lambda: pytorch_naive(x, positions, _cos_cache, _sin_cache)),
            ("PyTorch + compile", "PyTorch + compile",
             lambda: _pytorch_compiled(x, positions, _cos_cache, _sin_cache)),
        ]

        for display_name, dict_key, fn in cases:
            ms = triton.testing.do_bench(fn)
            gbps  = (3 * x.numel() * x.element_size()) / (ms * 1e6)
            tflops = (3 * x.numel()) / (ms * 1e9)
            data_dict[dict_key].append([gbps, tflops])
            print(f"  {label:<22} {display_name:<22} {ms:>8.3f} {gbps:>10.1f} {tflops:>10.4f}")
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["GB/s", "TFLOPS"],
            title=f"RoPE Cached Performance — Bielik Config (batch={batch}, head_dim={HEAD_DIM})",
            xlabel="Config",
            save_path=save_path,
            filename_prefix="rope_cached-summary-bielik-config",
            gpu_name=torch.cuda.get_device_name(0),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoPE cached benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    print("=" * 80)
    print("RoPE Cached Benchmark: Triton vs PyTorch Naive vs PyTorch + compile")
    print("=" * 80)
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

    print("--- Sweep 1: vary seq_len (batch=2, heads=12, fp32) ---")
    bench_seq_len.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print()
    print("--- Sweep 2: vary num_heads (batch=2, seq_len=512, fp32) ---")
    bench_num_heads.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print_summary(save_path=save_path if args.save_plots else "")

    if args.save_plots:
        print()
        print(f"Plots saved to {plot_dir.absolute()}/")
        print(f"  - rope-cached-bandwidth-vs-seq-len.png")
        print(f"  - rope-cached-bandwidth-vs-heads.png")
        print(f"  - rope_cached-summary-bielik-config-gb_s.png")
        print(f"  - rope_cached-summary-bielik-config-tflops.png")
