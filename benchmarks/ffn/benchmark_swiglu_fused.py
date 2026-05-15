#!/usr/bin/env python3
"""
Fused SwiGLU FFN Benchmark: Triton fused vs PyTorch compiled vs PyTorch unfused.

Usage:
  python benchmark_swiglu_fused.py
  python benchmark_swiglu_fused.py --save-plots
  python benchmark_swiglu_fused.py --save-plots --plot-dir ./results
"""

import json
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from common.plotting import plot_summary_comparison

from kernels.ffn.swiglu_fused import fused_ffn_swiglu, fused_ffn_swiglu_kernel

# Bielik-1.5B config
HIDDEN_SIZE = 1536
INTER_SIZE  = 8960

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_M = 8


def pytorch_unfused(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor) -> torch.Tensor:
    """Unfused - three separate kernel launches: matmul, matmul, silu+mul."""
    gate = x @ W_gate
    up   = x @ W_up
    return F.silu(gate) * up


@torch.compile
def _swiglu_compiled_fn(x, W_gate, W_up):
    return F.silu(x @ W_gate) * (x @ W_up)

def pytorch_compiled(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor) -> torch.Tensor:
    return _swiglu_compiled_fn(x, W_gate, W_up)

def _warmup_triton(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor,
                   output: torch.Tensor) -> None:
    M, K = x.shape
    N    = W_gate.shape[1]
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    fused_ffn_swiglu_kernel.warmup(
        x, W_gate, W_up, output,
        M, N, K,
        x.stride(0),      x.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        grid=grid,
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "pytorch_compiled", "pytorch_unfused"],
        line_names=["Triton (fused)", "PyTorch Compiled", "PyTorch Unfused"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOPS",
        plot_name="swiglu-ffn-tflops-vs-seq-len",
        args={"batch": 1, "hidden_size": HIDDEN_SIZE, "inter_size": INTER_SIZE,
              "dtype": torch.bfloat16},
    )
)
def bench_seq_len(seq_len, provider, batch, hidden_size, inter_size, dtype):
    M      = batch * seq_len
    x      = torch.randn(M, hidden_size, device="cuda", dtype=dtype)
    W_gate = torch.randn(hidden_size, inter_size, device="cuda", dtype=dtype)
    W_up   = torch.randn(hidden_size, inter_size, device="cuda", dtype=dtype)
    output = torch.empty(M, inter_size, device="cuda", dtype=dtype)

    if provider == "triton":
        _warmup_triton(x, W_gate, W_up, output)
        fn = lambda: fused_ffn_swiglu(x, W_gate, W_up)
    elif provider == "pytorch_compiled":
        pytorch_compiled(x, W_gate, W_up)  # warmup compilation
        fn = lambda: pytorch_compiled(x, W_gate, W_up)
    else:
        fn = lambda: pytorch_unfused(x, W_gate, W_up)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    tflops = lambda t: (4 * M * hidden_size * inter_size) / (t * 1e9)
    stats = {
        "provider": provider, "seq_len": seq_len, "batch": batch,
        "hidden_size": hidden_size, "inter_size": inter_size,
        "ms": ms, "min_ms": min_ms, "max_ms": max_ms,
        "tflops": tflops(ms), "min_tflops": tflops(max_ms), "max_tflops": tflops(min_ms),
    }
    with open("swiglu_ffn_bench_seq_len.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")
    return tflops(ms), tflops(max_ms), tflops(min_ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["inter_size"],
        x_vals=[1024, 2048, 4096, 8960, 11008, 14336],
        line_arg="provider",
        line_vals=["triton", "pytorch_compiled", "pytorch_unfused"],
        line_names=["Triton (fused)", "PyTorch Compiled", "PyTorch Unfused"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOPS",
        plot_name="swiglu-ffn-tflops-vs-inter-size",
        args={"n_rows": 512, "hidden_size": HIDDEN_SIZE, "dtype": torch.bfloat16},
    )
)
def bench_inter_size(inter_size, provider, n_rows, hidden_size, dtype):
    x      = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    W_gate = torch.randn(hidden_size, inter_size, device="cuda", dtype=dtype)
    W_up   = torch.randn(hidden_size, inter_size, device="cuda", dtype=dtype)
    output = torch.empty(n_rows, inter_size, device="cuda", dtype=dtype)

    if provider == "triton":
        _warmup_triton(x, W_gate, W_up, output)
        fn = lambda: fused_ffn_swiglu(x, W_gate, W_up)
    elif provider == "pytorch_compiled":
        pytorch_compiled(x, W_gate, W_up)
        fn = lambda: pytorch_compiled(x, W_gate, W_up)
    else:
        fn = lambda: pytorch_unfused(x, W_gate, W_up)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    tflops = lambda t: (4 * n_rows * hidden_size * inter_size) / (t * 1e9)
    stats = {
        "provider": provider, "inter_size": inter_size, "n_rows": n_rows,
        "hidden_size": hidden_size,
        "ms": ms, "min_ms": min_ms, "max_ms": max_ms,
        "tflops": tflops(ms), "min_tflops": tflops(max_ms), "max_tflops": tflops(min_ms),
    }
    with open("swiglu_ffn_bench_inter_size.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def print_summary(save_path=""):
    dtype       = torch.bfloat16
    hidden_size = HIDDEN_SIZE
    inter_size  = INTER_SIZE
    n_rows_list = [1, 8, 32, 128, 512, 1024]

    W_gate = torch.randn(hidden_size, inter_size, device="cuda", dtype=dtype)
    W_up   = torch.randn(hidden_size, inter_size, device="cuda", dtype=dtype)

    print()
    print("=" * 82)
    print(f"  SwiGLU FFN - Bielik config: hidden={hidden_size}, inter={inter_size}, bf16")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 82)
    print(f"  {'n_rows':<8} {'Provider':<22} {'ms':>8} {'GB/s':>10} {'TFLOPS':>10}")
    print("-" * 82)

    data_dict = {
        "Triton (fused)":   [],
        "PyTorch Compiled": [],
        "PyTorch Unfused":  [],
    }
    labels = [str(n) for n in n_rows_list]

    for n_rows in n_rows_list:
        x      = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
        output = torch.empty(n_rows, inter_size,  device="cuda", dtype=dtype)
        _warmup_triton(x, W_gate, W_up, output)
        pytorch_compiled(x, W_gate, W_up)  # warmup compilation

        nbytes = (n_rows * hidden_size + 2 * hidden_size * inter_size
                  + n_rows * inter_size) * x.element_size()
        flops  = 4 * n_rows * hidden_size * inter_size  # two matmuls

        cases = [
            ("Triton (fused)  ", "Triton (fused)",   lambda: fused_ffn_swiglu(x, W_gate, W_up)),
            ("PyTorch Compiled", "PyTorch Compiled",  lambda: pytorch_compiled(x, W_gate, W_up)),
            ("PyTorch Unfused ", "PyTorch Unfused",   lambda: pytorch_unfused(x, W_gate, W_up)),
        ]

        for display_name, dict_key, fn in cases:
            ms     = triton.testing.do_bench(fn)
            gbps   = nbytes / (ms * 1e6)
            tflops = flops  / (ms * 1e9)
            data_dict[dict_key].append([gbps, tflops])
            print(f"  {n_rows:<8} {display_name:<22} {ms:>8.3f} {gbps:>10.1f} {tflops:>10.4f}")
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["GB/s", "TFLOPS"],
            title="SwiGLU FFN Performance - Bielik Config",
            xlabel="n_rows (batch x seq_len)",
            save_path=save_path,
            filename_prefix="swiglu_ffn-summary-bielik-config",
            gpu_name=torch.cuda.get_device_name(0),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused SwiGLU FFN benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    print("=" * 82)
    print("SwiGLU FFN Benchmark: Triton Fused vs PyTorch Compiled vs PyTorch Unfused")
    print("=" * 82)
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

    print("--- Sweep 1: vary seq_len (batch=1, hidden=1536, inter=8960, bf16) ---")
    bench_seq_len.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print()
    print("--- Sweep 2: vary inter_size (n_rows=512, hidden=1536, bf16) ---")
    bench_inter_size.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print_summary(save_path=save_path if args.save_plots else "")

    if args.save_plots:
        print()
        print(f"Plots saved to {plot_dir.absolute()}/")
        print(f"  - swiglu-ffn-tflops-vs-seq-len.png")
        print(f"  - swiglu-ffn-tflops-vs-inter-size.png")
        print(f"  - swiglu_ffn-summary-bielik-config-gb_s.png")
        print(f"  - swiglu_ffn-summary-bielik-config-tflops.png")
