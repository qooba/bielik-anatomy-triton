#!/usr/bin/env python3
"""
Flash Attention Benchmark: Triton vs PyTorch naive vs PyTorch naive + torch.compile.

Usage:
  python benchmark_flash_attention.py
  python benchmark_flash_attention.py --save-plots  # save PNG files
  python benchmark_flash_attention.py --save-plots --plot-dir ./results
"""
import json
import sys
import argparse
from pathlib import Path

import torch
import triton
import triton.testing

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from common.plotting import plot_summary_comparison

from kernels.attention.flash_attention_simple import flash_attention_forward


def pytorch_naive(q, k, v, causal=True):
    batch, num_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]

    if num_heads != num_kv_heads:
        group = num_heads // num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    scale = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask[None, None], float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

_compiled_naive = torch.compile(pytorch_naive)

def pytorch_compiled(q, k, v, causal=True):
    return _compiled_naive(q, k, v, causal)


def _warmup_triton(q, k, v):
    flash_attention_forward(q, k, v, causal=True)
    torch.cuda.synchronize()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "pytorch_naive", "pytorch_compiled"],
        line_names=["Triton (flash)", "PyTorch Naive", "PyTorch Compiled"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOPS",
        plot_name="flash-attention-tflops-vs-seq-len",
        args={
            "batch": 1,
            "num_heads": 12,
            "num_kv_heads": 2,
            "head_dim": 128,
            "dtype": torch.bfloat16,
        },
    )
)
def bench_seq_len(seq_len, provider, batch, num_heads, num_kv_heads, head_dim, dtype):
    q = torch.randn(batch, num_heads,    seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    if provider == "triton":
        _warmup_triton(q, k, v)
        fn = lambda: flash_attention_forward(q, k, v, causal=True)
    elif provider == "pytorch_naive":
        fn = lambda: pytorch_naive(q, k, v, causal=True)
    else:  # pytorch_compiled
        pytorch_compiled(q, k, v, causal=True)
        torch.cuda.synchronize()
        fn = lambda: pytorch_compiled(q, k, v, causal=True)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    tflops = lambda t: (4 * batch * num_heads * seq_len ** 2 * head_dim) / (t * 1e9)
    stats = { "provider": provider, "seq_len": seq_len, "ms": ms, "min_ms": min_ms, "max_ms": max_ms, "tflops": tflops(ms), "min_tflops": tflops(max_ms), "max_tflops": tflops(min_ms) }
    with open("flash_attention_bench_seq_len.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
    return tflops(ms), tflops(max_ms), tflops(min_ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[1, 2, 4, 6, 8, 12, 16],
        line_arg="provider",
        line_vals=["triton", "pytorch_naive", "pytorch_compiled"],
        line_names=["Triton (flash)", "PyTorch Naive", "PyTorch Compiled"],
        styles=[("blue", "-"), ("green", "--"), ("red", ":")],
        ylabel="TFLOPS",
        plot_name="flash-attention-tflops-vs-heads",
        args={
            "batch": 1,
            "seq_len": 512,
            "head_dim": 128,
            "dtype": torch.bfloat16,
        },
    )
)
def bench_num_heads(num_heads, provider, batch, seq_len, head_dim, dtype):
    num_kv_heads = max(1, num_heads // 6)

    q = torch.randn(batch, num_heads,    seq_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    if provider == "triton":
        _warmup_triton(q, k, v)
        fn = lambda: flash_attention_forward(q, k, v, causal=True)
    elif provider == "pytorch_naive":
        fn = lambda: pytorch_naive(q, k, v, causal=True)
    else:  # pytorch_compiled
        pytorch_compiled(q, k, v, causal=True)
        torch.cuda.synchronize()
        fn = lambda: pytorch_compiled(q, k, v, causal=True)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    tflops = lambda t: (4 * batch * num_heads * seq_len ** 2 * head_dim) / (t * 1e9)
    stats = { "provider": provider, "num_heads": num_heads, "ms": ms, "min_ms": min_ms, "max_ms": max_ms, "tflops": tflops(ms), "min_tflops": tflops(max_ms), "max_tflops": tflops(min_ms) }
    with open("flash_attention_bench_num_heads.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def _peak_memory_mb(fn) -> float | None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except torch.cuda.OutOfMemoryError:
        return None


def print_summary(save_path=""):
    dtype = torch.bfloat16
    batch = 1
    num_heads = 12
    num_kv_heads = 2
    head_dim = 128

    seq_lens = [128, 256, 512, 1024, 2048, 4096]

    print()
    print("=" * 95)
    print(f"  Bielik config: batch={batch}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, dtype={dtype}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 95)
    print(f"  {'seq_len':<8} {'Provider':<22} {'ms':>8} {'TFLOPS':>10} {'Speedup':>9} {'Peak mem':>12}")
    print("-" * 95)

    data_dict = {
        "Triton (flash)": [],
        "PyTorch Naive": [],
        "PyTorch Compiled": [],
    }
    labels = [str(s) for s in seq_lens]

    for seq_len in seq_lens:
        q = torch.randn(batch, num_heads,    seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)

        _warmup_triton(q, k, v)
        try:
            pytorch_compiled(q, k, v, causal=True)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            pass

        flop_count = 4 * batch * num_heads * seq_len ** 2 * head_dim

        cases = [
            ("Triton (flash)  ", "Triton (flash)",
             lambda: flash_attention_forward(q, k, v, causal=True)),
            ("PyTorch Naive   ", "PyTorch Naive",
             lambda: pytorch_naive(q, k, v, causal=True)),
            ("PyTorch Compiled", "PyTorch Compiled",
             lambda: pytorch_compiled(q, k, v, causal=True)),
        ]

        row_ms = {}
        for display_name, dict_key, fn in cases:
            try:
                ms = triton.testing.do_bench(fn)
                tflops = flop_count / (ms * 1e9)
                mem = _peak_memory_mb(fn)
                row_ms[dict_key] = ms
                data_dict[dict_key].append([tflops, ms])
            except torch.cuda.OutOfMemoryError:
                ms = tflops = mem = None
                row_ms[dict_key] = None
                data_dict[dict_key].append([0.0, 0.0])

            speedup = ""
            if dict_key != "Triton (flash)" and row_ms.get("Triton (flash)") and ms:
                speedup = f"{ms / row_ms['Triton (flash)']:.2f}x"
            elif dict_key == "Triton (flash)":
                speedup = "—"

            mem_str = f"{mem:.0f} MB" if mem is not None else "OOM"
            ms_str  = f"{ms:.3f}" if ms is not None else "OOM"
            tf_str  = f"{tflops:.4f}" if tflops is not None else "OOM"
            print(f"  {seq_len:<8} {display_name:<22} {ms_str:>8} {tf_str:>10} {speedup:>9} {mem_str:>12}")

        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["TFLOPS", "Latency (ms)"],
            title=f"Flash Attention - Bielik Config (batch={batch}, heads={num_heads}, kv_heads={num_kv_heads})",
            xlabel="Sequence Length",
            save_path=save_path,
            filename_prefix="flash_attention-summary-bielik-config",
            gpu_name=torch.cuda.get_device_name(0),
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print("=" * 75)
    print("Flash Attention Benchmark: Triton vs PyTorch Naive vs torch.compile")
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

    print("Warming up torch.compile … ", end="", flush=True)
    _q = torch.randn(1, 12, 128, 128, device="cuda", dtype=torch.bfloat16)
    _k = torch.randn(1,  2, 128, 128, device="cuda", dtype=torch.bfloat16)
    _v = torch.randn(1,  2, 128, 128, device="cuda", dtype=torch.bfloat16)
    try:
        pytorch_compiled(_q, _k, _v)
        torch.cuda.synchronize()
        print("done")
    except Exception as e:
        print(f"failed ({e})")
    print()

    print("--- Sweep 1: vary seq_len (batch=1, heads=12, kv_heads=2, head_dim=128, bf16) ---")
    bench_seq_len.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print()
    print("--- Sweep 2: vary num_heads (batch=1, seq_len=512, head_dim=128, bf16) ---")
    bench_num_heads.run(show_plots=not args.save_plots, print_data=True, save_path=save_path)

    print_summary(save_path=save_path if args.save_plots else "")

    if args.save_plots:
        print(f"Plots saved to {plot_dir.absolute()}/")
        print(f"  - flash-attention-tflops-vs-seq-len.png")
        print(f"  - flash-attention-tflops-vs-heads.png")
        print(f"  - flash_attention-summary-bielik-config-tflops.png")
        print(f"  - flash_attention-summary-bielik-config-latency_ms.png")
