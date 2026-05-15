# Episode 6: Feed-Forward Network

[Back to Series Overview](README.md) | [Previous: Flash Attention](ep05-flash-attention.md) | [Next: Complete Decoder Layer](ep07-decoder-layer.md)

---

<p align="center">
    <a href="https://www.youtube.com/watch?v=zklv6OpoqWE">
        <img src="https://img.youtube.com/vi/zklv6OpoqWE/sddefault.jpg" alt="Episode 6: SwiGLU" style="max-width: 100%;">
    </a>
</p>

## Overview

This episode implements the SwiGLU activation, fuses the gate and up projections into a single kernel.

## Relevant Code

### Kernels
- [`kernels/ffn/swiglu_fused.py`](/kernels/ffn/swiglu_fused.py) — Swiglu fused kernel

### Benchmarks
- [`benchmarks/ffn/benchmark_swiglu_fused.py`](/benchmarks/ffn/benchmark_swiglu_fused.py) - Triton vs PyTorch unfused vs PyTorch+compile; two sweeps (seq_len, inter size)

### Results on my RTX 4060 Ti

<p align="center">
    <img src="/docs/plots/ffn/swiglu-ffn-tflops-vs-seq-len.png" alt="swiglu-ffn-tflops-vs-seq-len" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/ffn/swiglu-ffn-tflops-vs-inter-size.png" alt="swiglu-ffn-tflops-vs-inter-size" style="max-width: 100%;">
</p>
---

[Back to Series Overview](README.md) | [Previous: Flash Attention](ep05-flash-attention.md) | [Next: Complete Decoder Layer](ep07-decoder-layer.md)
