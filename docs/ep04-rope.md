# Episode 4: RoPE — Rotary Position Embeddings

[Back to Series Overview](README.md) | [Previous: RMSNorm and Softmax](ep03-rmsnorm-softmax.md) | [Next: Flash Attention](ep05-flash-attention.md)

---

<p align="center">
    <a href="https://www.youtube.com/watch?v=dG-jPI02cY0">
        <img src="https://img.youtube.com/vi/dG-jPI02cY0/sddefault.jpg" alt="Episode 4: RoPE" style="max-width: 100%;">
    </a>
</p>

## Overview

Without position encoding, a Transformer treats its input as an unordered set - "Dog bites man" and "Man bites dog" produce identical attention scores for the same token pairs. This episode dives deep into **Rotary Position Embeddings (RoPE)**: the math behind rotating Q and K vectors, where and how RoPE is applied in Bielik's architecture, and how to build an optimized single-pass Triton kernel that eliminates all transcendental operations from the hot path using a precomputed cos/sin cache.

## Relevant Code

### Kernels
- [`kernels/attention/rope_cached.py`](/kernels/attention/rope_cached.py) - optimised single-pass cached RoPE kernel (`rope_cached_kernel`, `build_rope_cache`, `apply_rope_cached_`)

### Benchmarks
- [`benchmarks/attention/benchmark_rope_cached.py`](/benchmarks/attention/benchmark_rope_cached.py) - Triton vs PyTorch naive vs PyTorch+compile; two sweeps (seq_len, num_heads)

### Results on my RTX 4060 Ti

<p align="center">
    <img src="/docs/plots/attention/rope-cached-bandwidth-vs-heads.png" alt="rope-cached-bandwidth-vs-heads" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/attention/rope-cached-bandwidth-vs-seq-len.png" alt="rope-cached-bandwidth-vs-seq-len" style="max-width: 100%;">
</p>

---

[Back to Series Overview](README.md) | [Previous: RMSNorm and Softmax](ep03-rmsnorm-softmax.md) | [Next: Flash Attention](ep05-flash-attention.md)
