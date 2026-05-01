# Episode 5: Flash Attention v2

[Back to Series Overview](README.md) | [Previous: RoPE](ep04-rope.md) | [Next: Feed-Forward Network](ep06-feed-forward-network.md)

---

<p align="center">
    <a href="https://www.youtube.com/watch?v=Npwu0puwQPw">
        <img src="https://img.youtube.com/vi/Npwu0puwQPw/sddefault.jpg" alt="Episode 5: Flash Attention" style="max-width: 100%;">
    </a>
</p>

## Overview

Standard attention materializes an O(N^2) score matrix in HBM, causing memory bottlenecks and out-of-memory errors on long sequences. Flash Attention keeps everything in fast on-chip SRAM using tiling and online softmax - never writing the full attention matrix to slow memory. 

## Relevant Code

### Kernels
- [`kernels/attention/flash_attention_simple.py`](/kernels/attention/flash_attention_simple.py) — Flash Attention kernel

### Benchmarks
- [`benchmarks/attention/benchmark_flash_attention.py`](/benchmarks/attention/benchmark_flash_attention.py) - Triton vs PyTorch naive vs PyTorch+compile; two sweeps (seq_len, num_heads)

### Results on my RTX 4060 Ti

<p align="center">
    <img src="/docs/plots/attention/flash-attention-tflops-vs-heads.png" alt="flash-attention-tflops-vs-heads" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/attention/flash-attention-tflops-vs-seq-len.png" alt="flash-attention-tflops-vs-seq-len" style="max-width: 100%;">
</p>
---

[Back to Series Overview](README.md) | [Previous: RoPE](ep04-rope.md) | [Next: Feed-Forward Network](ep06-feed-forward-network.md)
