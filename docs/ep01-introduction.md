# Episode 1: Introduction — Bielik Architecture and Triton

[Back to Series Overview](/README.md) | [Next: Matmul - Heart of the Transformer](/docs/ep02-matmul.md)

---

[![Episode 1: Introduction](https://img.youtube.com/vi/kyM2HOAOrOM/sddefault.jpg)](https://www.youtube.com/watch?v=kyM2HOAOrOM)

## Overview

The opening episode of the Bielik Anatomy series. We look at the architecture of Bielik 1.5B - a Polish language model with 1.6 billion parameters - and explain why Triton is the tool of choice for writing custom GPU kernels.

## Topics Covered

- **What is Bielik** - a 1.6B-parameter Polish LLM based on a Qwen-like architecture, created by the SpeakLeash community
- **Architecture walkthrough** - embedding layer, 32 decoder layers, final RMSNorm, language model head
- **Grouped Query Attention (GQA)** - 12 query heads with only 2 KV heads, achieving 55% parameter reduction
- **SwiGLU activation** - the gated feed-forward network used in modern LLMs
- **Why Triton over CUDA** - Python-like syntax, automatic block management, compiler optimizations, cross-GPU portability
- **Triton vs PyTorch** - RMSNorm example: 1 fused kernel in Triton vs 3-4 separate kernels in PyTorch, 2-3x faster
- **Series roadmap** - basic kernels, attention, FFN, text generation

## Key Takeaways

- Bielik uses modern architectural improvements (GQA, SwiGLU, RMSNorm) that we will implement one by one
- Triton lets you write high-performance GPU kernels without the boilerplate of raw CUDA
- The series is a learning journey - building every component from scratch

## References

- [Bielik model on HuggingFace](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct)
- [Triton documentation](https://triton-lang.org/)
- [SpeakLeash community](https://speakleash.org/)

---

[Back to Series Overview](/README.md) | [Next: Matmul - Heart of the Transformer](/docs/ep02-matmul.md)
