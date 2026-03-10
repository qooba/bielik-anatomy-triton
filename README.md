# Bielik Anatomy - Building a Polish LLM from Scratch with Triton GPU Kernels

A hands-on video series where we implement the Polish language model **Bielik 1.5B** from scratch using custom GPU kernels written in [Triton](https://triton-lang.org/). Every component — from matrix multiplication to text generation — is built step by step, optimized, and benchmarked against PyTorch.

**Model:** [Bielik-1.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) (1.6B parameters, Polish)

---

## Series Overview

| # | Episode | Key Result | Doc |
|---|---------|------------|-----|
| 01 | [Introduction — Bielik Architecture and Triton](/docs/ep01-introduction.md) | Architecture overview, GQA, SwiGLU, why Triton | [link](/docs/ep01-introduction.md) |
| 02 | [Matmul — Heart of the Transformer](/docs/ep02-matmul.md) | Tiled matmul with Tensor Cores, matching PyTorch perf | [link](/docs/ep02-matmul.md) |
| 03 | [Just Fuse It - Fused RMSNorm & Softmax](/docs/ep03-rmsnorm-softmax-fused.md) | Fused single-pass RMSNorm and Softmax with casual mask  | [link](/docs/ep03-rmsnorm-softmax-fused.md) |
---

## What You Will Learn

- How transformers work at the GPU instruction level
- Writing high-performance Triton kernels from scratch
- Tiling, Tensor Cores, kernel fusion, auto-tuning

## Prerequisites

- Python and basic ML/neural network knowledge
- General idea of how transformers work (helpful but not required)
- An NVIDIA GPU with CUDA support


## Project Structure

```
embers/
├── kernels/                 # Triton GPU kernels
│   ├── matmul/              #   Matrix multiplication variants
├── benchmarks/              # Performance benchmarks
│   ├── matmul/              #   Bechmarks for matmul kernels
└── docs/                    # Episodes docs
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/qooba/bielik-anatomy-triton
cd bielik-anatomy-triton

# Install dependencies
pip install -r requirements.txt

```
