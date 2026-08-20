# Bielik Anatomy - Building a Polish LLM from Scratch with Triton GPU Kernels

A hands-on video series where we implement the Polish language model **Bielik 1.5B** from scratch using custom GPU kernels written in [Triton](https://triton-lang.org/). Every component - from matrix multiplication to text generation - is built step by step, optimized, and benchmarked against PyTorch.

**Model:** [Bielik-1.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) (1.6B parameters, Polish)

---

## Series Overview

Each kernel episode has a companion Colab notebook — click the badge to open it and run the correctness check and benchmark on a free GPU, no local setup required. Episode 7 has a notebook too: it downloads the pretrained weights and runs the full Bielik model end-to-end.

| # | Episode | Key Result | Doc | Colab |
|---|---------|------------|-----|-------|
| 01 | [Introduction - Bielik Architecture and Triton](/docs/ep01-introduction.md) | Architecture overview, GQA, SwiGLU, why Triton | [link](/docs/ep01-introduction.md) | — |
| 02 | [Matmul - Heart of the Transformer](/docs/ep02-matmul.md) | Tiled matmul with Tensor Cores, matching PyTorch perf | [link](/docs/ep02-matmul.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep02-matmul.ipynb) |
| 03 | [Fused kernels - RMSNorm & Softmax](/docs/ep03-rmsnorm-softmax-fused.md) | Fused single-pass RMSNorm and Softmax with causal mask  | [link](/docs/ep03-rmsnorm-softmax-fused.md) | [RMSNorm](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep03-rmsnorm.ipynb) / [Softmax](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep03-softmax-causal.ipynb) |
| 04 | [RoPE](/docs/ep04-rope.md) | RoPE - Rotary Position Embedding | [link](/docs/ep04-rope.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep04-rope.ipynb) |
| 05 | [Flash Attention v2](/docs/ep05-flash-attention.md) | Flash Attention | [link](/docs/ep05-flash-attention.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep05-flash-attention.ipynb) |
| 06 | [SwiGLU FFN](/docs/ep06-feed-forward-network.md) | SwiGLU Feed Forward Network | [link](/docs/ep06-feed-forward-network.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep06-swiglu.ipynb) |
| 07 | [It's Alive](/docs/ep07-its-alive.md) | It's Alive - Let's talk with Bielik | [link](/docs/ep07-its-alive.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qooba/bielik-anatomy-triton/blob/main/notebooks/ep07-its-alive.ipynb) |
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
├── bielik/                  # Bielik Model using Trtion kernels
├── kernels/                 # Triton GPU kernels
│   ├── matmul/              #   Matrix multiplication variants
├── benchmarks/              # Performance benchmarks
│   ├── matmul/              #   Bechmarks for matmul kernels
├── notebooks/               # Colab notebooks - one per kernel (correctness check + benchmark)
└── docs/                    # Episodes docs
```

## Running Kernel Benchmarks on Colab (No GPU Required Locally)

Each kernel has a matching notebook in [`notebooks/`](/notebooks) with a correctness check against a plain PyTorch reference, followed by the full benchmark sweep from `benchmarks/`. Click a badge in the table above (or open a notebook directly) to run it on a free Colab GPU - just make sure to select `Runtime -> Change runtime type -> T4 GPU` first.

[`notebooks/ep07-its-alive.ipynb`](/notebooks/ep07-its-alive.ipynb) goes further: it downloads the pretrained Bielik-1.5B weights and runs the full model - built entirely from the Triton kernels above - to generate real Polish text, including an interactive chat cell.

Free-tier T4 GPUs are Turing (compute capability 7.5) and have no native BF16 tensor cores, so the matmul, flash-attention, SwiGLU, and full-model notebooks will run slower there than the BF16 results in the docs (measured on an RTX 4060 Ti, Ada) - each one calls this out where it matters. Don't switch these to float16 for T4's native FP16 tensor cores to "fix" this: Bielik was trained in bf16, and running the full model in fp16 causes activation overflow across the 32 stacked decoder layers, producing fluent-looking but semantically garbage output. For BF16-accurate and BF16-fast numbers, use an Ampere+ runtime (A100/L4 via Colab Pro).

## Getting Started

```bash
# Clone the repository
git clone https://github.com/qooba/bielik-anatomy-triton
cd bielik-anatomy-triton

# Install dependencies
pip install -r requirements.txt

```
