# Episode 7: It's Alive!

[Back to Series Overview](README.md) | [Previous: Feed-Forward Network](ep06-feed-forward-network.md) |

---

<p align="center">
    <a href="https://www.youtube.com/watch?v=6ftMCzdrPng">
        <img src="https://img.youtube.com/vi/6ftMCzdrPng/sddefault.jpg" alt="Episode 7: It's Alive" style="max-width: 100%;">
    </a>
</p>

## Overview

Every kernel built across episodes 1-6 - matmul, RMSNorm, Flash Attention with RoPE, SwiGLU - is wired together into a complete, runnable Bielik 1.5B model. Two new kernels fill the remaining gaps: an embedding lookup and bias support in matmul and SwiGLU. Pretrained HuggingFace weights are loaded via SafeTensors and the model generates real Polish text. Without a KV cache, throughput is 49 tok/s (5% over PyTorch), but Time To First Token is 28% faster.

## Key Results (RTX 4060 Ti, Bielik 1.5B Instruct, bfloat16, batch=1)

### Forward Pass Latency

| Seq len | Triton | PyTorch | Speedup |
|--------:|-------:|--------:|--------:|
|      32 | 19.96 ms | 21.33 ms | 1.07x |
|      64 | 19.57 ms | 22.09 ms | 1.13x |
|     128 | 20.27 ms | 21.42 ms | 1.06x |
|     256 | 22.54 ms | 27.02 ms | **1.20x** |
|     512 | 43.24 ms | 46.73 ms | 1.08x |
| **avg** | | | **1.11x** |

Warmup=3, runs=5. Best case at seq=256: long enough for Triton kernels to amortize launch overhead, short enough that memory bandwidth is not yet the dominant bottleneck.

**~20 ms latency floor at short sequences (<= 128 tokens):** both implementations are dominated by kernel launch, memory allocation, and CUDA synchronisation overhead - not actual compute. The attention and FFN work for 32 tokens is trivially fast.

**Convergence at seq=512:** attention is O(N2) in memory access; both implementations hit the same VRAM bandwidth ceiling (~288 GB/s on the 4060 Ti), narrowing the gap back to 1.08x.

### Generation Throughput

Prompt: *"Warszawa to stolica Polski i jedno z największych miast w Europie Środkowej."* (14 tokens), 64 new tokens generated, greedy decoding.

| Metric | Triton | PyTorch | Speedup |
|--------|-------:|--------:|--------:|
| Time To First Token (TTFT) | 21.5 ms | 27.5 ms | **1.28x** |
| Total generation time | 1309.5 ms | 1372.8 ms | 1.05x |
| Throughput | 48.87 tok/s | 46.62 tok/s | **1.05x** |

TTFT (1.28x) wins more than throughput (1.05x): TTFT measures a single forward pass over the 14-token prompt - a clean kernel efficiency signal. Throughput degrades because **generation without KV cache is O(N2)**: each new token requires a full forward pass over the entire growing sequence. By token 64 the input is 78 tokens long, and quadratic attention recomputation dominates, drowning out kernel-level gains.

### Memory Usage

| Stage | VRAM |
|-------|-----:|
| Triton model loaded | 3050 MB |
| HuggingFace model (derived) | ~3045 MB |

Both models consume ~3 GB as expected: 1.5B parameters x 2 bytes (bfloat16).

## Relevant Code

### Kernels
- [`kernels/embedding/embedding_simple.pykernels/embedding_kernel.py`](/kernels/embedding/embedding_simple.py) - embedding lookup kernel
- [`kernels/matmul/matmul_bias_tensorcore.py`](/kernels/matmul/matmul_bias_tensorcore.py) - tiled matmul with optional bias fusion
- [`kernels/ffn/swiglu_fused_bias.py`](/kernels/ffn/swiglu_fused_bias.py) - fused SwiGLU gate+up with dual bias

### Model
- [`bielik/model.py`](/bielik/model.py) - `BielikModel`
- [`bielik/layer.py`](/bielik/layer.py) - `Decoder Linear`
- [`bielik/chat.py`](/bielik/chat.py) - interactive ChatML chat interface

---

[Back to Series Overview](README.md) | [Previous: Feed-Forward Network](ep06-feed-forward-network.md) |