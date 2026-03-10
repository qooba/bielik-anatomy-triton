# Episode 2: Matmul - Heart of the Transformer

[Back to Series Overview](/README.md) | [Previous: Introduction](/docs/ep01-introduction.md) | [Next: RMSNorm and Softmax](/docs/ep03-rmsnorm-softmax-fused.md)

---

<p align="center">
    <a href="https://www.youtube.com/watch?v=W8NFUvYX-u8">
        <img src="https://img.youtube.com/vi/W8NFUvYX-u8/sddefault.jpg" alt="Episode 2: Matmul" style="max-width: 100%;">
    </a>
</p>

## Overview

Matrix multiplication is the single most frequent operation in a transformer - Bielik executes hundreds of matmuls per forward pass, accounting for roughly 80% of compute time. This episode builds a Triton matmul kernel from scratch and progressively optimizes it to match PyTorch/cuBLAS performance.

## Topics Covered

- **Why matmul matters** - the dominant operation in every transformer layer
- **Basic Triton matmul** - block decomposition, pointer arithmetic with broadcasting, K-loop, boundary masking
- **GPU memory hierarchy** - registers, shared memory (SRAM), L2 cache, global memory (HBM); speed differences up to 100x
- **Optimization 1: Grouped block ordering** - processing blocks in super-groups of 8 for better L2 cache reuse
- **Optimization 2: Auto-tuning** - `@triton.autotune` to automatically search optimal block sizes for different hardware
- **Optimization 3: Tensor Cores** - switching from FP32 to BF16 to engage hardware matrix units; 16x throughput gain; FP32 accumulator for numerical stability
- **Optimization 4: Pipeline and occupancy** - 5-stage pipeline to overlap loads with compute, 8 warps for better GPU occupancy

## Relevant Code

### Kernels
- [`kernels/matmul/matmul_basic.py`](/kernels/matmul/matmul_basic.py) - naive implementation
- [`kernels/matmul/matmul.py`](/kernels/matmul/matmul.py) - Tensor Core optimized with auto-tuning

### Benchmarks
- [`benchmarks/matmul/benchmark_matmul.py`](/benchmarks/matmul/benchmark_matmul.py) - performance comparison script
- [`benchmarks/matmul/verify_correctness.py`](/benchmarks/matmul/verify_correctness.py) - numerical correctness tests

## Benchmarks

- [`benchmarks/matmul/benchmark_matmul_guide.py`](/benchmarks/matmul/benchmark_matmul_guide.py)

To run benchmarks use:
```bash
make benchmark-matmul
```

### Results on my RTX 4060 Ti

<p align="center">
    <img src="/docs/plots/matmul/matmul-tflops-vs-size.png" alt="tflops vs size" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/matmul/matmul-summary-bielik-shapes-tflops.png" alt="tflops vs size summary" style="max-width: 100%;">
</p>



## References

- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [NVIDIA Tensor Core documentation](https://www.nvidia.com/en-us/data-center/tensor-cores/)

---

[Back to Series Overview](/README.md) | [Previous: Introduction](/docs/ep01-introduction.md) | [Next: RMSNorm and Softmax](/docs/ep03-rmsnorm-softmax-fused.md)
