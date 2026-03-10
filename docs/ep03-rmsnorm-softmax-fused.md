# Episode 3: Just Fuse It — RMSNorm and Softmax

[Back to Series Overview](README.md) | [Previous: Matmul](ep02-matmul.md) | [Next: RoPE and Attention](ep04-rope-attention.md)

---


<p align="center">
    <a href="https://www.youtube.com/watch?v=FD_xre7abZU">
        <img src="https://img.youtube.com/vi/FD_xre7abZU/sddefault.jpg" alt="Episode 3: Fused RMSNorm & SoftMax" style="max-width: 100%;">
    </a>
</p>

## Overview

RMSNorm runs 65 times per Bielik forward pass; softmax runs 32 times. Individually they look cheap - but they share a bottleneck that has nothing to do with compute: the **memory wall**. This episode introduces kernel fusion as the primary optimization technique for memory-bound operations, builds fused Triton kernels for RMSNorm and softmax (with causal mask), and compares them against PyTorch's own fused implementations.

**Key insight:** Kernel fusion doesn't reduce computation - it reduces data movement.

---

## Topics Covered

- **The Memory Wall** - why GPUs sit idle 98% of the time on element-wise ops: data starvation, not lack of cores

- **Kernel Fusion** - what fusion means: combine sequential kernels -> data stays in fast registers/SRAM instead of bouncing through slow global memory

- **Fused Single-Pass RMSNorm Kernel**
- **Softmax: Single-Pass with Causal Mask**


---

## Relevant Code

### Kernels

- [`kernels/normalization/rms_norm_simple.py`](/kernels/normalization/rms_norm_simple.py) - single-pass fused RMSNorm

- [`kernels/attention/softmax_causal_simple.py`](/kernels/attention/softmax_causal_simple.py) - single-pass fused softmax + causal mask

---

## Benchmarks

- [`benchmarks/normalization/benchmark_rms_norm.py`](/benchmarks/normalization/benchmark_rms_norm.py)

To run benchmarks use:
```bash
make benchmark-rms-norm
```

- [`benchmarks/attention/benchmark_softmax_causal.py`](/benchmarks/attention/benchmark_softmax_causal.py)

To run benchmarks use:
```bash
make benchmark-softmax
```


### Results on my RTX 4060 Ti

#### RMSNorm
<p align="center">
    <img src="/docs/plots/normalization/rmsnorm-bandwidth-vs-hidden-size.png" alt="rmsnorm-bandwidth-vs-hidden-size" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/normalization/rmsnorm-bandwidth-vs-rows.png" alt="rmsnorm-bandwidth-vs-rows" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/normalization/rms_norm-summary-bielik-config-tflops.png" alt="rms_norm-summary-bielik-config-tflops" style="max-width: 100%;">
</p>

#### Softmax
<p align="center">
    <img src="/docs/plots/attention/softmax-causal-bandwidth-vs-heads.png" alt="softmax-causal-bandwidth-vs-heads" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/attention/softmax-causal-bandwidth-vs-seq-len.png" alt="softmax-causal-bandwidth-vs-seq-len" style="max-width: 100%;">
</p>

<p align="center">
    <img src="/docs/plots/attention/softmax_causal-summary-bielik-config-tflops.png" alt="softmax_causal-summary-bielik-config-tflops" style="max-width: 100%;">
</p>


---

[Back to Series Overview](README.md) | [Previous: Matmul](ep02-matmul.md) | [Next: RoPE and Attention](ep04-rope-attention.md)
