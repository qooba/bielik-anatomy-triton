"""
GPU-specific auto-tuning configurations

This module provides optimized Triton configs for different GPU models.
Use these configs in @triton.autotune decorators.

Usage:
    from kernels.common.gpu_configs import get_flash_attention_configs

    @triton.autotune(
        configs=get_flash_attention_configs(),
        key=['N_CTX', 'HEAD_DIM', 'IS_CAUSAL'],
    )
    @triton.jit
    def my_kernel(...):
        pass
"""

import triton
import torch


def get_gpu_tier():
    """
    Detect GPU tier based on GPU name, compute capability and SM count.

    Returns:
        str: 'small', 'medium', 'large', or 'xl'
    """
    if not torch.cuda.is_available():
        return 'small'

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name.lower()
    compute_capability = props.major * 10 + props.minor
    sm_count = props.multi_processor_count

    # RTX 5090, H100, A100 (high SM count or compute capability 9.0+)
    if compute_capability >= 90 or sm_count >= 100 or any(x in gpu_name for x in ['h100', 'a100', '5090']):
        return 'xl'

    # RTX 4090, A6000, RTX 6000 Ada (high SM count: 128+ for 4090)
    elif sm_count >= 100 or any(x in gpu_name for x in ['4090', 'a6000', '6000 ada', 'l40']):
        return 'large'

    # RTX 3090, RTX 4080, A40 (medium SM count: 68-88)
    elif sm_count >= 60 or any(x in gpu_name for x in ['3090', '4080', 'a40', '3080 ti']):
        return 'medium'

    # RTX 4060 Ti, RTX 3060, T4 (low SM count: <60)
    else:
        return 'small'


def get_compute_capability():
    """
    Get GPU compute capability as (major, minor) tuple.

    Returns:
        tuple: (major, minor) e.g., (8, 9) for sm_89
    """
    if not torch.cuda.is_available():
        return (0, 0)

    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor)


def has_tensor_cores():
    """
    Check if GPU has tensor cores (Volta/sm_70 and newer).

    Tensor cores are available on:
    - Volta (sm_70): V100
    - Turing (sm_75): RTX 20 series, T4
    - Ampere (sm_80, sm_86): A100, RTX 30 series
    - Ada Lovelace (sm_89): RTX 40 series
    - Hopper (sm_90): H100, RTX 50 series

    Returns:
        bool: True if GPU has tensor cores
    """
    major, minor = get_compute_capability()
    return major >= 7  # sm_70 (Volta) and newer


def get_tensor_core_config(tier=None):
    """
    Get tensor core-optimized configurations for matmul and attention kernels.

    These configs are specifically tuned to maximize tensor core utilization:
    - Larger block sizes (256x256 vs 128x128 for xl tier)
    - More pipeline stages (5-6 vs 2 for xl tier)
    - More warps (8 vs 4) for better occupancy
    - Block dimensions are multiples of 16 (tensor core tile size for bf16)

    Args:
        tier: 'small', 'medium', 'large', 'xl', or None (auto-detect)

    Returns:
        dict with:
            'block_sizes': List of (M, N, K) tuples for matmul
            'num_stages': Optimal pipeline stages
            'num_warps': Optimal warp count
            'alignment': Required memory alignment (16 bytes for bf16)
    """
    if tier is None:
        tier = get_gpu_tier()

    if tier == 'xl':
        # RTX 5090, H100, A100 - Maximum tensor core utilization
        return {
            'block_sizes': [
                (256, 256, 64),  # Large square tiles
                (256, 128, 64),  # Wide tiles
                (128, 256, 64),  # Tall tiles
                (128, 128, 64),  # Fallback
            ],
            'num_stages': 5,  # Aggressive pipelining (5-6 stages)
            'num_warps': 8,   # Maximum warps for occupancy
            'alignment': 16,  # 16-byte alignment for bf16 tensor cores
        }

    elif tier == 'large':
        # RTX 4090, A6000 - High tensor core utilization
        return {
            'block_sizes': [
                (256, 128, 64),  # Large tiles
                (128, 256, 64),
                (128, 128, 64),
                (128, 64, 32),   # Fallback
            ],
            'num_stages': 4,  # Good pipelining (4-5 stages)
            'num_warps': 8,
            'alignment': 16,
        }

    elif tier == 'medium':
        # RTX 3090, RTX 4080 - Balanced tensor core usage
        return {
            'block_sizes': [
                (128, 128, 64),  # Medium tiles
                (128, 64, 32),
                (64, 128, 32),
                (64, 64, 32),    # Fallback
            ],
            'num_stages': 3,  # Moderate pipelining (3-4 stages)
            'num_warps': 4,
            'alignment': 16,
        }

    else:  # small
        # RTX 3060, T4 - Conservative tensor core usage
        return {
            'block_sizes': [
                (64, 64, 32),    # Small tiles
                (128, 64, 32),
                (64, 128, 32),
            ],
            'num_stages': 2,  # Conservative pipelining (2-3 stages)
            'num_warps': 4,
            'alignment': 16,
        }


def get_flash_attention_configs(tier=None):
    """
    Get Flash Attention auto-tuning configs for GPU tier.

    Args:
        tier: 'small', 'medium', 'large', 'xl', or None (auto-detect)

    Returns:
        list of triton.Config
    """
    if tier is None:
        tier = get_gpu_tier()

    if tier == 'xl':
        # RTX 5090, H100, A100 (200KB+ shared memory)
        return [
            # Large blocks with high pipelining
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=5, num_warps=8),

            # Medium blocks with max pipelining
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=4),

            # Fallback for very long sequences
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        ]

    elif tier == 'large':
        # RTX 4090, A6000 (150KB+ shared memory)
        return [
            # Large blocks with good pipelining
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),

            # Medium blocks with higher pipelining
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),

            # Fallback
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        ]

    elif tier == 'medium':
        # RTX 3090, RTX 4080 (100KB+ shared memory)
        return [
            # Balanced configs
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),

            # Fallback for long sequences
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        ]

    else:  # small
        # RTX 4060 Ti, RTX 3060 (101KB shared memory - current)
        return [
            # Conservative configs to fit shared memory
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        ]


def get_matmul_configs(tier=None):
    """
    Get Matmul auto-tuning configs for GPU tier.

    Args:
        tier: 'small', 'medium', 'large', 'xl', or None (auto-detect)

    Returns:
        list of triton.Config
    """
    if tier is None:
        tier = get_gpu_tier()

    if tier == 'xl':
        # RTX 5090, H100, A100
        return [
            # Large tiles with high pipelining
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),

            # Fallback
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        ]

    elif tier == 'large':
        # RTX 4090, A6000
        return [
            # Large tiles
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        ]

    elif tier == 'medium':
        # RTX 3090, RTX 4080
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        ]

    else:  # small
        # RTX 4060 Ti, RTX 3060
        return [
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        ]


def get_rms_norm_configs(tier=None):
    """
    Get RMSNorm auto-tuning configs for GPU tier.

    Args:
        tier: 'small', 'medium', 'large', 'xl', or None (auto-detect)

    Returns:
        list of triton.Config
    """
    if tier is None:
        tier = get_gpu_tier()

    if tier in ['xl', 'large']:
        # RTX 5090, RTX 4090, H100, A100
        return [
            triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        ]

    elif tier == 'medium':
        # RTX 3090, RTX 4080
        return [
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        ]

    else:  # small
        # RTX 4060 Ti, RTX 3060
        return [
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        ]


def get_rope_configs(tier=None):
    """
    Get RoPE auto-tuning configs for GPU tier.

    Args:
        tier: 'small', 'medium', 'large', 'xl', or None (auto-detect)

    Returns:
        list of triton.Config
    """
    if tier is None:
        tier = get_gpu_tier()

    if tier in ['xl', 'large']:
        # RTX 5090, RTX 4090, H100, A100
        return [
            triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        ]

    elif tier == 'medium':
        # RTX 3090, RTX 4080
        return [
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        ]

    else:  # small
        # RTX 4060 Ti, RTX 3060
        return [
            triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        ]


# GPU tier specifications for reference
GPU_SPECS = {
    'xl': {
        'examples': ['RTX 5090', 'H100', 'A100'],
        'shared_memory_kb': '200+',
        'compute_capability': '9.0+',
        'num_stages': '4-6',
        'block_sizes': 'Large (128x128, 256x256)',
        'expected_speedup': '2.5-3.5x vs PyTorch at long sequences',
    },
    'large': {
        'examples': ['RTX 4090', 'RTX 6000 Ada', 'A6000'],
        'shared_memory_kb': '150-200',
        'compute_capability': '8.9+',
        'num_stages': '3-5',
        'block_sizes': 'Medium-Large (64x128, 128x128)',
        'expected_speedup': '2.0-2.5x vs PyTorch at long sequences',
    },
    'medium': {
        'examples': ['RTX 3090', 'RTX 4080', 'A40'],
        'shared_memory_kb': '100-150',
        'compute_capability': '8.6+',
        'num_stages': '3-4',
        'block_sizes': 'Medium (64x64, 128x64)',
        'expected_speedup': '1.7-2.0x vs PyTorch at long sequences',
    },
    'small': {
        'examples': ['RTX 4060 Ti', 'RTX 3060', 'T4'],
        'shared_memory_kb': '100',
        'compute_capability': '7.5-8.9',
        'num_stages': '2-3',
        'block_sizes': 'Small-Medium (64x32, 64x64)',
        'expected_speedup': '1.5-1.8x vs PyTorch at long sequences',
    },
}


def print_gpu_info():
    """Print current GPU tier and recommendations."""
    if not torch.cuda.is_available():
        print("No CUDA GPU available")
        return

    tier = get_gpu_tier()
    props = torch.cuda.get_device_properties(0)
    major, minor = get_compute_capability()
    has_tc = has_tensor_cores()
    tc_config = get_tensor_core_config()

    print(f"GPU: {props.name}")
    print(f"Compute Capability: {major}.{minor} (sm_{major}{minor})")
    print(f"Tensor Cores: {'Available ✓' if has_tc else 'Not Available ✗'}")
    print(f"SM Count: {props.multi_processor_count}")
    print(f"Shared Memory (per block): {props.shared_memory_per_block // 1024} KB")
    print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"Detected Tier: {tier.upper()}")

    if has_tc:
        print(f"\nTensor Core Configuration ({tier.upper()} tier):")
        print(f"  Block sizes: {tc_config['block_sizes']}")
        print(f"  Pipeline stages: {tc_config['num_stages']}")
        print(f"  Warps: {tc_config['num_warps']}")
        print(f"  Alignment: {tc_config['alignment']} bytes")

    print(f"\nRecommendations for {tier.upper()} tier:")

    specs = GPU_SPECS[tier]
    for key, value in specs.items():
        if key != 'examples':
            print(f"  {key}: {value}")

    print(f"\nExample GPUs in this tier: {', '.join(specs['examples'])}")


if __name__ == '__main__':
    print_gpu_info()

    print("\n" + "="*80)
    print("Auto-tune Config Comparison")
    print("="*80)

    for tier in ['small', 'medium', 'large', 'xl']:
        print(f"\n{tier.upper()} TIER:")
        print(f"  Flash Attention: {len(get_flash_attention_configs(tier))} configs")
        print(f"  Matmul: {len(get_matmul_configs(tier))} configs")
        print(f"  RMSNorm: {len(get_rms_norm_configs(tier))} configs")
        print(f"  RoPE: {len(get_rope_configs(tier))} configs")
