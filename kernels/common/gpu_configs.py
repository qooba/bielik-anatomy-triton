import torch


def get_gpu_tier():
    """
    Detect GPU tier based on GPU name, compute capability and SM count.

    Returns:
        str: 'small', 'medium', 'large', or 'xl'
    """
    if not torch.cuda.is_available():
        return "small"

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name.lower()
    compute_capability = props.major * 10 + props.minor
    sm_count = props.multi_processor_count

    # RTX 5090, H100, A100 (high SM count or compute capability 9.0+)
    if (
        compute_capability >= 90
        or sm_count >= 100
        or any(x in gpu_name for x in ["h100", "a100", "5090"])
    ):
        return "xl"

    # RTX 4090, A6000, RTX 6000 Ada (high SM count: 128+ for 4090)
    elif sm_count >= 100 or any(x in gpu_name for x in ["4090", "a6000", "6000 ada", "l40"]):
        return "large"

    # RTX 3090, RTX 4080, A40 (medium SM count: 68-88)
    elif sm_count >= 60 or any(x in gpu_name for x in ["3090", "4080", "a40", "3080 ti"]):
        return "medium"

    # RTX 4060 Ti, RTX 3060, T4 (low SM count: <60)
    else:
        return "small"


def get_tensor_core_config(tier=None):

    if tier is None:
        tier = get_gpu_tier()

    if tier == "xl":
        # RTX 5090, H100, A100 - Maximum tensor core utilization
        return {
            "block_sizes": [
                (256, 256, 64),  # Large square tiles
                (256, 128, 64),  # Wide tiles
                (128, 256, 64),  # Tall tiles
                (128, 128, 64),  # Fallback
            ],
            "num_stages": 5,  # Aggressive pipelining (5-6 stages)
            "num_warps": 8,  # Maximum warps for occupancy
            "alignment": 16,  # 16-byte alignment for bf16 tensor cores
        }

    elif tier == "large":
        # RTX 4090, A6000 - High tensor core utilization
        return {
            "block_sizes": [
                (256, 128, 64),  # Large tiles
                (128, 256, 64),
                (128, 128, 64),
                (128, 64, 32),  # Fallback
            ],
            "num_stages": 4,  # Good pipelining (4-5 stages)
            "num_warps": 8,
            "alignment": 16,
        }

    elif tier == "medium":
        # RTX 3090, RTX 4080 - Balanced tensor core usage
        return {
            "block_sizes": [
                (128, 128, 64),  # Medium tiles
                (128, 64, 32),
                (64, 128, 32),
                (64, 64, 32),  # Fallback
            ],
            "num_stages": 3,  # Moderate pipelining (3-4 stages)
            "num_warps": 4,
            "alignment": 16,
        }

    else:  # small
        # RTX 3060, T4 - Conservative tensor core usage
        return {
            "block_sizes": [
                (64, 64, 32),  # Small tiles
                (128, 64, 32),
                (64, 128, 32),
            ],
            "num_stages": 2,  # Conservative pipelining (2-3 stages)
            "num_warps": 4,
            "alignment": 16,
        }
