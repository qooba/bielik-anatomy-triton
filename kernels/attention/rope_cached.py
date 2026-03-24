import triton
import triton.language as tl
import torch
import math

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['half_dim', 'n_tokens'],
)
@triton.jit
def rope_cached_kernel(
    x_ptr,
    positions_ptr,
    cos_cache_ptr,
    sin_cache_ptr,
    stride_token,
    pos_stride_batch,
    pos_stride_seq,
    seq_len,
    num_heads,
    half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    batch_idx = token_idx // (seq_len * num_heads)
    seq_idx = (token_idx // num_heads) % seq_len

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    token_base = token_idx * stride_token

    x_first  = tl.load(x_ptr + token_base +          offsets, mask=mask, other=0.0)
    x_second = tl.load(x_ptr + token_base + half_dim + offsets, mask=mask, other=0.0)

    pos_offset = batch_idx * pos_stride_batch + seq_idx * pos_stride_seq
    position  = tl.load(positions_ptr + pos_offset)   # scalar int32

    cache_row = position * half_dim
    cos_val = tl.load(cos_cache_ptr + cache_row + offsets, mask=mask, other=1.0)
    sin_val = tl.load(sin_cache_ptr + cache_row + offsets, mask=mask, other=0.0)

    x_first_out  = x_first * cos_val - x_second * sin_val
    x_second_out = x_first * sin_val + x_second * cos_val

    tl.store(x_ptr + token_base +          offsets, x_first_out,  mask=mask)
    tl.store(x_ptr + token_base + half_dim + offsets, x_second_out, mask=mask)


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta_base: float = 10000.0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = head_dim // 2

    log_scale = -(math.log(theta_base) * 2.0 / head_dim)

    i = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta = torch.exp(i * log_scale) 

    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, theta)

    cos_cache = torch.cos(angles).to(dtype).contiguous()
    sin_cache = torch.sin(angles).to(dtype).contiguous()
    return cos_cache, sin_cache


def apply_rope_cached_(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2

    if positions.ndim == 1:
        pos_stride_batch = 0
        pos_stride_seq = positions.stride(0)
    else:
        pos_stride_batch = positions.stride(0)
        pos_stride_seq = positions.stride(1)

    x_2d = x.view(-1, head_dim)
    stride_token = x_2d.stride(0)
    n_tokens = x_2d.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(half_dim)

    rope_cached_kernel[(n_tokens,)](
        x_2d,
        positions,
        cos_cache,
        sin_cache,
        stride_token,
        pos_stride_batch,
        pos_stride_seq,
        seq_len,
        num_heads,
        half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x
