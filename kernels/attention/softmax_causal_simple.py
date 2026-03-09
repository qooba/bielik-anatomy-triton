import torch
import triton
import triton.language as tl


@triton.jit
def softmax_causal_simple_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    seq_len,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Causal softmax kernel (simple) - single-pass softmax with causal masking.
    """
    row_idx = tl.program_id(axis=0)
    if row_idx >= n_rows:
        return

    position = row_idx % seq_len

    input_row_start = input_ptr + row_idx * input_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_vals = tl.load(input_row_start + col_offsets, mask=mask, other=-float("inf"))

    is_future = col_offsets > position
    row_vals = tl.where(is_future, -float("inf"), row_vals)

    row_max = tl.max(row_vals, axis=0)
    numerator = tl.exp(row_vals - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_row_start + col_offsets, softmax_output, mask=mask)


def softmax_causal_simple(x: torch.Tensor) -> torch.Tensor:
    """
    Apply causal softmax to attention scores (simple single-pass version).
    """

    batch, num_heads, seq_len, seq_len2 = x.shape
    assert seq_len == seq_len2, "Attention scores must be square (seq_len x seq_len)"

    x_2d = x.reshape(-1, seq_len)
    n_rows = x_2d.shape[0]
    n_cols = x_2d.shape[1]

    output = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    softmax_causal_simple_kernel[grid](
        x_2d,
        output,
        n_rows,
        n_cols,
        seq_len,
        x_2d.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(batch, num_heads, seq_len, seq_len)
