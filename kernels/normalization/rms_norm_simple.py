import triton
import triton.language as tl


@triton.jit
def rms_norm_simple_kernel(
    X,
    W,
    Y,
    stride_x_row,
    stride_y_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-pass RMSNorm kernel.
    """
    row_idx = tl.program_id(axis=0)

    X += row_idx * stride_x_row
    Y += row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(X + col_offsets, mask=mask, other=0.0)
    w = tl.load(W + col_offsets, mask=mask, other=1.0)

    x_squared = x * x
    mean_x_squared = tl.sum(x_squared) / n_cols
    rms = tl.sqrt(mean_x_squared + eps)
    x_normalized = x / rms
    y = x_normalized * w

    tl.store(Y + col_offsets, y, mask=mask)


def rms_norm_simple(x, weight, eps=1e-6):
    import torch

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    n_rows = x_2d.shape[0]

    output = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)

    grid = (n_rows,)

    rms_norm_simple_kernel[grid](
        x_2d,
        weight,
        output,
        x_2d.stride(0),
        output.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(orig_shape)
