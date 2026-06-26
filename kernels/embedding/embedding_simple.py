import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    token_ids_ptr,
    embedding_ptr,
    output_ptr,
    n_tokens,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    if pid_token >= n_tokens:
        return

    token_id = tl.load(token_ids_ptr + pid_token)

    offs = pid_chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim

    emb_row = token_id.to(tl.int64) * embedding_dim
    out_row = pid_token.to(tl.int64) * embedding_dim

    vec = tl.load(embedding_ptr + emb_row + offs, mask=mask, other=0.0)
    tl.store(output_ptr + out_row + offs, vec, mask=mask)


def embedding_forward(
    token_ids: torch.Tensor,
    embedding_table: torch.Tensor,
) -> torch.Tensor:
    assert token_ids.is_cuda and embedding_table.is_cuda, "Inputs must be on CUDA"
    assert embedding_table.ndim == 2, f"Embedding table must be 2D, got {embedding_table.ndim}D"

    vocab_size, embedding_dim = embedding_table.shape

    orig_shape = token_ids.shape
    token_ids_flat = token_ids.reshape(-1).contiguous()
    n_tokens = token_ids_flat.shape[0]

    output_flat = torch.empty(
        (n_tokens, embedding_dim),
        device=token_ids.device,
        dtype=embedding_table.dtype
    )

    BLOCK_SIZE = 128
    n_chunks = triton.cdiv(embedding_dim, BLOCK_SIZE)

    grid = (n_tokens, n_chunks)

    embedding_kernel[grid](
        token_ids_flat,
        embedding_table,
        output_flat,
        n_tokens,
        embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output_shape = list(orig_shape) + [embedding_dim]
    return output_flat.reshape(output_shape)