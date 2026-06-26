"""
Bielik Decoder Layer Implementation
"""

import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))

from kernels.normalization.rms_norm_simple import rms_norm_simple
from kernels.matmul.matmul_bias_tensorcore import matmul_bias_forward
from kernels.attention.rope_cached import apply_rope_cached_
from kernels.attention.flash_attention_simple import flash_attention_forward
from kernels.ffn.swiglu_fused_bias import fused_ffn_swiglu_bias


class BielikDecoderLayer:

    def __init__(
        self,
        layer_idx: int,
        config,
        weights: dict,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize decoder layer with weights from HuggingFace checkpoint.

        Args:
            layer_idx: Layer index (0-31 for Bielik)
            config: BielikConfig object
            weights: Dictionary containing layer weights from safetensors
            device: Device to load weights on
            dtype: Data type for weights
        """
        self.layer_idx = layer_idx
        self.config = config
        self.device = device
        self.dtype = dtype

        prefix = f"model.layers.{layer_idx}"

        # Attention block
        self.input_layernorm_weight = weights[f"{prefix}.input_layernorm.weight"].to(device=device, dtype=dtype)

        # Transpose weights: PyTorch stores as [out, in], our kernel expects [in, out]
        self.q_proj_weight = weights[f"{prefix}.self_attn.q_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.q_proj_bias = weights[f"{prefix}.self_attn.q_proj.bias"].to(device=device, dtype=dtype)

        self.k_proj_weight = weights[f"{prefix}.self_attn.k_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.k_proj_bias = weights[f"{prefix}.self_attn.k_proj.bias"].to(device=device, dtype=dtype)

        self.v_proj_weight = weights[f"{prefix}.self_attn.v_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.v_proj_bias = weights[f"{prefix}.self_attn.v_proj.bias"].to(device=device, dtype=dtype)

        self.o_proj_weight = weights[f"{prefix}.self_attn.o_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.o_proj_bias = weights[f"{prefix}.self_attn.o_proj.bias"].to(device=device, dtype=dtype)

        # FFN block
        self.post_attention_layernorm_weight = weights[f"{prefix}.post_attention_layernorm.weight"].to(device=device, dtype=dtype)

        self.gate_proj_weight = weights[f"{prefix}.mlp.gate_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.gate_proj_bias = weights[f"{prefix}.mlp.gate_proj.bias"].to(device=device, dtype=dtype)

        self.up_proj_weight = weights[f"{prefix}.mlp.up_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.up_proj_bias = weights[f"{prefix}.mlp.up_proj.bias"].to(device=device, dtype=dtype)

        self.down_proj_weight = weights[f"{prefix}.mlp.down_proj.weight"].T.contiguous().to(device=device, dtype=dtype)
        self.down_proj_bias = weights[f"{prefix}.mlp.down_proj.bias"].to(device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len] or [seq_len]
            cos_cache: [max_seq_len, head_dim // 2]
            sin_cache: [max_seq_len, head_dim // 2]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Attention Block
        # Save residual
        residual = hidden_states

        # Pre-normalization
        hidden_states = rms_norm_simple(
            hidden_states,
            self.input_layernorm_weight,
            eps=self.config.rms_norm_eps
        )

        # Q/K/V projections with bias
        # Reshape to 2D for matmul: [B, S, H] -> [B*S, H]
        hidden_2d = hidden_states.reshape(-1, hidden_size)

        q = matmul_bias_forward(hidden_2d, self.q_proj_weight, self.q_proj_bias)  # [B*S, 1536]
        k = matmul_bias_forward(hidden_2d, self.k_proj_weight, self.k_proj_bias)  # [B*S, 256]
        v = matmul_bias_forward(hidden_2d, self.v_proj_weight, self.v_proj_bias)  # [B*S, 256]

        # Reshape back to 3D
        q = q.reshape(batch_size, seq_len, -1)  # [B, S, 1536]
        k = k.reshape(batch_size, seq_len, -1)  # [B, S, 256]
        v = v.reshape(batch_size, seq_len, -1)  # [B, S, 256]

        # Reshape for multi-head attention
        # Q: [B, S, 1536] -> [B, S, 12, 128]
        q = q.view(batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim)

        # K, V: [B, S, 256] -> [B, S, 2, 128]
        k = k.view(batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim)
        v = v.view(batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim)

        # Apply RoPE (in-place) BEFORE transpose
        # RoPE expects [B, S, H, D], which is what we have now
        apply_rope_cached_(q, position_ids, cos_cache, sin_cache)
        apply_rope_cached_(k, position_ids, cos_cache, sin_cache)

        # NOW transpose for flash attention: [B, S, H, D] -> [B, H, S, D]
        q = q.transpose(1, 2).contiguous()  # [B, 12, S, 128]
        k = k.transpose(1, 2).contiguous()  # [B, 2, S, 128]
        v = v.transpose(1, 2).contiguous()  # [B, 2, S, 128]

        # Flash Attention (handles GQA + causal masking internally)
        attn_output = flash_attention_forward(
            q, k, v,
            causal=True,
            sm_scale=1.0 / (self.config.head_dim ** 0.5)
        )  # [B, 12, S, 128]

        # Reshape back: [B, 12, S, 128] -> [B, S, 12, 128] -> [B, S, 1536]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)

        # Output projection with bias
        # Reshape to 2D: [B, S, H] -> [B*S, H]
        attn_output_2d = attn_output.reshape(-1, hidden_size)
        attn_output_2d = matmul_bias_forward(attn_output_2d, self.o_proj_weight, self.o_proj_bias)
        attn_output = attn_output_2d.reshape(batch_size, seq_len, hidden_size)

        # Residual connection
        hidden_states = residual + attn_output

        # FFN Block
        # Save residual
        residual = hidden_states

        # Pre-normalization
        hidden_states = rms_norm_simple(
            hidden_states,
            self.post_attention_layernorm_weight,
            eps=self.config.rms_norm_eps
        )

        # Fused SwiGLU: SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
        hidden_states = fused_ffn_swiglu_bias(
            hidden_states,
            self.gate_proj_weight,
            self.gate_proj_bias,
            self.up_proj_weight,
            self.up_proj_bias
        )  # [B, S, 8960]

        # Down projection with bias
        # Reshape to 2D: [B, S, 8960] -> [B*S, 8960]
        hidden_2d = hidden_states.reshape(-1, self.config.intermediate_size)
        hidden_2d = matmul_bias_forward(hidden_2d, self.down_proj_weight, self.down_proj_bias)
        hidden_states = hidden_2d.reshape(batch_size, seq_len, self.config.hidden_size)  # [B, S, 1536]

        # Residual connection
        hidden_states = residual + hidden_states

        return hidden_states

    def __call__(self, *args, **kwargs):
        """Make layer callable like nn.Module"""
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return (
            f"BielikDecoderLayer(\n"
            f"  layer_idx={self.layer_idx},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_heads={self.config.num_attention_heads},\n"
            f"  num_kv_heads={self.config.num_key_value_heads},\n"
            f"  intermediate_size={self.config.intermediate_size}\n"
            f")"
        )
