"""
Bielik 1.5B Model Implementation using Triton Kernels
"""

import torch
import json
import os
import sys
from pathlib import Path
from safetensors.torch import load_file
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernels"))

from kernels.embedding.embedding_simple import embedding_forward
from kernels.normalization.rms_norm_simple import rms_norm_simple
from kernels.matmul.matmul_bias_tensorcore import matmul_bias_forward
from kernels.attention.rope_cached import build_rope_cache

from .config import BielikConfig
from .layer import BielikDecoderLayer


class BielikModel:
    """
    Bielik 1.5B Transformer Model using Triton kernels.

    Architecture:
        embed_tokens -> 32 x BielikDecoderLayer -> norm -> lm_head

    All compute-intensive operations use Triton kernels:
        - embedding_simple.py: Token embeddings
        - rms_norm_simple.py: RMSNorm (65 x per forward)
        - matmul_bias_tensorcore.py: All linear layers with bias (161 x per forward)
        - rope_cached.py: Rotary position embeddings (64 x per forward)
        - flash_attention_simple.py: Flash Attention with GQA (32 x per forward)
        - swiglu_fused_bias.py: Fused SwiGLU FFN (32 x per forward)

    Only trivial PyTorch operations used:
        - Residual addition (element-wise add)
        - Tensor reshaping/transpose (metadata ops, zero-cost)
        - Position IDs generation (one-time per forward)
    """

    def __init__(
        self,
        config: BielikConfig,
        weights: dict,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Bielik model with weights.

        Args:
            config: BielikConfig object
            weights: Dictionary containing model weights from safetensors
            device: Device to load model on
            dtype: Data type for weights
        """
        self.config = config
        self.device = device
        self.dtype = dtype

        print(f"Initializing Bielik model...")
        print(f"  Config: {config.num_layers} layers, {config.hidden_size}D, {config.num_attention_heads} heads")
        print(f"  Device: {device}, dtype: {dtype}")

        # Token embeddings
        self.embed_tokens = weights["model.embed_tokens.weight"].to(device=device, dtype=dtype)
        print(f"  Loaded embeddings: {self.embed_tokens.shape}")

        # Decoder layers
        print(f"  Loading {config.num_layers} decoder layers...")
        self.layers = []
        for i in range(config.num_layers):
            layer = BielikDecoderLayer(i, config, weights, device, dtype)
            self.layers.append(layer)
            if (i + 1) % 8 == 0:
                print(f"    Loaded layers 0-{i}")
        print(f"  All {config.num_layers} layers loaded")

        # Final norm
        self.norm_weight = weights["model.norm.weight"].to(device=device, dtype=dtype)

        # LM head (no bias) - transpose: PyTorch stores as [vocab, hidden], kernel expects [hidden, vocab]
        self.lm_head_weight = weights["lm_head.weight"].T.contiguous().to(device=device, dtype=dtype)
        print(f"  Loaded final norm and LM head")

        # Build RoPE cache (one-time at initialization)
        print(f"  Building RoPE cache (max_seq_len={config.max_position_embeddings}, theta={config.rope_theta})...")
        self.cos_cache, self.sin_cache = build_rope_cache(
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            theta_base=config.rope_theta,
            device=device,
            dtype=torch.float32,  # RoPE cache in fp32 for precision
        )
        print(f"  RoPE cache built: {self.cos_cache.shape}")

        print(f"Bielik model initialized successfully!")

    def forward(
        self,
        input_ids: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs, shape [batch, seq_len]
            return_logits: If True, return full logits; if False, return hidden states

        Returns:
            logits: [batch, seq_len, vocab_size] if return_logits=True
                    OR hidden_states: [batch, seq_len, hidden_size] if return_logits=False
        """
        batch_size, seq_len = input_ids.shape

        # Embedding lookup
        hidden_states = embedding_forward(input_ids, self.embed_tokens)
        # [batch, seq_len, hidden_size]

        # Generate position IDs
        # For single batch: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long)
        # Expand for batch: [batch, seq_len]
        if batch_size > 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids,
                self.cos_cache,
                self.sin_cache,
            )

        # Final normalization
        hidden_states = rms_norm_simple(
            hidden_states,
            self.norm_weight,
            eps=self.config.rms_norm_eps
        )

        if not return_logits:
            return hidden_states

        # LM head (no bias)
        # Reshape to 2D: [B, S, H] -> [B*S, H]
        hidden_2d = hidden_states.reshape(-1, self.config.hidden_size)
        logits_2d = matmul_bias_forward(hidden_2d, self.lm_head_weight, bias=None)
        # Reshape back to 3D: [B*S, V] -> [B, S, V]
        logits = logits_2d.reshape(batch_size, seq_len, -1)
        # [batch, seq_len, vocab_size]

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive text generation (without KV cache).

        Args:
            input_ids: Prompt tokens, shape [batch, prompt_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: If True, sample; if False, use greedy decoding
            eos_token_id: Token ID for end of sequence

        Returns:
            generated_ids: [batch, prompt_len + generated_len]
        """
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Forward pass through entire sequence (no KV cache yet)
            logits = self.forward(input_ids)  # [batch, seq_len, vocab_size]

            # Get logits for last token
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Sample or greedy decode
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_actual = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_actual)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return input_ids

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Load pretrained Bielik model from HuggingFace checkpoint.

        Args:
            model_path: Path to model directory or HuggingFace model ID
                       e.g., "speakleash/Bielik-1.5B-v3.0-Instruct"
                       or path to local directory with model.safetensors
            device: Device to load model on
            dtype: Data type for model weights

        Returns:
            BielikModel instance
        """
        # Check if it's a HuggingFace model ID or local path
        if not os.path.exists(model_path):
            # Try HuggingFace cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_id = model_path.replace("/", "--")
            model_dir = cache_dir / f"models--{model_id}"

            if model_dir.exists():
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        model_path = str(snapshots[0])

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please download it first using:\n"
                f"  from transformers import AutoModelForCausalLM\n"
                f"  model = AutoModelForCausalLM.from_pretrained('{model_path}')\n"
            )

        print(f"Loading Bielik model from: {model_path}")

        # Load config
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        config = BielikConfig.from_pretrained(config_dict)
        print(f"Loaded config:")
        print(config)

        # Load weights
        weights_path = Path(model_path) / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        print(f"\nLoading weights from: {weights_path}")
        weights = load_file(str(weights_path))
        print(f"Loaded {len(weights)} weight tensors")

        # Create model
        model = cls(config, weights, device, dtype)

        return model

    def count_parameters(self):
        """Count total number of parameters in the model."""
        total = 0

        # Embeddings
        total += self.embed_tokens.numel()

        # Layers
        for layer in self.layers:
            total += layer.input_layernorm_weight.numel()
            total += layer.q_proj_weight.numel() + layer.q_proj_bias.numel()
            total += layer.k_proj_weight.numel() + layer.k_proj_bias.numel()
            total += layer.v_proj_weight.numel() + layer.v_proj_bias.numel()
            total += layer.o_proj_weight.numel() + layer.o_proj_bias.numel()
            total += layer.post_attention_layernorm_weight.numel()
            total += layer.gate_proj_weight.numel() + layer.gate_proj_bias.numel()
            total += layer.up_proj_weight.numel() + layer.up_proj_bias.numel()
            total += layer.down_proj_weight.numel() + layer.down_proj_bias.numel()

        # Final
        total += self.norm_weight.numel()
        total += self.lm_head_weight.numel()

        return total

    def __repr__(self):
        params = self.count_parameters()
        return (
            f"BielikModel(\n"
            f"  config={self.config.num_layers} layers x {self.config.hidden_size}D,\n"
            f"  parameters={params:,} ({params/1e9:.2f}B),\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )
