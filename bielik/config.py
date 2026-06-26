class BielikConfig:
    """Configuration for Bielik 1.5B-v3.0-Instruct model"""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1536,
        num_layers: int = 32,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,  # GQA: 2 KV heads
        intermediate_size: int = 8960,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        pad_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.pad_token_id = pad_token_id

        self.head_dim = hidden_size // num_attention_heads
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        assert num_attention_heads % num_key_value_heads == 0, \
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"

    @classmethod
    def from_pretrained(cls, config_dict: dict):
        """Create config from HuggingFace config dictionary"""
        return cls(
            vocab_size=config_dict['vocab_size'],
            hidden_size=config_dict['hidden_size'],
            num_layers=config_dict['num_hidden_layers'],
            num_attention_heads=config_dict['num_attention_heads'],
            num_key_value_heads=config_dict['num_key_value_heads'],
            intermediate_size=config_dict['intermediate_size'],
            max_position_embeddings=config_dict['max_position_embeddings'],
            rms_norm_eps=config_dict['rms_norm_eps'],
            rope_theta=config_dict['rope_theta'],
            pad_token_id=config_dict.get('pad_token_id', 2),
        )

    def __repr__(self):
        return (
            f"BielikConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  max_position_embeddings={self.max_position_embeddings},\n"
            f"  rms_norm_eps={self.rms_norm_eps},\n"
            f"  rope_theta={self.rope_theta}\n"
            f")"
        )
