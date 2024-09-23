from dataclasses import dataclass, field
from transformers import PretrainedConfig

@dataclass
class TestModelConfig(PretrainedConfig):
    architectures: list[str] = field(default_factory=lambda: ["TestModel"])
    auto_map: dict[str, str] = field(
        default_factory=lambda: {"AutoConfig": "config.TestModelConfig"}
    )
    _commit_hash: str = ""
    attn_implementation: str = "flash_attn"
    vocab_size: int = 1024
    text_n_positions: int = 10
    num_attention_heads: int = 16
    hidden_size: int = 1024
    head_size: int = 64
    num_hidden_layers: int = 24


