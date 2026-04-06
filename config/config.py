"""Model and training configuration for Korean GPT."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration (~10M parameters)."""

    vocab_size: int = 8000
    max_seq_len: int = 256
    d_model: int = 320
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1280
    dropout: float = 0.1

    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


@dataclass
class TrainingConfig:
    """Training configuration with MPS optimization."""

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: Optional[int] = None

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Logging and saving
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

    # Paths
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "tokenizer/korean_bpe.model"

    # Device settings (MPS optimization)
    device: str = "mps"  # Use "cuda" for NVIDIA, "cpu" for CPU
    num_workers: int = 0  # MPS works best with 0
    pin_memory: bool = False  # Disable for MPS

    # Mixed precision (disabled for MPS stability)
    use_amp: bool = False


@dataclass
class TokenizerConfig:
    """Tokenizer training configuration."""

    vocab_size: int = 8000
    model_type: str = "bpe"
    character_coverage: float = 0.9995

    # Special tokens
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"

    # Output path
    model_prefix: str = "tokenizer/korean_bpe"
