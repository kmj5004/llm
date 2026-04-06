"""Token and Positional Embedding layers."""

import torch
import torch.nn as nn

from config.config import ModelConfig


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token IDs [batch_size, seq_len]

        Returns:
            Token embeddings [batch_size, seq_len, d_model]
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding (GPT-2 style)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.d_model
        )

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Positional embeddings [1, seq_len, d_model]
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.embedding(positions)


class GPTEmbedding(nn.Module):
    """Combined token and positional embedding with dropout."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.d_model = config.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token IDs [batch_size, seq_len]

        Returns:
            Combined embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape

        # Token embeddings
        token_emb = self.token_embedding(x)

        # Positional embeddings
        pos_emb = self.position_embedding(seq_len, x.device)

        # Combine and apply dropout
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)

        return embeddings
