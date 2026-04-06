"""Transformer Decoder Block with Pre-LayerNorm."""

import torch
import torch.nn as nn

from config.config import ModelConfig
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Decoder Block (GPT-2 style).

    Architecture:
        x -> LayerNorm -> Self-Attention -> + x -> LayerNorm -> FFN -> + x
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Layer normalization (pre-norm)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # Self-attention and feed-forward
        self.attention = MultiHeadSelfAttention(config)
        self.feedforward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional padding mask [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with pre-norm and residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x, attention_mask)
        x = x + residual

        # Feed-forward with pre-norm and residual
        residual = x
        x = self.ln2(x)
        x = self.feedforward(x)
        x = x + residual

        return x
