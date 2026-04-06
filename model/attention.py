"""Multi-Head Self-Attention with Causal Masking."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with causal masking for autoregressive generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Causal mask (registered as buffer, not parameter)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len),
                diagonal=1
            ).bool()
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional padding mask [batch_size, seq_len]
                           (1 for valid tokens, 0 for padding)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq]
            padding_mask = (1 - attention_mask).bool().unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask, float("-inf"))

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output
