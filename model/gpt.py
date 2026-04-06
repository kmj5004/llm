"""GPT Model - Decoder-only Transformer for Korean Language Modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig
from .embedding import GPTEmbedding
from .transformer_block import TransformerBlock


class GPT(nn.Module):
    """GPT Language Model with weight tying."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = GPTEmbedding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (weight-tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and lm_head
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> dict:
        """
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Padding mask [batch_size, seq_len] (1=valid, 0=pad)
            labels: Target token IDs for loss computation [batch_size, seq_len]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Get embeddings
        x = self.embedding(input_ids)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Top-P (nucleus) sampling parameter.
            do_sample: Whether to sample (True) or use greedy decoding (False).
            eos_token_id: Token ID to stop generation.

        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Truncate if exceeds max sequence length
                if generated.shape[1] >= self.config.max_seq_len:
                    context = generated[:, -self.config.max_seq_len:]
                else:
                    context = generated

                # Forward pass
                outputs = self(context)
                logits = outputs["logits"][:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply sampling strategy
                if do_sample:
                    # Top-K filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                        logits[indices_to_remove] = float("-inf")

                    # Top-P (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float("-inf")

                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for EOS
                if (next_token == eos_token_id).all():
                    break

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_breakdown(self) -> dict:
        """Get parameter count breakdown by component."""
        breakdown = {
            "embedding": sum(p.numel() for p in self.embedding.parameters()),
            "transformer_blocks": sum(
                p.numel() for block in self.blocks for p in block.parameters()
            ),
            "final_layernorm": sum(p.numel() for p in self.ln_f.parameters()),
            # lm_head is weight-tied, so not counted separately
        }
        breakdown["total"] = self.count_parameters()
        return breakdown
