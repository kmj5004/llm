#!/usr/bin/env python3
"""Script to train the Korean GPT model."""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.config import ModelConfig, TrainingConfig
from tokenizer.tokenizer import Tokenizer
from model.gpt import GPT
from data.dataset import create_dataloader
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Korean GPT model")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    # Paths
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/korean_bpe.model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Korean GPT Model Training")
    print("=" * 60)

    # Check tokenizer exists
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer not found at {args.tokenizer_path}")
        print("Please run 'python scripts/train_tokenizer.py' first")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(args.tokenizer_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create model config
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id
    )

    # Create training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )

    # Print configuration
    print("\nModel Configuration:")
    print(f"  vocab_size: {model_config.vocab_size}")
    print(f"  max_seq_len: {model_config.max_seq_len}")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  d_ff: {model_config.d_ff}")
    print(f"  dropout: {model_config.dropout}")

    print("\nTraining Configuration:")
    print(f"  batch_size: {training_config.batch_size}")
    print(f"  learning_rate: {training_config.learning_rate}")
    print(f"  max_epochs: {training_config.max_epochs}")
    print(f"  device: {training_config.device}")

    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")
    model = GPT(model_config)

    # Print parameter count
    param_breakdown = model.count_parameters_breakdown()
    print(f"\nParameter count:")
    print(f"  Embedding: {param_breakdown['embedding']:,}")
    print(f"  Transformer blocks: {param_breakdown['transformer_blocks']:,}")
    print(f"  Final LayerNorm: {param_breakdown['final_layernorm']:,}")
    print(f"  Total: {param_breakdown['total']:,}")

    # Create dataloader
    print("\n" + "-" * 60)
    print("Creating dataloader...")
    train_dataloader = create_dataloader(
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        batch_size=training_config.batch_size,
        corpus_names=["nsmc", "kcbert"],
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    print(f"Training batches: {len(train_dataloader):,}")

    # Create trainer
    print("\n" + "-" * 60)
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "-" * 60)
    print("Starting training...")
    trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {training_config.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
