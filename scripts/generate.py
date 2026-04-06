#!/usr/bin/env python3
"""Script to generate text using trained Korean GPT model."""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.config import ModelConfig
from tokenizer.tokenizer import Tokenizer
from model.gpt import GPT
from inference.generator import TextGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with Korean GPT")

    # Model paths
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/korean_bpe.model",
                        help="Path to tokenizer model")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation (if not provided, runs interactive mode)")
    parser.add_argument("--max-length", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-P (nucleus) sampling parameter")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")

    # Device
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"])

    # Interactive mode
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")

    # Show examples
    parser.add_argument("--examples", action="store_true",
                        help="Run generation with example prompts")

    return parser.parse_args()


EXAMPLE_PROMPTS = [
    ("일상", "오늘 날씨가 정말"),
    ("일상", "아침에 일어나서"),
    ("감정", "이 영화는 정말"),
    ("이야기", "옛날 옛적에"),
    ("이야기", "어느 날 갑자기"),
    ("음식", "맛있는 음식을 먹으면"),
    ("장소", "서울에서 가장"),
    ("철학", "인생에서 가장 중요한 것은"),
    ("철학", "행복이란"),
]


def load_model(checkpoint_path: str, tokenizer: Tokenizer, device: str) -> GPT:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Determine device
    if device == "mps" and torch.backends.mps.is_available():
        map_location = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Get config from checkpoint or use default
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Fallback to default config
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id
        )

    # Create model and load weights
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded with {model.count_parameters():,} parameters")

    return model


def main():
    args = parse_args()

    print("=" * 60)
    print("Korean GPT Text Generator")
    print("=" * 60)

    # Check files exist
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer not found at {args.tokenizer_path}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first with 'python scripts/train_model.py'")
        sys.exit(1)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(args.tokenizer_path)

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, tokenizer, args.device)

    # Create generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=args.device
    )

    # Generate text
    if args.examples:
        # Run with example prompts
        print("\n" + "=" * 60)
        print("Running with example prompts")
        print("=" * 60)

        for category, prompt in EXAMPLE_PROMPTS:
            print(f"\n[{category}] Prompt: {prompt}")
            print("-" * 40)

            generated = generator.generate(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=not args.greedy,
                num_return_sequences=1
            )
            print(f"Generated: {generated[0]}")

        print("\n" + "=" * 60)

    elif args.interactive or args.prompt is None:
        # Interactive mode
        generator.generate_interactive()
    else:
        # Single generation
        print("\n" + "-" * 60)
        print(f"Prompt: {args.prompt}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-K: {args.top_k}")
        print(f"Top-P: {args.top_p}")
        print(f"Max length: {args.max_length}")
        print("-" * 60)

        print("\nGenerating...")
        generated = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            num_return_sequences=args.num_samples
        )

        print("\n" + "=" * 60)
        for i, text in enumerate(generated):
            if args.num_samples > 1:
                print(f"\n[Sample {i + 1}]")
            print(text)
        print("=" * 60)


if __name__ == "__main__":
    main()
