#!/usr/bin/env python3
"""Script to train the SentencePiece tokenizer."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.trainer import extract_texts_from_korpora, train_tokenizer
from tokenizer.tokenizer import Tokenizer
from config.config import TokenizerConfig


def main():
    print("=" * 60)
    print("Korean BPE Tokenizer Training")
    print("=" * 60)

    # Configuration
    config = TokenizerConfig(
        vocab_size=8000,
        model_type="bpe",
        character_coverage=0.9995
    )

    print(f"\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model type: {config.model_type}")
    print(f"  Character coverage: {config.character_coverage}")

    # Extract texts
    print("\n" + "-" * 60)
    print("Step 1: Extracting texts from Korpora...")
    texts = extract_texts_from_korpora(["nsmc", "kcbert"])

    if not texts:
        print("Error: No texts extracted from Korpora")
        print("Make sure you have downloaded the corpora:")
        print("  python -c \"from Korpora import Korpora; Korpora.fetch('nsmc')\"")
        print("  python -c \"from Korpora import Korpora; Korpora.fetch('kcbert')\"")
        sys.exit(1)

    print(f"\nTotal texts extracted: {len(texts):,}")

    # Train tokenizer
    print("\n" + "-" * 60)
    print("Step 2: Training SentencePiece BPE tokenizer...")

    model_path = train_tokenizer(texts, config, output_dir="tokenizer")

    # Test tokenizer
    print("\n" + "-" * 60)
    print("Step 3: Testing tokenizer...")

    tokenizer = Tokenizer(model_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test encode/decode
    test_sentences = [
        "안녕하세요, 반갑습니다!",
        "오늘 날씨가 정말 좋네요.",
        "한국어 언어 모델을 만들고 있습니다.",
        "인공지능 기술이 빠르게 발전하고 있습니다."
    ]

    print("\nEncode/Decode test:")
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        pieces = tokenizer.tokenize(sentence)

        print(f"\n  Original: {sentence}")
        print(f"  Tokens:   {tokens[:20]}..." if len(tokens) > 20 else f"  Tokens:   {tokens}")
        print(f"  Pieces:   {pieces[:10]}..." if len(pieces) > 10 else f"  Pieces:   {pieces}")
        print(f"  Decoded:  {decoded}")

    print("\n" + "=" * 60)
    print(f"Tokenizer training complete!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
