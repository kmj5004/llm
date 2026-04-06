"""SentencePiece BPE tokenizer trainer for Korean text."""

import os
import tempfile
from typing import List, Optional

import sentencepiece as spm
from Korpora import Korpora

from config.config import TokenizerConfig


def extract_texts_from_korpora(corpus_names: List[str] = None) -> List[str]:
    """Extract texts from Korpora datasets.

    Args:
        corpus_names: List of corpus names to load. Defaults to nsmc and kcbert.

    Returns:
        List of text strings.
    """
    if corpus_names is None:
        corpus_names = ["nsmc", "kcbert"]

    texts = []

    for corpus_name in corpus_names:
        print(f"Loading corpus: {corpus_name}")
        try:
            corpus = Korpora.load(corpus_name)

            # Handle different corpus structures
            if hasattr(corpus, "train"):
                if hasattr(corpus.train, "texts"):
                    texts.extend(corpus.train.texts)
                elif hasattr(corpus.train, "text"):
                    texts.extend([item.text for item in corpus.train])

            if hasattr(corpus, "test"):
                if hasattr(corpus.test, "texts"):
                    texts.extend(corpus.test.texts)
                elif hasattr(corpus.test, "text"):
                    texts.extend([item.text for item in corpus.test])

            if hasattr(corpus, "dev"):
                if hasattr(corpus.dev, "texts"):
                    texts.extend(corpus.dev.texts)
                elif hasattr(corpus.dev, "text"):
                    texts.extend([item.text for item in corpus.dev])

            print(f"  Loaded {len(texts)} texts so far")

        except Exception as e:
            print(f"  Warning: Could not load {corpus_name}: {e}")
            continue

    return texts


def train_tokenizer(
    texts: List[str],
    config: Optional[TokenizerConfig] = None,
    output_dir: str = "tokenizer"
) -> str:
    """Train SentencePiece BPE tokenizer.

    Args:
        texts: List of training texts.
        config: Tokenizer configuration.
        output_dir: Directory to save the model.

    Returns:
        Path to the trained model file.
    """
    if config is None:
        config = TokenizerConfig()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write texts to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for text in texts:
            if text and text.strip():
                f.write(text.strip() + "\n")
        temp_path = f.name

    try:
        # Train SentencePiece model
        model_prefix = os.path.join(output_dir, "korean_bpe")

        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=config.vocab_size,
            model_type=config.model_type,
            character_coverage=config.character_coverage,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece=config.pad_token,
            bos_piece=config.bos_token,
            eos_piece=config.eos_token,
            unk_piece=config.unk_token,
            user_defined_symbols=[],
            max_sentence_length=4096,
            shuffle_input_sentence=True,
        )

        print(f"Tokenizer saved to: {model_prefix}.model")
        return f"{model_prefix}.model"

    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def main():
    """Main function to train tokenizer from Korpora."""
    print("Extracting texts from Korpora...")
    texts = extract_texts_from_korpora()

    if not texts:
        raise ValueError("No texts extracted from Korpora")

    print(f"Total texts: {len(texts)}")
    print("Training tokenizer...")

    config = TokenizerConfig()
    model_path = train_tokenizer(texts, config)

    print(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
