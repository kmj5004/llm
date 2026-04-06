"""Dataset class for Korean text data from Korpora."""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from Korpora import Korpora

from tokenizer.tokenizer import Tokenizer


class KoreanTextDataset(Dataset):
    """Korean text dataset with sliding window for language modeling."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_len: int = 256,
        corpus_names: List[str] = None,
        stride: int = None
    ):
        """
        Args:
            tokenizer: Trained tokenizer instance.
            max_seq_len: Maximum sequence length.
            corpus_names: List of Korpora corpus names to load.
            stride: Sliding window stride. Defaults to max_seq_len (non-overlapping).
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len

        if corpus_names is None:
            corpus_names = ["nsmc", "kcbert"]

        # Load and tokenize all texts
        self.examples = self._prepare_examples(corpus_names)

    def _prepare_examples(self, corpus_names: List[str]) -> List[torch.Tensor]:
        """Load texts and create training examples with sliding window."""
        texts = self._load_texts(corpus_names)

        # Tokenize all texts into one long sequence
        all_tokens = []
        for text in texts:
            if text and text.strip():
                tokens = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                all_tokens.extend(tokens)
                # Add EOS between documents
                all_tokens.append(self.tokenizer.eos_token_id)

        print(f"Total tokens: {len(all_tokens):,}")

        # Create examples with sliding window
        examples = []
        for i in range(0, len(all_tokens) - self.max_seq_len, self.stride):
            chunk = all_tokens[i:i + self.max_seq_len + 1]  # +1 for labels
            if len(chunk) == self.max_seq_len + 1:
                examples.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Created {len(examples):,} training examples")
        return examples

    def _load_texts(self, corpus_names: List[str]) -> List[str]:
        """Load texts from Korpora datasets."""
        texts = []

        for corpus_name in corpus_names:
            print(f"Loading corpus: {corpus_name}")
            try:
                corpus = Korpora.load(corpus_name)

                # Extract texts from different corpus structures
                for split_name in ["train", "test", "dev"]:
                    split = getattr(corpus, split_name, None)
                    if split is None:
                        continue

                    if hasattr(split, "texts"):
                        texts.extend(split.texts)
                    elif hasattr(split, "text"):
                        texts.extend([item.text for item in split])
                    elif hasattr(split, "__iter__"):
                        for item in split:
                            if hasattr(item, "text"):
                                texts.append(item.text)

                print(f"  Loaded {len(texts)} texts so far")

            except Exception as e:
                print(f"  Warning: Could not load {corpus_name}: {e}")
                continue

        return texts

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: Token IDs [max_seq_len]
            labels: Target token IDs [max_seq_len]
        """
        tokens = self.examples[idx]
        input_ids = tokens[:-1]  # All but last
        labels = tokens[1:]      # All but first (shifted by 1)
        return input_ids, labels


def create_dataloader(
    tokenizer: Tokenizer,
    max_seq_len: int = 256,
    batch_size: int = 32,
    corpus_names: List[str] = None,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Korean text dataset.

    Args:
        tokenizer: Trained tokenizer instance.
        max_seq_len: Maximum sequence length.
        batch_size: Batch size.
        corpus_names: List of corpus names.
        stride: Sliding window stride.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers (0 for MPS).
        pin_memory: Whether to pin memory (False for MPS).

    Returns:
        DataLoader instance.
    """
    dataset = KoreanTextDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        corpus_names=corpus_names,
        stride=stride
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    return dataloader
