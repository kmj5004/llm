"""Tokenizer wrapper class for SentencePiece."""

from typing import List, Union

import sentencepiece as spm


class Tokenizer:
    """Wrapper class for SentencePiece tokenizer."""

    def __init__(self, model_path: str):
        """Initialize tokenizer from trained model.

        Args:
            model_path: Path to the .model file.
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Special token IDs
        self.pad_token_id = self.sp.pad_id()
        self.bos_token_id = self.sp.bos_id()
        self.eos_token_id = self.sp.eos_id()
        self.unk_token_id = self.sp.unk_id()

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp.get_piece_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.
            add_bos: Whether to add BOS token at the beginning.
            add_eos: Whether to add EOS token at the end.

        Returns:
            List of token IDs.
        """
        token_ids = self.sp.encode(text, out_type=int)

        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs or batch of token IDs.
            skip_special_tokens: Whether to remove special tokens from output.

        Returns:
            Decoded text string or list of strings.
        """
        # Handle batch input
        if token_ids and isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]

        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id
            }
            token_ids = [t for t in token_ids if t not in special_ids]

        return self.sp.decode(token_ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int = None,
        padding: bool = True
    ) -> List[List[int]]:
        """Encode batch of texts with optional padding.

        Args:
            texts: List of input texts.
            add_bos: Whether to add BOS token.
            add_eos: Whether to add EOS token.
            max_length: Maximum sequence length (truncate if longer).
            padding: Whether to pad sequences to max_length.

        Returns:
            List of token ID lists.
        """
        encoded = [self.encode(text, add_bos, add_eos) for text in texts]

        if max_length is not None:
            # Truncate sequences
            encoded = [ids[:max_length] for ids in encoded]

            if padding:
                # Pad sequences
                encoded = [
                    ids + [self.pad_token_id] * (max_length - len(ids))
                    for ids in encoded
                ]

        return encoded

    def get_piece(self, token_id: int) -> str:
        """Get the string piece for a token ID."""
        return self.sp.id_to_piece(token_id)

    def get_id(self, piece: str) -> int:
        """Get the token ID for a string piece."""
        return self.sp.piece_to_id(piece)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into string pieces."""
        return self.sp.encode(text, out_type=str)
