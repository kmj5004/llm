"""Text generation with Top-K/Top-P sampling."""

from typing import List, Optional

import torch
import torch.nn.functional as F

from model.gpt import GPT
from tokenizer.tokenizer import Tokenizer


class TextGenerator:
    """Text generator with various sampling strategies."""

    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        device: str = "mps"
    ):
        """
        Args:
            model: Trained GPT model.
            tokenizer: Trained tokenizer.
            device: Device to run inference on.
        """
        self.model = model
        self.tokenizer = tokenizer

        # Set device
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_length: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Keep only top-k tokens for sampling.
            top_p: Keep tokens with cumulative probability <= top_p.
            do_sample: Whether to sample (True) or use greedy decoding (False).
            num_return_sequences: Number of sequences to generate.

        Returns:
            List of generated text strings.
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Repeat for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)

        # Generate
        with torch.no_grad():
            generated_ids = self._generate_tokens(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )

        # Decode
        generated_texts = []
        for seq in generated_ids:
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_length: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Top-P (nucleus) filtering.
            do_sample: Whether to sample or use greedy decoding.

        Returns:
            Generated token IDs [batch_size, total_length]
        """
        generated = input_ids
        eos_token_id = self.tokenizer.eos_token_id
        max_seq_len = self.model.config.max_seq_len

        for _ in range(max_length):
            # Truncate context if needed
            if generated.shape[1] >= max_seq_len:
                context = generated[:, -max_seq_len:]
            else:
                context = generated

            # Forward pass
            outputs = self.model(context)
            logits = outputs["logits"][:, -1, :]  # Last token logits

            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # Get next token
            if do_sample:
                next_token = self._sample(logits, top_k, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Sample from logits with Top-K and Top-P filtering.

        Args:
            logits: Logits [batch_size, vocab_size]
            top_k: Top-K filtering parameter.
            top_p: Top-P (nucleus) filtering parameter.

        Returns:
            Sampled token IDs [batch_size, 1]
        """
        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            # Scatter back to original order
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def generate_interactive(self) -> None:
        """Interactive text generation loop."""
        print("Interactive Korean GPT Text Generator")
        print("=" * 50)
        print("Commands:")
        print("  quit/exit - Exit the generator")
        print("  temp=X    - Set temperature (e.g., temp=0.8)")
        print("  topk=X    - Set top_k (e.g., topk=50)")
        print("  topp=X    - Set top_p (e.g., topp=0.9)")
        print("  len=X     - Set max_length (e.g., len=100)")
        print("  examples  - Show example prompts")
        print("=" * 50)

        example_prompts = [
            # 일상 대화
            "오늘 날씨가 정말",
            "아침에 일어나서",
            "주말에는 보통",

            # 감정/의견
            "이 영화는 정말",
            "요즘 기분이",
            "내가 좋아하는 것은",

            # 이야기 시작
            "옛날 옛적에",
            "어느 날 갑자기",
            "그 사람은",

            # 음식/장소
            "맛있는 음식을 먹으면",
            "서울에서 가장",
            "한국의 겨울은",

            # 생각/철학
            "인생에서 가장 중요한 것은",
            "행복이란",
            "사람들은 왜",
        ]

        temperature = 1.0
        top_k = 50
        top_p = 0.9
        max_length = 100

        # Show some example prompts at start
        import random
        print("\nExample prompts to try:")
        for p in random.sample(example_prompts, min(5, len(example_prompts))):
            print(f"  - {p}")

        while True:
            try:
                prompt = input("\nPrompt: ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                # Parse settings
                if prompt.startswith("temp="):
                    temperature = float(prompt.split("=")[1])
                    print(f"Temperature set to {temperature}")
                    continue
                elif prompt.startswith("topk="):
                    top_k = int(prompt.split("=")[1])
                    print(f"Top-K set to {top_k}")
                    continue
                elif prompt.startswith("topp="):
                    top_p = float(prompt.split("=")[1])
                    print(f"Top-P set to {top_p}")
                    continue
                elif prompt.startswith("len="):
                    max_length = int(prompt.split("=")[1])
                    print(f"Max length set to {max_length}")
                    continue
                elif prompt.lower() == "examples":
                    print("\nExample prompts:")
                    print("-" * 40)
                    print("[일상 대화]")
                    print("  - 오늘 날씨가 정말")
                    print("  - 아침에 일어나서")
                    print("  - 주말에는 보통")
                    print("\n[감정/의견]")
                    print("  - 이 영화는 정말")
                    print("  - 요즘 기분이")
                    print("  - 내가 좋아하는 것은")
                    print("\n[이야기 시작]")
                    print("  - 옛날 옛적에")
                    print("  - 어느 날 갑자기")
                    print("  - 그 사람은")
                    print("\n[음식/장소]")
                    print("  - 맛있는 음식을 먹으면")
                    print("  - 서울에서 가장")
                    print("  - 한국의 겨울은")
                    print("\n[생각/철학]")
                    print("  - 인생에서 가장 중요한 것은")
                    print("  - 행복이란")
                    print("  - 사람들은 왜")
                    print("-" * 40)
                    continue

                # Generate
                print("\nGenerating...")
                generated = self.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True
                )

                print("\n" + "=" * 50)
                print("Generated:")
                print(generated[0])
                print("=" * 50)

            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
