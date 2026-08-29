import os
from typing import IO, BinaryIO

import torch

from cs336_basics.layers import TransformerLM, softmax
from cs336_basics.tokenizer import BPE_Tokenizer


class TextGenerator:
    def __init__(
        self, model: TransformerLM, tokenizer: BPE_Tokenizer, checkpoint_src: str | os.PathLike | BinaryIO | IO[bytes]
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self._load_checkpoint(checkpoint_src)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _load_checkpoint(self, src: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
        checkpoint = torch.load(src, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        eos_token: str,
        temp: float = 1.0,
        p: float = 0.9,
        max_generation: int | None = None,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # get eos_id
        eos_id = self.tokenizer.encode(eos_token)[0]

        prompt_length = x.shape[-1]
        left_budget = self.model.context_length - prompt_length
        if max_generation is None or max_generation > left_budget:
            max_generation = left_budget
        if max_generation <= 0:
            raise ValueError(
                f"Max generation length is negative! "
                f"Max context length is {self.model.context_length}, "
                f"Prompt length is {prompt_length}."
            )

        for _ in range(max_generation):
            logits: torch.Tensor = self.model(x)  # (1, seq_len, vocab_size)
            last_logits = logits[0, -1, :]  # (vocab_size,)

            # Temperature scaling & Softmax
            probs = softmax(last_logits.div_(temp), dim=-1)

            # Top-p
            sorted_prob, sorted_idxes = torch.sort(probs, descending=True)
            csum = torch.cumsum(sorted_prob, dim=-1).sub_(sorted_prob)
            mask = csum.greater_equal(p)
            sorted_prob.masked_fill_(mask, 0.0)

            # sampling
            sample_idx = torch.multinomial(sorted_prob, 1)  # (1,)
            next_token = sorted_idxes[sample_idx].unsqueeze(0)

            if next_token.item() == eos_id:
                break

            # cat
            x = torch.cat([x, next_token], dim=-1)

        generated_ids = x[0, prompt_length:]
        return self.tokenizer.decode(generated_ids.tolist())
