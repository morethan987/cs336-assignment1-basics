import os
from typing import IO, BinaryIO

import torch

from cs336_basics.layers import TransformerLM
from cs336_basics.tokenizer import BPE_Tokenizer
from cs336_basics.train_loop import TextGenerator


def _load_checkpoint(model: TransformerLM, src: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])


def main():
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        device=torch.device("cuda:0"),
    )
    _load_checkpoint(model, src="checkpoints/")

    eos_token = "<|endoftext|>"
    tokenizer = BPE_Tokenizer.from_files(
        vocab_filepath="outputs/bpe_trainer/20260725_204945/vocab.pkl",
        merges_filepath="outputs/bpe_trainer/20260725_204945/merges.pkl",
        special_tokens=[eos_token],
    )

    generator = TextGenerator(model, tokenizer)

    prompt = "Once upon a time, in a warm and sunny place, there was a big pit."
    output = generator.generate(prompt, eos_token, temp=0.8, p=0.9)
    print(f"Input prompt: {prompt}")
    print(f"Generated output: {output}")


if __name__ == "__main__":
    main()
