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
    _load_checkpoint(
        model, src="/root/cs336/cs336-assignment1-basics/checkpoints/mini_test_20260830_114028/checkpoint_latest.pt"
    )

    eos_token = "<|endoftext|>"
    tokenizer = BPE_Tokenizer.from_files(
        vocab_filepath="outputs/bpe_trainer/20260725_204945/vocab.pkl",
        merges_filepath="outputs/bpe_trainer/20260725_204945/merges.pkl",
        special_tokens=[eos_token],
    )

    generator = TextGenerator(model, tokenizer)

    prompt = "Once upon a time, in a warm and sunny place, there was a big pit."
    output = generator.generate(prompt, eos_token, temp=0.8, p=0.9)
    print(f"Input prompt:\n{prompt}")
    print("Original text piece:")
    print(
        'Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.\nTom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."\nSam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."'
    )
    print(f"Generated output:\n{output}")


if __name__ == "__main__":
    main()
