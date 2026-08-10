import os
import tempfile
from datetime import datetime as dt
from datetime import timedelta, timezone

import numpy as np

from cs336_basics.tokenizer import BPE_Tokenizer

output_dir = "outputs/bpe_tokenizer"


def encode_to_bin_stream(tokenizer: BPE_Tokenizer, input_file: str, output_file: str):
    with open(output_file, "wb") as fout, open(input_file, encoding="utf-8") as fin:
        buffer = np.empty(1024 * 1024, dtype=np.uint16)
        idx = 0
        for token in tokenizer.encode_iterable(fin):
            buffer[idx] = token
            idx += 1

            if idx == len(buffer):
                fout.write(buffer.tobytes())
                idx = 0

        if idx:
            fout.write(buffer[:idx].tobytes())


def encode_tinystory():
    ts_tokenizer = BPE_Tokenizer.from_files(
        vocab_filepath="outputs/bpe_trainer/20260725_204945/vocab.pkl",
        merges_filepath="outputs/bpe_trainer/20260725_204945/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    time_stmp = dt.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, time_stmp)
    os.makedirs(save_dir, exist_ok=True)
    v_save = os.path.join(save_dir, "TinyStoriesV2-GPT4-valid-tokenized.bin")
    t_save = os.path.join(save_dir, "TinyStoriesV2-GPT4-train-tokenized.bin")

    valid_file = "data/TinyStoriesV2-GPT4-valid.txt"
    encode_to_bin_stream(ts_tokenizer, valid_file, v_save)

    train_file = "data/TinyStoriesV2-GPT4-train.txt"
    encode_to_bin_stream(ts_tokenizer, train_file, t_save)


def encode_owt():
    owt_tokenizer = BPE_Tokenizer.from_files(
        vocab_filepath="outputs/bpe_trainer/20260727_015403/vocab.pkl",
        merges_filepath="outputs/bpe_trainer/20260727_015403/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    time_stmp = dt.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, time_stmp)
    os.makedirs(save_dir, exist_ok=True)
    v_save = os.path.join(save_dir, "owt_valid_tokenized.bin")
    t_save = os.path.join(save_dir, "owt_train_tokenized.bin")

    valid_file = "data/owt_valid.txt"
    encode_to_bin_stream(owt_tokenizer, valid_file, v_save)

    train_file = "data/owt_train.txt"
    encode_to_bin_stream(owt_tokenizer, train_file, t_save)


def tiny_test():
    ts_tokenizer = BPE_Tokenizer.from_files(
        vocab_filepath="outputs/bpe_trainer/20260725_204945/vocab.pkl",
        merges_filepath="outputs/bpe_trainer/20260725_204945/merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    valid_file = "data/TinyStoriesV2-GPT4-valid.txt"

    with open(valid_file, encoding="utf-8") as fin:
        original_text = fin.read()

    with tempfile.NamedTemporaryFile() as tmp:
        # test streaming encode -> write
        encode_to_bin_stream(
            ts_tokenizer,
            valid_file,
            tmp.name,
        )

        # load streamed result
        loaded_tokens = np.fromfile(
            tmp.name,
            dtype=np.uint16,
        )

    # decode check
    decoded_text = ts_tokenizer.decode(loaded_tokens.tolist())

    assert decoded_text == original_text, (
        f"Decoded text mismatch\noriginal length: {len(original_text)}\ndecoded length: {len(decoded_text)}"
    )

    print("✅ streaming tokenize -> save -> load -> decode OK")


if __name__ == "__main__":
    tiny_test()
    # encode_tinystory()
    # encode_owt()
