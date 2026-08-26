import mmap
import random
import re
import time

from numpy import average

from cs336_basics.tokenizer import BPE_Tokenizer

ts_tokenizer = BPE_Tokenizer.from_files(
    vocab_filepath="outputs/bpe_trainer/20260725_204945/vocab.pkl",
    merges_filepath="outputs/bpe_trainer/20260725_204945/merges.pkl",
    special_tokens=["<|endoftext|>"],
)

owt_tokenizer = BPE_Tokenizer.from_files(
    vocab_filepath="outputs/bpe_trainer/20260727_015403/vocab.pkl",
    merges_filepath="outputs/bpe_trainer/20260727_015403/merges.pkl",
    special_tokens=["<|endoftext|>"],
)


def sample_docs(
    file_path: str,
    n: int,
    special_tokens: list[str] | None,
) -> list[str]:
    """
    Sample n documents from file, sliced by special_tokens.

    Documents are separated by special tokens.
    Returned strings may contain special tokens, but special tokens
    will never be broken.

    Uses mmap + reservoir sampling, suitable for large files.
    """

    if n <= 0:
        return []

    if not special_tokens:
        with open(file_path, encoding="utf-8") as f:
            return [f.read()]

    # Longest first to avoid prefix matching problem:
    # "<|end|>" vs "<|endoftext|>"
    special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    samples: list[str] = []

    with open(file_path, "rb") as f:
        mm = mmap.mmap(
            f.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )

        # Decode only when needed.
        # We search in bytes to avoid decoding the whole file.
        byte_pattern = re.compile(b"|".join(re.escape(x.encode("utf-8")) for x in special_tokens))

        start = 0
        count = 0

        for match in byte_pattern.finditer(mm):
            end = match.end()

            # Include special token in previous document
            doc = mm[start:end]

            if doc:
                doc = doc.decode("utf-8")

                count += 1

                if len(samples) < n:
                    samples.append(doc)
                else:
                    # reservoir sampling
                    idx = random.randint(0, count - 1)
                    if idx < n:
                        samples[idx] = doc

            start = end

        # Remaining tail
        if start < len(mm):
            doc = mm[start:].decode("utf-8")

            count += 1

            if len(samples) < n:
                samples.append(doc)
            else:
                idx = random.randint(0, count - 1)
                if idx < n:
                    samples[idx] = doc

        mm.close()

    return samples


def compression_ratio(doc: str, tokenizer: BPE_Tokenizer):
    """
    Get the compression ratio of the tokenizer on document.
    """
    l1 = len(doc.encode("utf-8"))
    l2 = len(tokenizer.encode(doc))
    return l1 / l2


def test_average_compression_ratio():
    special_tokens = ["<|endoftext|>"]

    ts_file = "data/TinyStoriesV2-GPT4-valid.txt"
    docs = sample_docs(ts_file, 10, special_tokens)
    ts_res = [compression_ratio(doc, ts_tokenizer) for doc in docs]
    print(f"TinyStory compression ratio results:\n{[f'{x:.3f}' for x in ts_res]}")
    print(f"Average: {average(ts_res):.3f}")

    print()

    owt_file = "data/owt_valid.txt"
    docs = sample_docs(owt_file, 10, special_tokens)
    owt_res = [compression_ratio(doc, owt_tokenizer) for doc in docs]
    print(f"OpenWebText compression ratio results:\n{[f'{x:.3f}' for x in owt_res]}")
    print(f"Average: {average(owt_res):.3f}")


def test_owt_decode_with_ts():
    special_tokens = ["<|endoftext|>"]
    owt_file = "data/owt_valid.txt"
    docs = sample_docs(owt_file, 10, special_tokens)
    owt_res_with_ts = [compression_ratio(doc, ts_tokenizer) for doc in docs]
    ave_owt_res_with_ts = average(owt_res_with_ts)
    print(
        f"Compression ratio results of OpenWebText encoded with TinyStory tokenizer:\n{[f'{x:.3f}' for x in owt_res_with_ts]}"
    )
    print(f"Average: {ave_owt_res_with_ts:.3f}")

    print()

    owt_res = [compression_ratio(doc, owt_tokenizer) for doc in docs]
    ave_owt_res = average(owt_res)
    print(f"OpenWebText compression ratio results:\n{[f'{x:.3f}' for x in owt_res]}")
    print(f"Average: {ave_owt_res:.3f}")
    print(f"Relatively drops:{(ave_owt_res - ave_owt_res_with_ts) / (ave_owt_res):.4f}")

    print()

    print("One sample output:")
    doc = docs[5]
    encoded_with_ts = [ts_tokenizer.decode([idx]) for idx in ts_tokenizer.encode(doc)]
    print(f"Encoding results of OpenWebText with TinyStory tokenizer:\n{encoded_with_ts[:10]}")

    print()

    encoded_with_owt = [owt_tokenizer.decode([idx]) for idx in owt_tokenizer.encode(doc)]
    print(f"Encoding results of OpenWebText with OpenWebText tokenizer:\n{encoded_with_owt[:10]}")


def benchmark_tokenizer(
    file_path: str,
    tokenizer: BPE_Tokenizer,
):
    """
    Benchmark tokenizer.

    Returns:
        bytes/sec
        tokens/sec
        compression_ratio
    """

    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    input_bytes = len(text.encode("utf-8"))

    # warmup
    tokenizer.encode(text[:10000])

    start = time.perf_counter()
    tokens = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    num_tokens = len(tokens)

    return {
        "mb_per_sec": input_bytes / elapsed / 1024 / 1024,
        "tokens_per_sec": num_tokens / elapsed,
        "compression_ratio": input_bytes / num_tokens,
    }


def test_throughput():
    tests = [
        ("TinyStories", "data/TinyStoriesV2-GPT4-valid.txt", ts_tokenizer),
        ("OpenWebText", "data/owt_valid.txt", owt_tokenizer),
    ]

    print(f"{'Dataset':<15}{'MB/s':>12}{'tokens/s':>15}{'bytes/token':>15}")
    print("-" * 60)

    for name, file_path, tokenizer in tests:
        result = benchmark_tokenizer(file_path, tokenizer)

        print(
            f"{name:<15}"
            f"{result['mb_per_sec']:>12.2f}"
            f"{result['tokens_per_sec']:>15,.0f}"
            f"{result['compression_ratio']:>15.3f}"
        )


if __name__ == "__main__":
    # test_average_compression_ratio()
    # test_owt_decode_with_ts()
    test_throughput()
