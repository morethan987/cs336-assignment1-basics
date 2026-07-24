import os
import regex as re
from collections import Counter
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # tackel empty
    if len(special_tokens) == 0:
        return chunk_boundaries

    base_chunk_size = 4096  # Read ahead by 4k bytes at a time
    max_special_token_len = max([len(token) for token in special_tokens])
    mini_chunk_size = base_chunk_size + max_special_token_len  # set overlap

    # attach boundaries to the nearest special token
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        position = initial_position  # A slide position cursor
        while True:
            file.seek(position)
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            min_found_at = mini_chunk_size + 1
            for token in special_tokens:
                found_at = mini_chunk.find(token)
                if found_at != -1:
                    min_found_at = min(min_found_at, found_at)
            if min_found_at != mini_chunk_size + 1:
                chunk_boundaries[bi] = position + min_found_at
                break
            position += base_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_by_special_tokens(chunk: str, special_tokens: list[bytes]) -> list[str]:
    if len(special_tokens) == 0:
        return [chunk]
    special_tokens_str = [b.decode("utf-8") for b in special_tokens]
    split_pat = "|".join(re.escape(token) for token in special_tokens_str)
    return re.split(split_pat, chunk)


def init_word_tokens(words: list[str] | Counter[str]) -> dict[str, list[bytes]]:
    """
    Convert words to basic byte tokens
    Args:
        words(list[str]): word list
    Returns:
        w_tokens(dict[str, list[bytes]]): the tokens a word contains
    """
    w_tokens: dict[str, list[bytes]] = {}
    for word in words:
        w_tokens[word] = [bytes([idx]) for idx in word.encode("utf-8")]
    return w_tokens


def pretokenize(
    input_path: str | os.PathLike[str], special_tokens: list[bytes], num_processes: int = 4
) -> Counter[str]:
    """
    Pretokenize a document with an internal pattern
    Args:
        doc(str): document needed to be pre-tokenized
    Returns:
        w_freq(dict[str, int]): word frequnces
    """
    w_freq = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)
        # TODO: parallelize it
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")
            docs = split_by_special_tokens(chunk, special_tokens)
            for doc in docs:
                words = re.findall(PAT, doc)
                for word in words:
                    assert isinstance(word, str)
                    w_freq[word] += 1
    return w_freq


def pretokenize_parallel(
    input_path: str | os.PathLike[str], special_tokens: list[bytes], num_processes: int = 4
) -> Counter[str]:
    """
    Parallele Pretokenize a document with an internal pattern
    Args:
        doc(str): document needed to be pre-tokenized
    Returns:
        w_freq(dict[str, int]): word frequnces
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)
    tasks = [(input_path, boundaries[i], boundaries[i + 1], special_tokens) for i in range(len(boundaries) - 1)]
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        futs = [ex.submit(word_count, *t) for t in tasks]
        partials = [f.result() for f in futs]
    w_freq = Counter()
    for p in partials:
        w_freq += p
    return w_freq


def word_count(input_path: str | os.PathLike[str], start: int, end: int, special_tokens: list[bytes]) -> Counter[str]:
    w_freq = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        docs = split_by_special_tokens(chunk, special_tokens)
        for doc in docs:
            words = re.findall(PAT, doc)
            for word in words:
                assert isinstance(word, str)
                w_freq[word] += 1
    return w_freq
