import heapq
import mmap
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import regex as re


def find_chunk_boundaries(
    file_path: str | os.PathLike[str],
    desired_num_chunks: int,
    special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # with open(file_path, "rb") as f:
    with open(file_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # Get total file size in bytes
        file_size = len(mm)

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        # tackel empty
        if len(special_tokens) == 0:
            return chunk_boundaries

        # attach boundaries to the nearest special token
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            chunk_boundaries[bi] = min(
                (pos for token in special_tokens if ((pos := mm.find(token, initial_position)) != -1)),
                default=file_size - 1,
            )

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


def pretokenize(input_path: str | os.PathLike[str], special_tokens: list[bytes]) -> Counter[str]:
    """
    Pretokenize a document with an internal pattern
    Args:
        doc(str): document needed to be pre-tokenized
    Returns:
        w_freq(dict[str, int]): word frequnces
    """
    return word_count(input_path, special_tokens, 0, -1)


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
    boundaries = find_chunk_boundaries(input_path, num_processes, special_tokens)
    tasks = [(input_path, special_tokens, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    with ProcessPoolExecutor(max_workers=num_processes) as ex:
        futs = [ex.submit(word_count, *t) for t in tasks]
        partials = [f.result() for f in futs]
    w_freq = Counter()
    for p in partials:
        w_freq += p
    return w_freq


def word_count(
    input_path: str | os.PathLike[str], special_tokens: list[bytes], start: int, end: int = -1
) -> Counter[str]:
    w_freq = Counter()
    pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    with open(input_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        # end of file
        if end < 0:
            end = len(mm)
        heap: list[tuple[int, bytes]] = []
        # heap init
        for token in special_tokens:
            pos = mm.find(token, start, end)
            if pos != -1:
                heapq.heappush(heap, (pos, token))

        prev = start
        while heap:
            pos, token = heapq.heappop(heap)
            # consume normal text
            if pos > prev:
                text = mm[prev:pos].decode("utf-8")
                w_freq.update(m.group() for m in pat.finditer(text))
            prev = pos + len(token)
            # find next token
            nxt_pos = mm.find(token, prev, end)
            if nxt_pos != -1:
                heapq.heappush(heap, (nxt_pos, token))

        # consume text left
        if prev < end:
            text = mm[prev:end].decode("utf-8")
            w_freq.update(m.group() for m in pat.finditer(text))

    return w_freq
