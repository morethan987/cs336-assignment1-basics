import heapq
import os
import pickle
from collections.abc import Iterable, Iterator
from itertools import pairwise

from .pretokenization import PAT


class BPE_Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab  # should be the same order with merges
        self.vocab_rev = {v: k for k, v in vocab.items()}
        self.merge_ranks = {
            (self.vocab_rev[tk1], self.vocab_rev[tk2]): self.vocab_rev[tk1 + tk2] for tk1, tk2 in merges
        }
        self.special_tokens = special_tokens
        self.memo: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BPE_Tokenizer":
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges.
        """
        if not os.path.isfile(vocab_filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_filepath}")
        if not os.path.isfile(merges_filepath):
            raise FileNotFoundError(f"Merges file not found: {merges_filepath}")

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        if not self.special_tokens:
            return self._encode_normal_text(text)

        res: list[int] = []
        heap: list[tuple[int, int, str]] = []
        # heap init
        for tk in self.special_tokens:
            pos = text.find(tk)
            if pos != -1:
                heapq.heappush(heap, (pos, -len(tk), tk))

        prev = 0
        while heap:
            pos, _, tk = heapq.heappop(heap)
            if pos >= prev:
                res += self._encode_normal_text(text[prev:pos])
                res.append(self.vocab_rev[tk.encode("utf-8")])
                prev = pos + len(tk)
            nxt = text.find(tk, prev)
            if nxt != -1:
                heapq.heappush(heap, (nxt, -len(tk), tk))
        if prev < len(text):
            res += self._encode_normal_text(text[prev:])

        return res

    def _encode_normal_text(self, chunk: str) -> list[int]:
        """
        Encode an input text without special tokens into a sequence of token IDs.
        """
        res: list[int] = []
        for match in PAT.finditer(chunk):
            word = match.group()
            if word in self.memo:
                res += self.memo[word]
                continue

            tks_idx = [self.vocab_rev[bytes([idx])] for idx in word.encode("utf-8")]
            while True:
                mrg_idx = len(self.vocab)
                mrg: tuple[int, int] = (-1, -1)
                for pair in pairwise(tks_idx):
                    if pair in self.merge_ranks and self.merge_ranks[pair] < mrg_idx:
                        mrg_idx = self.merge_ranks[pair]
                        mrg = pair
                if mrg_idx == len(self.vocab):
                    break

                i = 0
                while i < len(tks_idx) - 1:
                    if tks_idx[i] == mrg[0] and tks_idx[i + 1] == mrg[1]:
                        tks_idx[i : i + 2] = [mrg_idx]
                    else:
                        i += 1

            self.memo[word] = tks_idx
            res += tks_idx

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        DO NOT pass a large single line file.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byts = b"".join(self.vocab[id] for id in ids)
        return byts.decode("utf-8", errors="replace")
