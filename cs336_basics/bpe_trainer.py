import heapq
import os
import pickle
from collections import Counter
from datetime import datetime as dt
from datetime import timedelta, timezone
from itertools import pairwise

from cs336_basics.pretokenization import init_word_tokens, pretokenize_parallel


class Reverse_Byte_Pair:
    def __init__(self, bp: tuple[bytes, bytes]) -> None:
        self.byte_pair = bp

    def __lt__(self, other: "Reverse_Byte_Pair") -> bool:
        return self.byte_pair > other.byte_pair

    def get(self) -> tuple[bytes, bytes]:
        return self.byte_pair


class BPE_Trainer:
    def __init__(
        self, input_path: str | os.PathLike[str], vocab_size: int, special_tokens: list[str], num_processes: int = 4
    ) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size  # including initial bytes, special tokens and bpe genetated tokens
        self.num_processes = num_processes
        self.output_dir = "outputs/bpe_trainer"

        self.special_tokens = [s.encode("utf-8") for s in special_tokens]
        self.num_special_tokens = len(special_tokens)

        self.vocab = self._init_vocab()
        self.merges: list[tuple[bytes, bytes]] = []

    def _init_vocab(self) -> dict[int, bytes]:
        """
        Initialize vocabulary with 0~255 and special tokens
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for idx in range(self.num_special_tokens):
            vocab[256 + idx] = self.special_tokens[idx]
        return vocab

    def pickling(self) -> tuple[str, str]:
        """
        Save vocabularies and merges into files with pickle
        Returns:
            paths(tuple[str, str]): a tuple represents vocab path and merges path
        """
        time_stmp = dt.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, time_stmp)
        os.makedirs(save_dir, exist_ok=True)
        v_path = os.path.join(save_dir, "vocab.pkl")
        m_path = os.path.join(save_dir, "merges.pkl")
        with open(v_path, "wb") as f:
            pickle.dump(self.vocab, f)
        with open(m_path, "wb") as f:
            pickle.dump(self.merges, f)
        return v_path, m_path

    @staticmethod
    def unpickling(vocab_path: str, merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Load vocabularies and merges from files with pickle
        """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return vocab, merges

    def train(self) -> None:
        """
        Train BPE model
        """
        # w_freq = pretokenize(self.input_path, self.special_tokens)
        w_freq = pretokenize_parallel(self.input_path, self.special_tokens, self.num_processes)
        w_tokens = init_word_tokens(w_freq)
        bp_words: dict[tuple[bytes, bytes], set[str]] = {}
        bp_counter: Counter[tuple[bytes, bytes]] = Counter()  # store true coount data
        bp_heap: list[tuple[int, Reverse_Byte_Pair]] = []  # to get most freqent byte pair
        for word in w_freq:
            tokens = w_tokens[word]
            for byte_pair in pairwise(tokens):
                bp_counter[byte_pair] += w_freq[word]
                bp_words.setdefault(byte_pair, set()).add(word)
        bp_heap = [(-freq, Reverse_Byte_Pair(pair)) for pair, freq in bp_counter.items()]
        heapq.heapify(bp_heap)

        # for loop to merge bytes
        for _ in range(self.vocab_size - len(self.vocab)):
            bp_merge: tuple[bytes, bytes] | None = None
            while bp_heap:  # lazy delete
                neg_freq, re_candidate = heapq.heappop(bp_heap)
                candidate = re_candidate.get()
                if candidate in bp_counter and -neg_freq == bp_counter[candidate]:
                    bp_merge = candidate
                    break
            assert bp_merge is not None

            tk_new = bp_merge[0] + bp_merge[1]
            self.merges.append(bp_merge)
            self.vocab[len(self.vocab)] = tk_new

            def _cnt_rm(key: tuple[bytes, bytes], v: int):
                if key not in bp_counter:
                    return
                bp_counter[key] -= v
                if bp_counter[key] <= 0:
                    bp_counter.pop(key)
                    bp_words.pop(key)
                    return
                heapq.heappush(bp_heap, (-bp_counter[key], Reverse_Byte_Pair(key)))

            def _cnt_add(key: tuple[bytes, bytes], word: str):
                bp_counter[key] += w_freq[word]
                heapq.heappush(bp_heap, (-bp_counter[key], Reverse_Byte_Pair(key)))
                bp_words.setdefault(key, set()).add(word)

            for word in bp_words[bp_merge]:
                tks = w_tokens[word]
                lp = 0  # left pointer
                while lp < len(tks) - 1:
                    rp = lp + 1  # right pointer, always exist
                    if tks[lp] == bp_merge[0] and tks[rp] == bp_merge[1]:
                        if lp != 0:  # check left is valid
                            _cnt_rm((tks[lp - 1], bp_merge[0]), w_freq[word])
                            _cnt_add((tks[lp - 1], tk_new), word)
                        if rp != len(tks) - 1:  # check right is valid
                            _cnt_rm((bp_merge[1], tks[rp + 1]), w_freq[word])
                            _cnt_add((tk_new, tks[rp + 1]), word)
                        tks[lp : rp + 1] = [tks[lp] + tks[rp]]
                        _cnt_rm(bp_merge, w_freq[word])
                    lp += 1

    def ans_for_adapter(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
        return self.vocab, self.merges
