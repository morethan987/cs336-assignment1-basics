import os
from collections import Counter

from cs336_basics.pretokenization import pretokenize, init_word_tokens, pretokenize_parallel


class BPE_Trainer:
    def __init__(
        self, input_path: str | os.PathLike[str], vocab_size: int, special_tokens: list[str], num_processes: int = 4
    ) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size  # including initial bytes, special tokens and bpe genetated tokens
        self.num_processes = num_processes

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

    def train(self) -> None:
        """
        Train BPE model
        """
        # w_freq = pretokenize(self.input_path, self.special_tokens, self.num_processes)
        w_freq = pretokenize_parallel(self.input_path, self.special_tokens, self.num_processes)
        w_tokens = init_word_tokens(w_freq)
        bp_words: dict[tuple[bytes, bytes], set[str]] = {}
        bp_counter: Counter[tuple[bytes, bytes]] = Counter()
        for word in w_freq:
            tokens = w_tokens[word]
            for byte_pair in zip(tokens[:-1], tokens[1:]):
                bp_counter[byte_pair] += w_freq[word]
                bp_words.setdefault(byte_pair, set()).add(word)

        # for loop to merge bytes
        for _ in range(self.vocab_size - len(self.vocab)):
            bp_merge = max(bp_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
            bp_new = bp_merge[0] + bp_merge[1]
            self.merges.append(bp_merge)
            self.vocab[len(self.vocab)] = bp_new

            for word in bp_words[bp_merge]:
                tks = w_tokens[word]
                lp = 0  # left pointer
                while lp < len(tks) - 1:
                    rp = lp + 1  # right pointer, always exist
                    if tks[lp] == bp_merge[0] and tks[rp] == bp_merge[1]:
                        if lp != 0:  # check left is valid
                            bp_counter[(tks[lp - 1], bp_merge[0])] -= w_freq[word]
                            bp_counter[(tks[lp - 1], bp_new)] += w_freq[word]
                            bp_words.setdefault((tks[lp - 1], bp_new), set()).add(word)
                        if rp != len(tks) - 1:  # check right is valid
                            bp_counter[(bp_merge[1], tks[rp + 1])] -= w_freq[word]
                            bp_counter[(bp_new, tks[rp + 1])] += w_freq[word]
                            bp_words.setdefault((bp_new, tks[rp + 1]), set()).add(word)
                        tks[lp : rp + 1] = [tks[lp] + tks[rp]]
                        bp_counter[bp_merge] -= w_freq[word]
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
