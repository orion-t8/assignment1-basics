from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics.pretokenization_example import train_bpe_parallel_fast_merge, merge_pair

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.bytes2ids: dict[bytes, int] = dict()
        for idx, b in vocab.items():
            self.bytes2ids[b] = idx
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.st2bytes = {st: st.encode('UTF-8') for st in self.special_tokens}
        st_pattern = '|'.join(re.escape(st) for st in self.special_tokens)
        self.st_pattern = re.compile(f'({st_pattern})')
        self.pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode_chunk(self, text: str) -> list[int]:
        text_bytes = tuple(map(lambda k: bytes([k]), list(text.encode('UTF-8'))))
        # track the ordered list of pairs, find the pair that appears first in self.merges, merge the pair and update the list
        while True:
            found = False
            adjacnet_pairs = set(zip(text_bytes[:-1], text_bytes[1:]))
            for pair in self.merges:
                if pair in adjacnet_pairs:
                    found = True
                    text_bytes = merge_pair(pair, text_bytes)
                    break
            if not found:
                break
        return [self.bytes2ids[b] for b in text_bytes]

    def encode(self, text: str) -> list[int]:
        # split by special tokens
        res = []
        parts = self.st_pattern.split(text) # even index is text, odd inedex is delimitter
        for i in range(len(parts)):
            if i % 2 == 0:
                res += self.encode_chunk(parts[i])
            else:
                res += [self.bytes2ids[self.st2bytes[parts[i]]]]
        raise res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
