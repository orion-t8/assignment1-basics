from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics.pretokenization_example import merge_pair

def merge_chunk(merge_rank: dict[tuple[bytes, bytes], int], text: str) -> tuple[bytes, ...]:
    cur_bytes = tuple(map(lambda k: bytes([k]), list(text.encode('UTF-8'))))
    invalid_rank = len(merge_rank)
    # track the ordered list of pairs, find the pair that appears first in self.merges, merge the pair and update the list
    while True:
        min_rank = invalid_rank
        min_rank_pair: tuple[bytes, bytes] | None = None
        for i in range(len(cur_bytes) - 1):
            pair = (cur_bytes[i], cur_bytes[i+1])
            cur_rank = merge_rank.get(pair, invalid_rank)
            if cur_rank < min_rank:
                min_rank = cur_rank
                min_rank_pair = pair
        if min_rank == invalid_rank:
            return cur_bytes
        cur_bytes = merge_pair(min_rank_pair, cur_bytes)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merge_rank: dict[tuple[bytes, bytes], int] = dict() # used in encoding, fast lookup
        for i in range(len(self.merges)):
            self.merge_rank[self.merges[i]] = i
        self.bytes2ids: dict[bytes, int] = dict()
        for idx, b in vocab.items():
            self.bytes2ids[b] = idx
        self.pretokenize_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.special_tokens = special_tokens
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.st2bytes = {st: st.encode('UTF-8') for st in self.special_tokens}
            st_pattern = '|'.join(re.escape(st) for st in self.special_tokens)
            self.st_pattern = re.compile(f'({st_pattern})')

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        res: list[int] = []
        if not self.special_tokens:
            for match in re.finditer(self.pretokenize_pattern, text):
                res += [self.bytes2ids[b] for b in merge_chunk(self.merge_rank, match.group())]
            return res

        # split by special tokens
        parts = self.st_pattern.split(text) # even index is text, odd inedex is delimitter
        for i in range(len(parts)):
            if i % 2 == 0:
                for match in re.finditer(self.pretokenize_pattern, parts[i]):
                    res += [self.bytes2ids[b] for b in merge_chunk(self.merge_rank, match.group())]
            else:
                res += [self.bytes2ids[self.st2bytes[parts[i]]]]
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            result: list[int] = self.encode(text)
            yield from result

    def decode(self, ids: list[int]) -> str:
        text_bytes = b''
        for idx in ids:
            text_bytes += self.vocab[idx]
        return text_bytes.decode('UTF-8', errors='replace')