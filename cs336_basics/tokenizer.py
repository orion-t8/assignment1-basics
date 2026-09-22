from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics.pretokenization_example import merge_pair

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

        self.max_cache_pretoken_size = 10000
        self.pretoken_cache: dict[str, tuple[int, ...]] = dict()

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def merge_pretoken(self, text: str) -> tuple[bytes, ...]:
        # try to match in cache, if found, re-insert to update key's position
        res = self.pretoken_cache.pop(text, tuple())
        if len(res) > 0:
            self.pretoken_cache[text] = res
            return res

        # evict the oldest key if cache size exceeds max cache size
        if len(self.pretoken_cache) == self.max_cache_pretoken_size:
            oldest_key = next(iter(self.pretoken_cache))
            self.pretoken_cache.pop(oldest_key)
        encoded = text.encode("utf-8")
        cur_bytes = tuple(encoded[i:i+1] for i in range(len(encoded)))
        invalid_rank = len(self.merge_rank)
        # track the ordered list of pairs, find the pair that appears first in self.merges, merge the pair and update the list
        while True:
            min_rank = invalid_rank
            min_rank_pair: tuple[bytes, bytes] | None = None
            for i in range(len(cur_bytes) - 1):
                pair = (cur_bytes[i], cur_bytes[i+1])
                cur_rank = self.merge_rank.get(pair, invalid_rank)
                if cur_rank < min_rank:
                    min_rank = cur_rank
                    min_rank_pair = pair
            if min_rank == invalid_rank:
                self.pretoken_cache[text] = cur_bytes
                return cur_bytes
            cur_bytes = merge_pair(min_rank_pair, cur_bytes)

    def encode(self, text: str) -> list[int]:
        res: list[int] = []
        if not self.special_tokens:
            for match in re.finditer(self.pretokenize_pattern, text):
                res += [self.bytes2ids[b] for b in self.merge_pretoken(match.group())]
            return res

        # split by special tokens
        parts = self.st_pattern.split(text) # even index is text, odd inedex is delimitter
        for i in range(len(parts)):
            if i % 2 == 0:
                for match in re.finditer(self.pretokenize_pattern, parts[i]):
                    res += [self.bytes2ids[b] for b in self.merge_pretoken(match.group())]
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