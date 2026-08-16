import os
from typing import BinaryIO
from collections import Counter
import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def compute_freq(text: str) -> Counter[tuple[bytes, ...]]:
    freq: Counter[tuple[bytes, ...]] = Counter()
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(pattern, text):
        bytes_tuple = tuple(map(lambda k: bytes([k]), list(match.group().encode('UTF-8'))))
        freq.update([bytes_tuple])
    return freq

def count_adjacent_pairs(freq: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for key, value in freq.items():
        for first, second in zip(key[:-1], key[1:]):
            counts[(first, second)] += value
    return counts

def merge(freq: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes]) -> Counter[tuple[bytes, ...]]:
    res: Counter[tuple[bytes, ...]] = Counter()
    for key, value in freq.items():
        i = 0
        merged: tuple[bytes, ...] = tuple()
        while i < len(key) - 1:
            if key[i] == pair[0] and key[i+1] == pair[1]:
                merged += (pair[0] + pair[1],)
                i += 2
            else:
                merged += (key[i],)
                i += 1
        assert i == len(key) - 1 or i == len(key)
        if i == len(key) - 1:
            merged += (key[i], )
        res[merged] = value
    return res

def train_bpe(input_file: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert len(special_tokens) >= 1
    # avoid such case: |<end>|, |<endoftext>|, then splitting first on |<end>| will chunk |<endoftext>| into two parts
    st_sorted = sorted(special_tokens, key=len, reverse=True)
    st_example = st_sorted[0].encode('UTF-8')
    with open(input_file, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, st_example)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        freq: Counter[tuple[bytes, ...]] = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pattern = '|'.join(re.escape(st) for st in st_sorted)
            splitted_chunk = re.split(pattern, chunk)
            for t in splitted_chunk:
                freq += compute_freq(t)

    vocab: dict[int, bytes] = {i: st_sorted[i].encode('UTF-8') for i in range(len(st_sorted))}
    for i in range(256):
        vocab[len(st_sorted) + i] = bytes([i])
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        counts: Counter[tuple[bytes, bytes]] = count_adjacent_pairs(freq)
        if len(counts) == 0:
            break
        pair = max(counts, key=lambda k: (counts[k], k))
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        freq = merge(freq, pair)
    return vocab, merges