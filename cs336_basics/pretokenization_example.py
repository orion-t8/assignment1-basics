import os
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re
from concurrent.futures import ProcessPoolExecutor
import time
import copy

PROFILE_TIMING = False

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

def count_adjacent_pairs(freq: Counter[tuple[bytes, ...]]) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]]]:
    """
    Count occurrences of adjacent pairs.
    Return count and the indexing of adjacent pair to the set of byte strings that contribute to its count
    """
    counts: Counter[tuple[bytes, bytes]] = Counter()
    pair2bytes: defaultdict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]] = defaultdict(Counter)
    for key, value in freq.items():
        for pair in zip(key[:-1], key[1:]):
            counts[pair] += value
            pair2bytes[pair][key] += 1
    return counts, pair2bytes

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

def process_chunk(input_file: str, start: int, end: int, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    st_sorted = sorted(special_tokens, key=len, reverse=True)
    freq: Counter[tuple[bytes, ...]] = Counter()
    with open(input_file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pattern = '|'.join(re.escape(st) for st in st_sorted)
        splitted_chunk = re.split(pattern, chunk)
        for t in splitted_chunk:
            freq += compute_freq(t)
    return freq

def update_accounting(freq: Counter[tuple[bytes, ...]],
                      pair: tuple[bytes, bytes],
                      counts: Counter[tuple[bytes, bytes]],
                      pair2bytes: defaultdict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]]) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]]]:
    new_counts = counts.copy()
    new_pair2bytes = copy.deepcopy(pair2bytes)
    for bytes_tuple in pair2bytes[pair]:
        for i in range(len(bytes_tuple) - 1):
            if bytes_tuple[i] != pair[0] or bytes_tuple[i+1] != pair[1]:
                continue
            count = freq[bytes_tuple]
            # first, update counts and mapping of the pair itself
            new_counts[pair] -= count
            new_counts = +new_counts
            new_pair2bytes[pair][bytes_tuple] -= 1
            new_pair2bytes[pair] = +new_pair2bytes[pair]
            if i > 0:
                # Consider *, A, B. Update pair (*, AB)
                new_pair = (bytes_tuple[i-1], pair[0] + pair[1])
                new_counts[new_pair] += count
                new_pair2bytes[new_pair][bytes_tuple] += 1
                # update pair (*, A)
                affected_pair = (bytes_tuple[i-1], pair[0])
                new_counts[affected_pair] -= count
                new_counts = +new_counts
                new_pair2bytes[affected_pair][bytes_tuple] -= 1
                new_pair2bytes[affected_pair] = +new_pair2bytes[affected_pair]
            if i < len(bytes_tuple) - 2:
                # Consider A, B, *. Update pair (AB, *)
                new_pair = (pair[0] + pair[1], bytes_tuple[i-1])
                new_counts[new_pair] += count
                new_pair2bytes[new_pair][bytes_tuple] += 1
                # update pair (B, *)
                affected_pair = (pair[1], bytes_tuple[i+2])
                new_counts[affected_pair] -= count
                new_counts = +new_counts
                new_pair2bytes[affected_pair][bytes_tuple] -= 1
                new_pair2bytes[affected_pair] = +new_pair2bytes[affected_pair]
    assert new_counts[pair] == 0
    assert len(new_pair2bytes[pair]) == 0
    return new_counts, new_pair2bytes


def train_bpe_parallel_fast_merge(input_file: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert len(special_tokens) >= 1
    # avoid such case: |<end>|, |<endoftext>|, then splitting first on |<end>| will chunk |<endoftext>| into two parts
    st_sorted = sorted(special_tokens, key=len, reverse=True)
    
    start = 0.0
    end = 0.0
    if PROFILE_TIMING:
        start = time.perf_counter()
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, st_sorted[0].encode('UTF-8'))
    if PROFILE_TIMING:
        end = time.perf_counter()
        print(f"boundry: {(end - start) * 1000:.3f} ms")

    if PROFILE_TIMING:
        start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_chunk, input_file, start, end, st_sorted)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        results = [fut.result() for fut in futures]
    if PROFILE_TIMING:
        end = time.perf_counter()
        print(f"process pool: {(end - start) * 1000:.3f} ms")

    freq: Counter[tuple[bytes, ...]] = Counter()
    for r in results:
        freq += r

    vocab: dict[int, bytes] = {i: st_sorted[i].encode('UTF-8') for i in range(len(st_sorted))}
    for i in range(256):
        vocab[len(st_sorted) + i] = bytes([i])
    merges: list[tuple[bytes, bytes]] = []

    if PROFILE_TIMING:
        start = time.perf_counter()

    # No need to count and merge every time, but instead track the following two indices:
    # adjacent pair counts and which pretokens contribute to the count
    counts, pair2bytes = count_adjacent_pairs(freq)
    while len(vocab) < vocab_size:
        # if all pretokens are merged into one token, then break
        if len(counts) == 0:
            break
        pair = max(counts, key=lambda k: (counts[k], k))
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        counts, pair2bytes = update_accounting(freq, pair, counts, pair2bytes)
    if PROFILE_TIMING:
        end = time.perf_counter()
        print(f"merge loop: {(end - start) * 1000:.3f} ms")
    return vocab, merges

def train_bpe_parallel(input_file: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert len(special_tokens) >= 1
    # avoid such case: |<end>|, |<endoftext>|, then splitting first on |<end>| will chunk |<endoftext>| into two parts
    st_sorted = sorted(special_tokens, key=len, reverse=True)
    
    start = 0.0
    end = 0.0
    if PROFILE_TIMING:
        start = time.perf_counter()
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, st_sorted[0].encode('UTF-8'))
    if PROFILE_TIMING:
        end = time.perf_counter()
        print(f"boundry: {(end - start) * 1000:.3f} ms")

    if PROFILE_TIMING:
        start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_chunk, input_file, start, end, st_sorted)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        results = [fut.result() for fut in futures]
    if PROFILE_TIMING:
        end = time.perf_counter()
        print(f"process pool: {(end - start) * 1000:.3f} ms")

    freq: Counter[tuple[bytes, ...]] = Counter()
    for r in results:
        freq += r

    vocab: dict[int, bytes] = {i: st_sorted[i].encode('UTF-8') for i in range(len(st_sorted))}
    for i in range(256):
        vocab[len(st_sorted) + i] = bytes([i])
    merges: list[tuple[bytes, bytes]] = []

    if PROFILE_TIMING:
        start = time.perf_counter()
    while len(vocab) < vocab_size:
        counts, _ = count_adjacent_pairs(freq)
        if len(counts) == 0:
            break
        pair = max(counts, key=lambda k: (counts[k], k))
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        freq = merge(freq, pair)
    if PROFILE_TIMING:
        end = time.perf_counter()
        print(f"merge loop: {(end - start) * 1000:.3f} ms")
    return vocab, merges

def train_bpe(input_file: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert len(special_tokens) >= 1
    # avoid such case: |<end>|, |<endoftext>|, then splitting first on |<end>| will chunk |<endoftext>| into two parts
    st_sorted = sorted(special_tokens, key=len, reverse=True)
    start_time = 0.0
    end_time = 0.0
    with open(input_file, "rb") as f:
        if PROFILE_TIMING:
            start_time = time.perf_counter()
        boundaries = find_chunk_boundaries(f, num_processes, st_sorted[0].encode('UTF-8'))
        if PROFILE_TIMING:
            end_time = time.perf_counter()
            print(f"boundary: {(end_time - start_time) * 1000:.3f} ms")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        freq: Counter[tuple[bytes, ...]] = Counter()
        if PROFILE_TIMING:
            start_time = time.perf_counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pattern = '|'.join(re.escape(st) for st in st_sorted)
            splitted_chunk = re.split(pattern, chunk)
            for t in splitted_chunk:
                freq += compute_freq(t)
        if PROFILE_TIMING:
            end_time = time.perf_counter()
            print(f"pretokenization: {(end_time - start_time) * 1000:.3f} ms")

    vocab: dict[int, bytes] = {i: st_sorted[i].encode('UTF-8') for i in range(len(st_sorted))}
    for i in range(256):
        vocab[len(st_sorted) + i] = bytes([i])
    merges: list[tuple[bytes, bytes]] = []

    if PROFILE_TIMING:
        start_time = time.perf_counter()
    while len(vocab) < vocab_size:
        counts, _ = count_adjacent_pairs(freq)
        if len(counts) == 0:
            break
        pair = max(counts, key=lambda k: (counts[k], k))
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        freq = merge(freq, pair)
    if PROFILE_TIMING:
        end_time = time.perf_counter()
        print(f"merge loop: {(end_time - start_time) * 1000:.3f} ms")

    return vocab, merges