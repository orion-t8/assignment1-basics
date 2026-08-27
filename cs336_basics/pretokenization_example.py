import os
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm

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

def count_adjacent_pairs(freq: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    """
    Count occurrences of adjacent pairs.
    Return count and the indexing of adjacent pair to the set of byte strings that contribute to its count
    """
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for key, value in freq.items():
        for pair in zip(key[:-1], key[1:]):
            counts[pair] += value
    return counts

def count_adjacent_pairs_idx_version(pretoken_tokens: dict[int, tuple[bytes,...]], pretoken_freq: Counter[int]) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[int]]]:
    """
    Count occurrences of adjacent pairs.
    Return count and the indexing of adjacent pair to the set of byte strings that contribute to its count
    """
    counts: Counter[tuple[bytes, bytes]] = Counter()
    pair2ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for idx, freq in pretoken_freq.items():
        token = pretoken_tokens[idx]
        for pair in zip(token[:-1], token[1:]):
            counts[pair] += freq
            pair2ids[pair].add(idx)
    return counts, pair2ids

def merge_pair(pair: tuple[bytes, bytes], token: tuple[bytes, ...]) -> tuple[bytes, ...]:
    i = 0
    merged: tuple[bytes, ...] = tuple()
    while i < len(token) - 1:
        if token[i] == pair[0] and token[i+1] == pair[1]:
            merged += (pair[0] + pair[1],)
            i += 2
        else:
            merged += (token[i],)
            i += 1
    assert i == len(token) - 1 or i == len(token)
    if i == len(token) - 1:
        merged += (token[i], )
    return merged

def merge(freq: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes]) -> Counter[tuple[bytes, ...]]:
    res: Counter[tuple[bytes, ...]] = Counter()
    for key, value in freq.items():
        merged = merge_pair(pair, key)
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


def update_accounting(freq: Counter[int],
                      pair_to_merge: tuple[bytes, bytes],
                      tokens: dict[int, tuple[bytes, ...]],
                      counts: Counter[tuple[bytes, bytes]],
                      pair2ids: defaultdict[tuple[bytes, bytes], set[int]]):
    token_size = 0
    num_indices = len(pair2ids[pair_to_merge])
    if PROFILE_TIMING:
        print("#ids to scan: %d, percentage: %f" % (num_indices, num_indices / len(freq)))

    for idx in list(pair2ids[pair_to_merge]): # use list() to create a copy of keys
        token = tokens[idx]
        if PROFILE_TIMING:
            token_size += len(token)
        # subtract pair count and pair mapping corresponding to this token
        for pair in zip(token[:-1], token[1:]):
            counts[pair] -= freq[idx]
            if counts[pair] <= 0:
                del counts[pair]
            pair2ids[pair].discard(idx)
            if not pair2ids[pair]:
                del pair2ids[pair]

        # merge the pair in this token
        tokens[idx] = merge_pair(pair_to_merge, token)

        # add pair count and pair mappings correponding to the merged token
        for pair in zip(tokens[idx][:-1], tokens[idx][1:]):
            counts[pair] += freq[idx]
            pair2ids[pair].add(idx)
    if PROFILE_TIMING:
        print("Avg token length: %f" % (token_size / num_indices))


def generate_idx_version(freq: Counter[tuple[bytes, ...]] = Counter()) -> tuple[dict[int, tuple[bytes, ...]], Counter[int]]:
    # compute pretoken_tokens[id], pretoken_freq[id]
    pretoken_tokens: dict[int, tuple[bytes, ...]] = dict() # dict value will change due to merging operation
    pretoken_freq: Counter[int] = Counter() # invariant throughout merging
    for idx, (key, value) in enumerate(freq.items()):
        pretoken_tokens[idx] = key
        pretoken_freq[idx] = value
    return pretoken_tokens, pretoken_freq

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

    pretoken_tokens, pretoken_freq = generate_idx_version(freq)

    vocab: dict[int, bytes] = {i: st_sorted[i].encode('UTF-8') for i in range(len(st_sorted))}
    for i in range(256):
        vocab[len(st_sorted) + i] = bytes([i])

    merges: list[tuple[bytes, bytes]] = []

    if PROFILE_TIMING:
        start = time.perf_counter()

    # No need to count and merge every time, but instead track the following two indices:
    # adjacent pair counts and which pretokens contribute to the count
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    num_merges_to_learn = vocab_size - len(vocab)
    with tqdm(total=num_merges_to_learn, desc="BPE Merges") as pbar:
        while len(vocab) < vocab_size:
            # if all pretokens are merged into one token, then break
            if len(counts) == 0:
                break
            pair = max(counts, key=lambda k: (counts[k], k))
            merges.append(pair)
            vocab[len(vocab)] = pair[0] + pair[1]
            update_accounting(pretoken_freq, pair, pretoken_tokens, counts, pair2ids)
            pbar.update(1)
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
        counts = count_adjacent_pairs(freq)
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
        counts = count_adjacent_pairs(freq)
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