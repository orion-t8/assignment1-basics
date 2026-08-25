from cs336_basics.pretokenization_example import compute_freq, count_adjacent_pairs, merge, train_bpe, train_bpe_parallel
from collections import Counter
import time

def test_compute_freq():
    freq = compute_freq("")
    assert freq == Counter()

    freq = compute_freq(" ")
    assert freq == {(b' ',): 1}

    freq = compute_freq("hello")
    reference = {(b'h', b'e', b'l', b'l', b'o'): 1}
    assert freq == reference

    freq = compute_freq("hello hello")
    reference = {(b'h', b'e', b'l', b'l', b'o'): 1,
                 (b' ', b'h', b'e', b'l', b'l', b'o'): 1}
    assert freq == reference

    freq = compute_freq("hello hello hello")
    reference = {(b'h', b'e', b'l', b'l', b'o'): 1,
                 (b' ', b'h', b'e', b'l', b'l', b'o'): 2}
    assert freq == reference

    freq = compute_freq("hello hello world")
    reference = {(b'h', b'e', b'l', b'l', b'o'): 1,
                 (b' ', b'h', b'e', b'l', b'l', b'o'): 1,
                 (b' ', b'w', b'o', b'r', b'l', b'd'): 1}
    assert freq == reference

    freq = compute_freq("hello \n hello world")
    reference = {(b'h', b'e', b'l', b'l', b'o'): 1,
                 (b' ', b'\n'): 1,
                 (b' ', b'h', b'e', b'l', b'l', b'o'): 1,
                 (b' ', b'w', b'o', b'r', b'l', b'd'): 1}
    assert freq == reference

def test_count_adjacent_pairs():
    freq = compute_freq("")
    counts, pair2bytes = count_adjacent_pairs(freq)
    assert counts == Counter()
    assert pair2bytes == {}

    freq = compute_freq(" ")
    counts, pair2bytes = count_adjacent_pairs(freq)
    assert counts == Counter()
    assert pair2bytes == {}

    freq = compute_freq("hhhhh")
    counts, pair2bytes = count_adjacent_pairs(freq)
    assert counts == {(b'h', b'h'): 4}
    assert pair2bytes == {(b'h', b'h'): {(b'h', b'h', b'h', b'h', b'h')}}

    freq = compute_freq("hello")
    counts, pair2bytes = count_adjacent_pairs(freq)
    reference = {(b'h', b'e'): 1,
                 (b'e', b'l'): 1,
                 (b'l', b'l'): 1,
                 (b'l', b'o'): 1}
    assert counts == reference
    assert pair2bytes == {(b'h', b'e'): {(b'h', b'e', b'l', b'l', b'o')},
                          (b'e', b'l'): {(b'h', b'e', b'l', b'l', b'o')},
                          (b'l', b'l'): {(b'h', b'e', b'l', b'l', b'o')},
                          (b'l', b'o'): {(b'h', b'e', b'l', b'l', b'o')}}

    freq = compute_freq("abab")
    counts, pair2bytes = count_adjacent_pairs(freq)
    assert counts == {(b'a', b'b'): 2, (b'b', b'a'): 1}
    assert pair2bytes == {(b'a', b'b'): {(b'a', b'b', b'a', b'b')}, (b'b', b'a'): {(b'a', b'b', b'a', b'b')}}

def test_merge():
    pair = (b'h', b'e')

    freq = Counter()
    assert merge(freq, pair) == Counter()

    freq = Counter({(b' ',): 1})
    assert merge(freq, pair) == Counter({(b' ',): 1})

    freq = Counter({(b'h', ): 2})
    assert merge(freq, pair) == Counter({(b'h', ): 2})

    freq = Counter({(b'h', b'e'): 1})
    assert merge(freq, pair) == Counter({(b'he',): 1})

    freq = Counter({(b'h', b'e'): 1,
                    (b' ', b'h', b'e', b'l', b'l', b'o'): 1})
    assert merge(freq, pair) == Counter({(b'he',): 1,
                                         (b' ', b'he', b'l', b'l', b'o'): 1})

    freq = Counter({(b'h', b'e', b'h', b'e'): 1})
    assert merge(freq, pair) == Counter({(b'he', b'he'): 1})

    pair = (b'l', b'l')
    freq = Counter({(b'l', b'l'): 1})
    assert merge(freq, pair) == Counter({(b'll',): 1})

    freq = Counter({(b'l', b'l', b'l'): 1})
    assert merge(freq, pair) == Counter({(b'll', b'l'): 1})

def test_train_bpe():
    special_tokens = ["<|endoftext|>"]
    input_file = "./data/pretokenization_example.txt"
    vocab, merges = train_bpe(input_file, 258, special_tokens)
    assert merges == [(b's', b't')]
    assert vocab[257] == b'st'

    vocab, merges = train_bpe(input_file, 259, special_tokens)
    assert merges == [(b's', b't'), (b'e', b'st')]
    assert vocab[257] == b'st' and vocab[258] == b'est'

    vocab, merges = train_bpe(input_file, 263, special_tokens)
    assert merges == [(b's', b't'),
                      (b'e', b'st'),
                      (b'o', b'w'),
                      (b'l', b'ow'),
                      (b'w', b'est'),
                      (b'n', b'e')]
    assert (vocab[257] == b'st' and vocab[258] == b'est' and vocab[259] == b'ow' and vocab[260] == b'low' and
            vocab[261] == b'west' and vocab[262] == b'ne')

def test_profile_train_bpe():
    special_tokens = ["<|endoftext|>"]
    input_file = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)

def test_train_bpe_parallel():
    special_tokens = ["<|endoftext|>"]
    """
    input_file = "./data/pretokenization_example3.txt"
    start = time.perf_counter()
    vocab_serial, merges_serial = train_bpe(input_file, 300, special_tokens, 4)
    end = time.perf_counter()
    print(f"串行耗时: {(end - start) * 1000:.3f} ms")

    start = time.perf_counter()
    vocab_parallel, merges_parallel = train_bpe_parallel(input_file, 300, special_tokens, 4)
    end = time.perf_counter()
    print(f"并行耗时: {(end - start) * 1000:.3f} ms")
    assert vocab_serial == vocab_parallel and merges_serial == merges_parallel
    """

    input_file = "./tests/fixtures/tinystories_sample_5M.txt"
    num_processes = 4
    start = time.perf_counter()
    vocab_serial, merges_serial = train_bpe(input_file, 257, special_tokens, num_processes)
    end = time.perf_counter()
    print(f"串行耗时: {(end - start) * 1000:.3f} ms")

    start = time.perf_counter()
    vocab_parallel, merges_parallel = train_bpe_parallel(input_file, 257, special_tokens, num_processes)
    end = time.perf_counter()
    print(f"并行耗时: {(end - start) * 1000:.3f} ms")
    assert vocab_serial == vocab_parallel and merges_serial == merges_parallel