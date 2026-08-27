from cs336_basics.pretokenization_example import *
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
    counts = count_adjacent_pairs(freq)
    assert counts == Counter()

    freq = compute_freq(" ")
    counts = count_adjacent_pairs(freq)
    assert counts == Counter()

    freq = compute_freq("hhhhh")
    counts = count_adjacent_pairs(freq)
    assert counts == {(b'h', b'h'): 4}

    freq = compute_freq("hello")
    counts = count_adjacent_pairs(freq)
    reference = {(b'h', b'e'): 1,
                 (b'e', b'l'): 1,
                 (b'l', b'l'): 1,
                 (b'l', b'o'): 1}
    assert counts == reference

    freq = compute_freq("abab")
    counts = count_adjacent_pairs(freq)
    assert counts == {(b'a', b'b'): 2, (b'b', b'a'): 1}

def test_count_adjacent_pairs_idx_version():
    freq = compute_freq("hhhhh")
    pretoken_tokens, pretoken_freq = generate_idx_version(freq)
    assert pretoken_tokens == {0: (b'h', b'h', b'h', b'h', b'h')}
    assert pretoken_freq == {0: 1}
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    assert counts == {(b'h', b'h'): 4}
    assert pair2ids == {(b'h', b'h'): {0}}

    freq = compute_freq("hello")
    pretoken_tokens, pretoken_freq = generate_idx_version(freq)
    assert pretoken_tokens == {0: (b'h', b'e', b'l', b'l', b'o')}
    assert pretoken_freq == {0: 1}
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    reference = {(b'h', b'e'): 1,
                 (b'e', b'l'): 1,
                 (b'l', b'l'): 1,
                 (b'l', b'o'): 1}
    assert counts == reference
    assert pair2ids == {(b'h', b'e'): {0},
                          (b'e', b'l'): {0},
                          (b'l', b'l'): {0},
                          (b'l', b'o'): {0}}

    freq = compute_freq("abab")
    pretoken_tokens, pretoken_freq = generate_idx_version(freq)
    assert pretoken_tokens == {0: (b'a', b'b', b'a', b'b')}
    assert pretoken_freq == {0: 1}
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    assert counts == {(b'a', b'b'): 2, (b'b', b'a'): 1}
    assert pair2ids == {(b'a', b'b'): {0}, (b'b', b'a'): {0}}

    freq = compute_freq("ab hh")
    pretoken_tokens, pretoken_freq = generate_idx_version(freq)
    assert pretoken_tokens == {0: (b'a', b'b'), 1: (b' ', b'h', b'h')}
    assert pretoken_freq == {0: 1, 1: 1}
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    assert counts == {(b'a', b'b'): 1, (b' ', b'h'): 1, (b'h', b'h'): 1}
    assert pair2ids == {(b'a', b'b'): {0}, (b' ', b'h'): {1}, (b'h', b'h'): {1}}

    freq = compute_freq("ab abhh hh hh")
    pretoken_tokens, pretoken_freq = generate_idx_version(freq)
    assert pretoken_tokens == {0: (b'a', b'b'), 1: (b' ', b'a', b'b', b'h', b'h'), 2: (b' ', b'h', b'h')}
    assert pretoken_freq == {0: 1, 1: 1, 2: 2}
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    assert counts == {(b'a', b'b'): 2, (b' ', b'a'): 1, (b'b', b'h'): 1, (b' ', b'h'): 2, (b'h', b'h'): 3}
    assert pair2ids == {(b'a', b'b'): {0, 1}, (b' ', b'a'): {1}, (b'b', b'h'): {1}, (b' ', b'h'): {2}, (b'h', b'h'): {1, 2}}

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

def test_update_accounting():
    freq = compute_freq("ab abhh hh hh")
    pretoken_tokens, pretoken_freq = generate_idx_version(freq)
    assert pretoken_tokens == {0: (b'a', b'b'), 1: (b' ', b'a', b'b', b'h', b'h'), 2: (b' ', b'h', b'h')}
    assert pretoken_freq == {0: 1, 1: 1, 2: 2}
    counts, pair2ids = count_adjacent_pairs_idx_version(pretoken_tokens, pretoken_freq)
    assert counts == {(b'a', b'b'): 2, (b' ', b'a'): 1, (b'b', b'h'): 1, (b' ', b'h'): 2, (b'h', b'h'): 3}
    assert pair2ids == {(b'a', b'b'): {0, 1}, (b' ', b'a'): {1}, (b'b', b'h'): {1}, (b' ', b'h'): {2}, (b'h', b'h'): {1, 2}}
    pair = max(counts, key=lambda k: (counts[k], k))
    assert pair == (b'h', b'h')

    update_accounting(pretoken_freq, pair, pretoken_tokens, counts, pair2ids)
    assert pretoken_tokens == {0: (b'a', b'b'),
                               1: (b' ', b'a', b'b', b'hh'),
                               2: (b' ', b'hh')}
    assert counts == {(b'a', b'b'): 2, (b' ', b'a'): 1, (b'b', b'hh'): 1, (b' ', b'hh'): 2}
    assert pair2ids == {(b'a', b'b'): {0, 1}, (b' ', b'a'): {1}, (b'b', b'hh'): {1}, (b' ', b'hh'): {2}}

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
    vocab_size = 500
    '''
    start = time.perf_counter()
    vocab_serial, merges_serial = train_bpe(input_file, vocab_size, special_tokens, num_processes)
    end = time.perf_counter()
    print(f"串行耗时: {(end - start) * 1000:.3f} ms")
    '''

    start = time.perf_counter()
    vocab_parallel, merges_parallel = train_bpe_parallel(input_file, vocab_size, special_tokens, num_processes)
    end = time.perf_counter()
    print(f"并行耗时: {(end - start) * 1000:.3f} ms")
    #assert vocab_serial == vocab_parallel and merges_serial == merges_parallel

    start = time.perf_counter()
    vocab_fastmerge, merges_fastmerge = train_bpe_parallel_fast_merge(input_file, vocab_size, special_tokens, num_processes)
    end = time.perf_counter()
    print(f"并行+merge优化耗时: {(end - start) * 1000:.3f} ms")
    assert vocab_parallel == vocab_fastmerge and merges_parallel == merges_fastmerge