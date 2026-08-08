from cs336_basics.pretokenization_example import compute_freq, count_adjacent_pairs, merge
from collections import Counter

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