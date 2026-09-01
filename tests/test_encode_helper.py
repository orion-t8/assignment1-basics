from cs336_basics.tokenizer import merge_chunk

def test_merge_chunk():
    text = "hello"
    merges = [(b'h', b'e'), (b'l', b'l')]
    res = merge_chunk(merges, text)
    assert res == (b'he', b'll', b'o')

    merges = []
    res = merge_chunk(merges, text)
    assert res ==(b'h', b'e', b'l', b'l', b'o')

    text = "aaa"
    merges = [(b'a', b'a')]
    res = merge_chunk(merges, text)
    assert res ==(b'aa', b'a')

    text = "aaaa"
    merges = [(b'a', b'a')]
    res = merge_chunk(merges, text)
    assert res ==(b'aa', b'aa')