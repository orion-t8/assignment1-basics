from cs336_basics.pretokenization_example import *
import cProfile
import pstats

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]

    input_file = "./tests/fixtures/tinystories_sample_5M.txt"
    num_processes = 4
    vocab_size = 500

    with cProfile.Profile() as profile_slowmerge:
        vocab_parallel, merges_parallel = train_bpe_parallel(input_file, vocab_size, special_tokens, num_processes)
    results = pstats.Stats(profile_slowmerge)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("./tests/slowmerge.prof")

    with cProfile.Profile() as profile_fastmerge:
        vocab_fastmerge, merges_fastmerge = train_bpe_parallel_fast_merge(input_file, vocab_size, special_tokens, num_processes)
    results = pstats.Stats(profile_fastmerge)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats("./tests/fastmerge.prof")
