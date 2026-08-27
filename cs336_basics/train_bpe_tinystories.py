from cs336_basics.pretokenization_example import train_bpe_parallel_fast_merge
import cProfile
import pstats
from datetime import datetime
import pickle


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]

    input_file = "./data/TinyStoriesV2-GPT4-train.txt"
    num_processes = 10
    vocab_size = 10000

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe_parallel_fast_merge(input_file, vocab_size, special_tokens, num_processes)

    with open(f"./output/vocab_{current_time}.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"./output/merges_{current_time}.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    #results.print_stats()
    results.dump_stats(f"./profile/tinystories_{current_time}.prof")
