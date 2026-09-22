from cs336_basics.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import os

def tqdm_line_wrapper(file_obj, pbar):
    for line in file_obj:
        line_bytes_len = len(line.encode("utf-8"))
        yield line
        pbar.update(line_bytes_len)

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_file("output/tinystories/vocab_20260921_204713.pkl",
                                    "output/tinystories/merges_20260921_204713.pkl",
                                    special_tokens)
    train_path = "data/TinyStoriesV2-GPT4-train.txt"
    total_bytes = os.path.getsize(train_path)
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Processing training set") as pbar:
        with open(train_path, "r", encoding="utf-8", newline="") as f:
            wrapped_iterator = tqdm_line_wrapper(f, pbar)
            token_ids_iter = tokenizer.encode_iterable(wrapped_iterator)
            token_array = np.fromiter(token_ids_iter, dtype=np.uint16)
            np.save("data/TinyStoriesV2-GPT4-train-tokenized.npy", token_array)

    valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    total_bytes = os.path.getsize(valid_path)
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Processing validation set") as pbar:
        with open("data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8", newline="") as f:
            wrapped_iterator = tqdm_line_wrapper(f, pbar)
            token_ids_iter = tokenizer.encode_iterable(wrapped_iterator)
            token_array = np.fromiter(token_ids_iter, dtype=np.uint16)
            np.save("data/TinyStoriesV2-GPT4-valid-tokenized.npy", token_array)