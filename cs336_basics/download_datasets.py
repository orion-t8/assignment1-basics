# /// script
# dependencies = [
#     "datasets",
# ]
# ///
from datasets import load_dataset, load_from_disk

# 替换为你需要的数据集名称
'''
dataset = load_dataset("stanford-cs336/owt-sample", encoding='utf-8') 
dataset.save_to_disk('./data')
print("数据集下载成功！")
'''

dataset = load_from_disk("./data")

print(dataset)
print(dataset.column_names)

with open("./data/owt-train.txt", "w", encoding="utf-8") as f:
    for example in dataset['train']:
        text = example['text']
        f.write(text)
        f.write("\n")

with open("./data/owt-valid.txt", "w", encoding="utf-8") as f:
    for example in dataset['validation']:
        text = example['text']
        f.write(text)
        f.write("\n")