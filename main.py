from data.dataset import  DiffPairDataset

from pathlib import Path

dataset = DiffPairDataset(Path("/home/ggustafson/stuff/llm-diff-splitter/data/patches/"))

print(dataset[37].first_diff)
