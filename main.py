from data.dataset import  DiffPairDataset, tar_directory_to_dataframe, tar_directory_to_parquet

from pathlib import Path

directory_of_tar_files = Path("/home/ggustafson/stuff/llm-diff-splitter/data/patches/")

# dataset = DiffPairDataset(directory_of_tar_files)
# print(dataset[37].first_diff)


# df = tar_directory_to_dataframe(directory_of_tar_files)
# print(df)

tar_directory_to_parquet(directory_of_tar_files, Path("output.parquet"))
