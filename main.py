from data.dataset import  DiffPairDataset, read_dataframe_from_tar_directory, write_parquet_from_tar_directory, load_huggingface_dataset

from datasets import load_dataset

from pathlib import Path


directory_of_tar_files = Path("/home/ggustafson/stuff/llm-diff-splitter/data/patches/")
output_parquet_filename = Path("output.parquet")

# dataset = DiffPairDataset(directory_of_tar_files)
# print(dataset[37].first_diff)


# df = tar_directory_to_dataframe(directory_of_tar_files)
# print(df)

# tar_directory_to_parquet(directory_of_tar_files, Path(output_parquet_filename))

dataset = load_huggingface_dataset(output_parquet_filename)
print(dataset)

iterator = dataset["train"].iter(batch_size=1)
row = next(iterator)
print(row["text"])


