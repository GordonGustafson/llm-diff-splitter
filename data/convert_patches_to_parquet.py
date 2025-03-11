from dataset import write_parquet_from_tar_directory

from pathlib import Path


directory_of_tar_files = Path("/home/ggustafson/stuff/llm-diff-splitter/data/patches/")
output_parquet_filename = Path("output.parquet")

write_parquet_from_tar_directory(directory_of_tar_files, Path(output_parquet_filename))