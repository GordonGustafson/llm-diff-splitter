from dataset import write_parquet_from_tar_directory

from pathlib import Path
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()

directory_of_tar_files = directory_of_script / "patches"
output_parquet_filename = directory_of_script / "output.parquet"

write_parquet_from_tar_directory(directory_of_tar_files, Path(output_parquet_filename))