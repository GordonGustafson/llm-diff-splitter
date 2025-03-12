from dataset import write_parquet_from_tar_directory, read_dataframe_from_tar_directory

import pyarrow
import pyarrow.parquet

from pathlib import Path
import pathlib

SMALL_COMBINED_DIFF_THRESHOLD = 1000

directory_of_script = pathlib.Path(__file__).parent.resolve()

directory_of_tar_files = directory_of_script / "patches"
all_diffs_output_path = directory_of_script / "all_diffs.parquet"
small_diffs_output_path = directory_of_script / f"combined-diffs-less-than-{SMALL_COMBINED_DIFF_THRESHOLD}-chars.parquet"

all_diffs = read_dataframe_from_tar_directory(directory_of_tar_files)
all_diffs_pyarrow_table = pyarrow.Table.from_pandas(df=all_diffs)
pyarrow.parquet.write_table(all_diffs_pyarrow_table, all_diffs_output_path)

small_diffs = all_diffs[all_diffs["combined_diff"].map(len) < SMALL_COMBINED_DIFF_THRESHOLD]
small_diffs_pyarrow_table = pyarrow.Table.from_pandas(df=small_diffs)
pyarrow.parquet.write_table(small_diffs_pyarrow_table, small_diffs_output_path)
