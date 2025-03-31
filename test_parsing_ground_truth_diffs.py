from data.dataset import load_huggingface_dataset
from diff_analyzer import parse_multiple_file_diffs
from eval import PARQUET_DATASET_PATH

import os


def check_row(row):
    _ = parse_multiple_file_diffs(row["first_diff"])
    _ = parse_multiple_file_diffs(row["second_diff"])
    _ = parse_multiple_file_diffs(row["combined_diff"])
    return {}

def test_parsing_ground_truth_diffs():
    datasets = load_huggingface_dataset(PARQUET_DATASET_PATH)
    _ = datasets.map(check_row, num_proc=os.cpu_count(), load_from_cache_file=False)

if __name__ == "__main__":
    test_parsing_ground_truth_diffs()
