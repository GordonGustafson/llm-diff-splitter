import datasets
import pandas as pd
import pyarrow
import pyarrow.parquet
from torch.utils.data import Dataset


from dataclasses import dataclass, asdict
import io
from pathlib import Path
import tarfile


_FIRST_DIFF_FILENAME_IN_TAR = "first-diff.patch"
_SECOND_DIFF_FILENAME_IN_TAR = "second-diff.patch"
_COMBINED_DIFF_FILENAME_IN_TAR = "combined-diff.patch"

# 80 hash signs followed by a newline is a single token in Llama 3.
END_COMBINED_DIFF_MARKER = "#" * 80 + "\n"
END_FIRST_DIFF_MARKER = END_COMBINED_DIFF_MARKER

@dataclass
class DiffPair:
    first_diff: str
    second_diff: str
    combined_diff: str


def _read_str_from_tarfile(archive: tarfile.TarFile, filename: str) -> str:
    with archive.extractfile(filename) as binary, io.TextIOWrapper(binary, encoding='latin-1') as text:
        return text.read()


def _read_diff_pair_from_tar_path(tar_path: Path) -> DiffPair:
    with tarfile.open(tar_path) as archive:
        return DiffPair(
            first_diff=_read_str_from_tarfile(archive, _FIRST_DIFF_FILENAME_IN_TAR),
            second_diff=_read_str_from_tarfile(archive, _SECOND_DIFF_FILENAME_IN_TAR),
            combined_diff=_read_str_from_tarfile(archive, _COMBINED_DIFF_FILENAME_IN_TAR),
        )


def read_dataframe_from_tar_directory(directory_of_tar_files: Path) -> pd.DataFrame:
    dicts = (asdict(_read_diff_pair_from_tar_path(tar_path))
             for tar_path in directory_of_tar_files.glob("*"))
    return pd.DataFrame(dicts)


def write_parquet_from_tar_directory(directory_of_tar_files: Path, parquet_output_path: Path) -> None:
    dataframe = read_dataframe_from_tar_directory(directory_of_tar_files)
    table = pyarrow.Table.from_pandas(df=dataframe)
    pyarrow.parquet.write_table(table, parquet_output_path)


def diff_pair_from_dict(dic: dict[str, str]) -> DiffPair:
    return DiffPair(
        first_diff=dic["first_diff"],
        second_diff=dic["second_diff"],
        combined_diff=dic["combined_diff"],
    )

def get_separate_prompt_and_completion(dic: dict[str, str]) -> dict[str, str]:
    prompt = f"{dic['combined_diff']}{END_COMBINED_DIFF_MARKER}"
    completion = f"{dic['first_diff']}{END_FIRST_DIFF_MARKER}{dic['second_diff']}"
    return {"prompt": prompt, "completion": completion}

def get_combined_prompt_and_completion(dic: dict[str, str]) -> dict[str, str]:
    text = f"{dic['combined_diff']}{END_COMBINED_DIFF_MARKER}{dic['first_diff']}{END_FIRST_DIFF_MARKER}{dic['second_diff']}"
    return {"prompt_and_completion": text}

def load_huggingface_dataset(parquet_path: Path) -> datasets.DatasetDict:
    return datasets.load_dataset("parquet", data_files=str(parquet_path))

class DiffPairDataset(Dataset):
    def __init__(self, directory_of_tar_files: Path) -> None:
        self.patch_directory = directory_of_tar_files
        self.file_paths = list(directory_of_tar_files.glob("*"))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, i: int) -> DiffPair:
        file_path = self.file_paths[i]
        return _read_diff_pair_from_tar_path(file_path)
