from dataclasses import dataclass

from torch.utils.data import Dataset

import tarfile
import io
from pathlib import Path


_FIRST_DIFF_FILENAME_IN_TAR = "first-diff.patch"
_SECOND_DIFF_FILENAME_IN_TAR = "second-diff.patch"
_COMBINED_DIFF_FILENAME_IN_TAR = "combined-diff.patch"


@dataclass
class DiffPair:
    first_diff: str
    second_diff: str
    combined_diff: str


def _read_str_from_tarfile(archive: tarfile.TarFile, filename: str) -> str:
    with archive.extractfile(filename) as binary, io.TextIOWrapper(binary) as text:
        return text.read()


def _read_diff_pair_from_tar_path(tar_path: Path) -> DiffPair:
    with tarfile.open(tar_path) as archive:
        return DiffPair(
            first_diff=_read_str_from_tarfile(archive, _FIRST_DIFF_FILENAME_IN_TAR),
            second_diff=_read_str_from_tarfile(archive, _SECOND_DIFF_FILENAME_IN_TAR),
            combined_diff=_read_str_from_tarfile(archive, _COMBINED_DIFF_FILENAME_IN_TAR),
        )


class DiffPairDataset(Dataset):
    def __init__(self, patch_directory: Path) -> None:
        self.patch_directory = patch_directory
        self.file_paths = list(patch_directory.glob("*"))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, i: int) -> DiffPair:
        file_path = self.file_paths[i]
        return _read_diff_pair_from_tar_path(file_path)
