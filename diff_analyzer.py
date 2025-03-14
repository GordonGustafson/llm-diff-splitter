from data.dataset import END_COMBINED_DIFF_MARKER, END_FIRST_DIFF_MARKER
from dataclasses import dataclass

if END_COMBINED_DIFF_MARKER != END_FIRST_DIFF_MARKER:
    raise ValueError("END_COMBINED_DIFF_MARKER != END_FIRST_DIFF_MARKER")
DIFF_SEPARATOR = END_COMBINED_DIFF_MARKER


@dataclass
class DiffMetrics:
    num_diff_separators: int


def get_diff_metrics(diff: str) -> DiffMetrics:
    num_diff_separators = diff.count(DIFF_SEPARATOR)

    return DiffMetrics(num_diff_separators=num_diff_separators)


def diff_metrics_to_reward(metrics: DiffMetrics) -> float:
    if metrics.num_diff_separators == 1:
        return 1.0
    else:
        return 0.0