from data.dataset import END_COMBINED_DIFF_MARKER, END_FIRST_DIFF_MARKER
from dataclasses import dataclass

if END_COMBINED_DIFF_MARKER != END_FIRST_DIFF_MARKER:
    raise ValueError("END_COMBINED_DIFF_MARKER != END_FIRST_DIFF_MARKER")
DIFF_SEPARATOR = END_COMBINED_DIFF_MARKER


class ParseError(Exception):
    pass

@dataclass
class Hunk:
    left_start_line_number: int
    left_num_lines: int
    right_start_line_number: int
    right_num_lines: int
    lines: list[str]


@dataclass
class FileDiff:
    left_filename: str
    right_filename: str
    # left_short_sha: str
    # right_short_sha: str
    hunks: list[Hunk]


def _get_empty_hunk_from_start_line(start_line) -> Hunk:
    # Example start_line: "@@ -16,7 +16,7 @@ def blah():"
    between_at_signs = start_line.removeprefix("@@ -").split("@@")[0].strip()
    comma_separated_pairs = between_at_signs.split(" +")
    values = [int(value) for pair in comma_separated_pairs for value in pair.split(',')]
    return Hunk(left_start_line_number=values[0],
                left_num_lines=values[1],
                right_start_line_number=values[2],
                right_num_lines=values[3],
                lines=[])

def parse_file_diff(lines: list[str]) -> tuple[FileDiff, int]:
    if not lines[0].startswith("diff "):
        raise ParseError("Missing 'diff ...' on first line")
    if not lines[1].startswith("index "):
        raise ParseError("Missing 'index ...' on second line")

    if lines[2].startswith("--- "):
        left_filename = lines[2].removeprefix("--- ")
    else:
        raise ParseError("Missing '--- ...' on third line")

    if lines[3].startswith("+++ "):
        right_filename = lines[3].removeprefix("+++ ")
    else:
        raise ParseError("Missing '+++ ...' on fourth line")

    hunks = []
    current_hunk = None
    index_of_last_line_consumed = 0
    for index_of_last_line_consumed, line in enumerate(lines[4:]):
        if line.startswith(("diff ", "index ")):
            break
        if line.startswith("@@"):
            if current_hunk is not None:
                hunks.append(current_hunk)
            current_hunk = _get_empty_hunk_from_start_line(line)
        else:
            current_hunk.lines.append(line)
    if current_hunk is not None:
        hunks.append(current_hunk)

    file_diff = FileDiff(left_filename=left_filename,
                         right_filename=right_filename,
                         hunks=hunks)
    return file_diff, 4 + index_of_last_line_consumed




@dataclass
class DiffMetrics:
    num_diff_separators: int

def get_diff_metrics(combined_diff: str, generated_diff: str) -> DiffMetrics:
    num_diff_separators = generated_diff.count(DIFF_SEPARATOR)

    return DiffMetrics(num_diff_separators=num_diff_separators)


def diff_metrics_to_reward(metrics: DiffMetrics) -> float:
    if metrics.num_diff_separators == 1:
        return 1.0
    else:
        return -1.0