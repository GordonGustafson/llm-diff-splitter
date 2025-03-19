from data.dataset import END_COMBINED_DIFF_MARKER, END_FIRST_DIFF_MARKER
from dataclasses import dataclass

import itertools

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


@dataclass
class ModelOutput:
    first_commit_diffs: list[FileDiff]
    second_commit_diffs: list[FileDiff]


def get_left_file_lines(hunk: Hunk) -> str:
    return "\n".join(line[1:] for line in hunk.lines
                     if line.startswith((" ", "-")))

def get_right_file_lines(hunk: Hunk) -> str:
    return "\n".join(line[1:] for line in hunk.lines
                     if line.startswith((" ", "+")))

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

def parse_file_diff_from_lines(lines: list[str]) -> tuple[FileDiff, int]:
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
    index_of_last_line_consumed = 4
    for index_of_last_line_consumed, line in itertools.islice(enumerate(lines), 4, None):
        if line.startswith(("diff ", "index ")):
            index_of_last_line_consumed -= 1
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
    num_lines_consumed = index_of_last_line_consumed + 1
    return file_diff, num_lines_consumed

def parse_multiple_file_diffs(text: str) -> list[FileDiff]:
    if text.strip() == "":
        return []
    lines = text.split("\n")
    index_of_next_line_to_parse = 0
    result = []
    while index_of_next_line_to_parse < len(lines):
        file_diff, num_lines_parsed = parse_file_diff_from_lines(lines[index_of_next_line_to_parse:])
        index_of_next_line_to_parse += num_lines_parsed
        result.append(file_diff)
    return result

def parse_model_output(output: str) -> ModelOutput:
    output_split = output.split(DIFF_SEPARATOR)
    return ModelOutput(first_commit_diffs=parse_multiple_file_diffs(output_split[0]),
                       second_commit_diffs=parse_multiple_file_diffs(output_split[1]))


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