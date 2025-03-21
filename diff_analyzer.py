from functools import reduce
from typing import TypeAlias, Iterator

from data.dataset import END_COMBINED_DIFF_MARKER, END_FIRST_DIFF_MARKER
from dataclasses import dataclass

import itertools
import re

if END_COMBINED_DIFF_MARKER != END_FIRST_DIFF_MARKER:
    raise ValueError("END_COMBINED_DIFF_MARKER != END_FIRST_DIFF_MARKER")
DIFF_SEPARATOR = END_COMBINED_DIFF_MARKER


FileName: TypeAlias = str

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
    left_filename: FileName
    right_filename: FileName
    # left_short_sha: str
    # right_short_sha: str
    hunks: list[Hunk]


@dataclass
class ParsedDiffPair:
    first_commit_diffs: list[FileDiff]
    second_commit_diffs: list[FileDiff]


SparseFileLines: TypeAlias = dict[int, str]
SparseCommitState: TypeAlias = dict[FileName, SparseFileLines]

@dataclass
class IOUStats:
    num_only_in_a: int
    num_only_in_b: int
    num_in_intersection: int

    def iou(self) -> float:
        union = self.num_only_in_a + self.num_only_in_b + self.num_in_intersection
        if union == 0:
            # TODO: Unclear if we should return 0 or 1 here?
            return 1.0
        return self.num_in_intersection / union


def _merge_dict_sequence(dict_iterator: Iterator[dict]) -> dict:
    return reduce(lambda a, b: a | b, dict_iterator, {})

def _get_start_line_num_lines_tuple(s: str) -> tuple[int, int]:
    if "," in s:
        split = s.split(",")
        return int(split[0]), int(split[1])
    # If there's only 1 line, the number of lines can be omitted.
    return int(s), 1

def _get_empty_hunk_from_start_line(start_line) -> Hunk:
    # Example start_line: "@@ -16,7 +16,7 @@ def blah():"
    between_at_signs = start_line.removeprefix("@@ -").split("@@")[0].strip()
    comma_separated_pairs = between_at_signs.split(" +")
    values = [value for pair in comma_separated_pairs for value in _get_start_line_num_lines_tuple(pair)]
    return Hunk(left_start_line_number=values[0],
                left_num_lines=values[1],
                right_start_line_number=values[2],
                right_num_lines=values[3],
                lines=[])

def parse_file_diff_from_lines(lines: list[str]) -> tuple[(FileDiff | None), int]:
    next_line_to_consume_index = 0

    if lines[next_line_to_consume_index].strip() == "":
        return None, 1

    if not lines[next_line_to_consume_index].startswith("diff "):
        raise ParseError(f"Missing 'diff ...' on first line, which is {lines[next_line_to_consume_index]}")
    next_line_to_consume_index += 1

    file_mode_changes = False
    if lines[next_line_to_consume_index].startswith("old mode"):
        next_line_to_consume_index += 1
        file_mode_changes = True

    if lines[next_line_to_consume_index].startswith("new mode"):
        next_line_to_consume_index += 1
        file_mode_changes = True

        if len(lines) > next_line_to_consume_index and lines[next_line_to_consume_index].strip() == "":
            next_line_to_consume_index += 1

    if len(lines) > next_line_to_consume_index and lines[next_line_to_consume_index].startswith("new file mode"):
        next_line_to_consume_index += 1
        file_mode_changes = True

    if len(lines) > next_line_to_consume_index and lines[next_line_to_consume_index].startswith("deleted file mode"):
        next_line_to_consume_index += 1
        file_mode_changes = True

    if len(lines) > next_line_to_consume_index and not lines[next_line_to_consume_index].startswith("index "):
        raise ParseError(f"Missing 'index ...' on expected line, which is {lines[next_line_to_consume_index]}")
    next_line_to_consume_index += 1

    if len(lines) > next_line_to_consume_index:
        binary_files_match = re.match("^Binary files (.*) and (.*) differ$", lines[next_line_to_consume_index])
        if binary_files_match:
            next_line_to_consume_index += 1
            groups = binary_files_match.groups()
            file_diff = FileDiff(left_filename=groups[0], right_filename=groups[1], hunks=[])
            return file_diff, next_line_to_consume_index

    if len(lines) > next_line_to_consume_index and lines[next_line_to_consume_index].startswith("--- "):
        left_filename = lines[next_line_to_consume_index].removeprefix("--- ")
    elif file_mode_changes:
        # If the file's has mode changes it may not have content changes. In this case we
        # pull the filename from the `diff ...` line. Feel free to switch to pulling the
        # filenames from the `diff ...` line all the time.
        filenames_line = lines[0].removeprefix("diff ")
        # TODO: remove other diff flags?
        filenames_line = filenames_line.removeprefix("--git ")
        # TODO: handle paths with spaces.
        left_filename, right_filename = filenames_line.split(" ")
        # If there's an empty line after this, optionally consume it too. Not sure why this happened in the ground truth data.
        if len(lines) > next_line_to_consume_index and lines[next_line_to_consume_index].strip() == "":
            next_line_to_consume_index += 1
        return FileDiff(left_filename=left_filename, right_filename=right_filename, hunks=[]), next_line_to_consume_index
    else:
        raise ParseError(f"Missing '--- ...' (file_mode_changes set to False, so expected file contents to change). Line was {lines[next_line_to_consume_index]}")
    next_line_to_consume_index += 1

    if lines[next_line_to_consume_index].startswith("+++ "):
        right_filename = lines[next_line_to_consume_index].removeprefix("+++ ")
    else:
        raise ParseError(f"Missing '+++ ...'. Line was {lines[next_line_to_consume_index]}")
    next_line_to_consume_index += 1

    hunks = []
    current_hunk = None
    index_of_last_line_consumed = 4
    for index_of_last_line_consumed, line in itertools.islice(enumerate(lines), next_line_to_consume_index, None):
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
        if file_diff is not None:
            result.append(file_diff)
    return result

def parse_diff_pair(output: str) -> ParsedDiffPair:
    output_split = output.split(DIFF_SEPARATOR)
    if len(output_split) == 1:
        return ParsedDiffPair(first_commit_diffs=parse_multiple_file_diffs(output_split[0]),
                              second_commit_diffs=[])
    return ParsedDiffPair(first_commit_diffs=parse_multiple_file_diffs(output_split[0]),
                          second_commit_diffs=parse_multiple_file_diffs(output_split[1]))


def get_left_file_lines_from_hunk(hunk: Hunk) -> SparseFileLines:
    return {(hunk.left_start_line_number + line_index): line[1:]
            for line_index, line in enumerate(hunk.lines)
            if line.startswith((" ", "-"))}

def get_right_file_lines_from_hunk(hunk: Hunk) -> SparseFileLines:
    return {(hunk.right_start_line_number + line_index): line[1:]
            for line_index, line in enumerate(hunk.lines)
            if line.startswith((" ", "+"))}

def get_left_commit_state_for_file(file_diff: FileDiff) -> SparseCommitState:
    sequence_of_dicts = (get_left_file_lines_from_hunk(hunk) for hunk in file_diff.hunks)
    left_file_lines = _merge_dict_sequence(sequence_of_dicts)
    return {file_diff.left_filename: left_file_lines}

def get_right_commit_state_for_file(file_diff: FileDiff) -> SparseCommitState:
    sequence_of_dicts = (get_right_file_lines_from_hunk(hunk) for hunk in file_diff.hunks)
    right_file_lines = _merge_dict_sequence(sequence_of_dicts)
    return {file_diff.right_filename: right_file_lines}

def get_left_commit_state_for_all_files(file_diffs: list[FileDiff]) -> SparseCommitState:
    return _merge_dict_sequence(get_left_commit_state_for_file(file_diff)
                                for file_diff in file_diffs)

def get_right_commit_state_for_all_files(file_diffs: list[FileDiff]) -> SparseCommitState:
    return _merge_dict_sequence(get_right_commit_state_for_file(file_diff)
                                for file_diff in file_diffs)

def iou_stats_between_files(a: SparseFileLines, b: SparseFileLines) -> IOUStats:
    a_set = set(a.items())
    b_set = set(b.items())
    return IOUStats(num_only_in_a=len(a_set - b_set),
                    num_only_in_b=len(b_set - a_set),
                    num_in_intersection=len(a_set & b_set))


def iou_stats_between_commits(commit_a: SparseCommitState, commit_b: SparseCommitState) -> IOUStats:
    list_of_iou_stats = []
    for file in commit_a.keys() | commit_b.keys():
        a_lines = commit_a.get(file, {})
        b_lines = commit_b.get(file, {})
        list_of_iou_stats.append(iou_stats_between_files(a_lines, b_lines))

    num_lines_only_in_a = sum(stats.num_only_in_a for stats in list_of_iou_stats)
    num_lines_only_in_b = sum(stats.num_only_in_b for stats in list_of_iou_stats)
    num_lines_in_intersection = sum(stats.num_in_intersection for stats in list_of_iou_stats)

    return IOUStats(num_only_in_a=num_lines_only_in_a,
                    num_only_in_b=num_lines_only_in_b,
                    num_in_intersection=num_lines_in_intersection)


def _mean_iou_between_diffs(predicted: ParsedDiffPair, ground_truth: ParsedDiffPair) -> float:
    predicted_first_left = get_left_commit_state_for_all_files(predicted.first_commit_diffs)
    predicted_first_right = get_right_commit_state_for_all_files(predicted.first_commit_diffs)
    predicted_second_left = get_left_commit_state_for_all_files(predicted.second_commit_diffs)
    predicted_second_right = get_right_commit_state_for_all_files(predicted.second_commit_diffs)

    ground_truth_first_left = get_left_commit_state_for_all_files(ground_truth.first_commit_diffs)
    ground_truth_first_right = get_right_commit_state_for_all_files(ground_truth.first_commit_diffs)
    ground_truth_second_left = get_left_commit_state_for_all_files(ground_truth.second_commit_diffs)
    ground_truth_second_right = get_right_commit_state_for_all_files(ground_truth.second_commit_diffs)

    iou = (iou_stats_between_commits(predicted_first_left, ground_truth_first_left).iou()
           + iou_stats_between_commits(predicted_first_right, ground_truth_first_right).iou()
           + iou_stats_between_commits(predicted_second_left, ground_truth_second_left).iou()
           + iou_stats_between_commits(predicted_second_right, ground_truth_second_right).iou())

    return iou / 4

def max_mean_iou_between_diffs(predicted: ParsedDiffPair, ground_truth: ParsedDiffPair) -> float:
    predicted_flipped = ParsedDiffPair(first_commit_diffs=predicted.second_commit_diffs,
                                       second_commit_diffs=predicted.first_commit_diffs)
    return max(_mean_iou_between_diffs(predicted, ground_truth),
               _mean_iou_between_diffs(predicted_flipped, ground_truth))


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