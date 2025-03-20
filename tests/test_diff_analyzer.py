import unittest

import sys
import os
# Enable importing modules located in parent directory.
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from diff_analyzer import parse_file_diff_from_lines, FileDiff, Hunk, parse_multiple_file_diffs, DIFF_SEPARATOR, \
    parse_diff_pair, ParsedDiffPair, IOUStats, iou_stats_between_files, iou_stats_between_commits

_TWO_FILES_DIFF_STR = """diff --git a/file1.txt b/file1.txt
index 88de97e..7ca04a1 100644
--- a/file1.txt
+++ b/file1.txt
@@ -1,1 +1,1 @@
+I am file1
diff --git a/file2.txt b/file2.txt
index 88de97e..7ca04a1 100644
--- a/file2.txt
+++ b/file2.txt
@@ -1,1 +1,1 @@
+I am file2"""

_TWO_FILES_DIFF_RESULT = [FileDiff(left_filename="a/file1.txt",
                                   right_filename="b/file1.txt",
                                   hunks=[Hunk(left_start_line_number=1,
                                                left_num_lines=1,
                                                right_start_line_number=1,
                                                right_num_lines=1,
                                                lines=["+I am file1"])]),
                          FileDiff(left_filename="a/file2.txt",
                                    right_filename="b/file2.txt",
                                    hunks=[Hunk(left_start_line_number=1,
                                                left_num_lines=1,
                                                right_start_line_number=1,
                                                right_num_lines=1,
                                                lines=["+I am file2"])])]

class TestDiffAnalyzer(unittest.TestCase):
    def test_parse_file_diff_from_lines(self):
        diff_str = """diff --git a/inference.py b/inference.py
index 88de97e..7ca04a1 100644
--- a/inference.py
+++ b/inference.py
@@ -1,5 +1,5 @@
 from data.dataset import END_COMBINED_DIFF_MARKER
-from train_with_rl import BASE_MODEL_NAME, MODEL_NAME, MAX_TOKEN_LENGTH
+from train import MODEL_NAME, MAX_TOKEN_LENGTH

 import torch
 from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
@@ -16,7 +16,7 @@ saved_model_dir = directory_of_script / "fine_tuned_llama-3.2-1B"

 # load base LLM model and tokenizer
 model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
-tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
+tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

 # Load the Lora model
 model = PeftModel.from_pretrained(model, str(saved_model_dir))
"""
        split_diff_str = diff_str.split("\n")
        result = parse_file_diff_from_lines(split_diff_str)
        expected_result = FileDiff(left_filename="a/inference.py",
                                   right_filename="b/inference.py",
                                   hunks=[
                                       Hunk(left_start_line_number=1,
                                            left_num_lines=5,
                                            right_start_line_number=1,
                                            right_num_lines=5,
                                            lines=""" from data.dataset import END_COMBINED_DIFF_MARKER
-from train_with_rl import BASE_MODEL_NAME, MODEL_NAME, MAX_TOKEN_LENGTH
+from train import MODEL_NAME, MAX_TOKEN_LENGTH

 import torch
 from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer""".split("\n")),
                                       Hunk(left_start_line_number=16,
                                            left_num_lines=7,
                                            right_start_line_number=16,
                                            right_num_lines=7,
                                            lines="""
 # load base LLM model and tokenizer
 model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
-tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
+tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

 # Load the Lora model
 model = PeftModel.from_pretrained(model, str(saved_model_dir))
""".split("\n"))])
        assert result == (expected_result, len(split_diff_str))
        
    def test_parse_file_diff_from_lines_5_lines_header(self):
        diff_str = """diff --git a/eval.py b/eval.py
new file mode 100644
index 0000000..f2e997b
--- /dev/null
+++ b/eval.py
@@ -1,1 +1,1 @@
+dummy change"""
        split_diff_str = diff_str.split("\n")
        result = parse_file_diff_from_lines(split_diff_str)
        expected_result = FileDiff(left_filename="/dev/null",
                                   right_filename="b/eval.py",
                                   hunks=[
                                       Hunk(left_start_line_number=1,
                                            left_num_lines=1,
                                            right_start_line_number=1,
                                            right_num_lines=1,
                                            lines=["+dummy change"])])
        assert result == (expected_result, 7)

    def test_parse_file_diff_with_mode_changes_only(self):
        diff_str = """diff --git a/filename1 b/filename2
new file mode 100644
index 000000000..e69de29bb"""
        split_diff_str = diff_str.split("\n")
        result = parse_file_diff_from_lines(split_diff_str)
        expected_result = FileDiff(left_filename="a/filename1",
                                   right_filename="b/filename2",
                                   hunks=[])
        assert result == (expected_result, 3)

    def test_parse_multiple_file_diffs(self):
        result = parse_multiple_file_diffs(_TWO_FILES_DIFF_STR)
        expected_result = _TWO_FILES_DIFF_RESULT
        assert result == expected_result

    def test_parse_model_output(self):
        model_output_str = DIFF_SEPARATOR.join([_TWO_FILES_DIFF_STR, _TWO_FILES_DIFF_STR])
        result = parse_diff_pair(model_output_str)
        expected_result = ParsedDiffPair(first_commit_diffs=_TWO_FILES_DIFF_RESULT,
                                         second_commit_diffs=_TWO_FILES_DIFF_RESULT)
        assert result == expected_result

    def test_iou_stats_between_files(self):
        iou_stats = iou_stats_between_files({1: "a", 2: "b"}, {2: "b", 3: "c"})
        assert iou_stats == IOUStats(num_only_in_a=1, num_only_in_b=1, num_in_intersection=1)
        iou_stats = iou_stats_between_files({1: "a", 2: "b"}, {})
        assert iou_stats == IOUStats(num_only_in_a=2, num_only_in_b=0, num_in_intersection=0)
        iou_stats = iou_stats_between_files({}, {2: "b", 3: "c"})
        assert iou_stats == IOUStats(num_only_in_a=0, num_only_in_b=2, num_in_intersection=0)
        iou_stats = iou_stats_between_files({2: "b", 3: "c"}, {2: "b", 3: "c"})
        assert iou_stats == IOUStats(num_only_in_a=0, num_only_in_b=0, num_in_intersection=2)

    def test_iou_stats_between_commits(self):
        commit_a = {"file1": {1: "a", 2: "b"}, "file2": {1: "a", 2: "b"}}
        commit_b = {"file2": {2: "b", 3: "c"}, "file3": {2: "b", 3: "c", 4: "d"}}
        iou_stats = iou_stats_between_commits(commit_a, commit_b)
        assert iou_stats == IOUStats(num_only_in_a=3, num_only_in_b=4, num_in_intersection=1)

    def test_iou(self):
        assert IOUStats(num_only_in_a=3, num_only_in_b=4, num_in_intersection=1).iou() == 0.125


if __name__ == '__main__':
    unittest.main()

