import unittest

# Enable importing modules located in parent directory.
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from diff_analyzer import parse_file_diff, FileDiff, Hunk

class TestDiffAnalyzer(unittest.TestCase):
    def test_parse_file_diff(self):
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
        result = parse_file_diff(split_diff_str)
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
        assert result == (expected_result, len(split_diff_str) - 1)


if __name__ == '__main__':
    unittest.main()

