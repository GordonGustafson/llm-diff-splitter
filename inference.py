from data.dataset import END_COMBINED_DIFF_MARKER

from transformers import pipeline, set_seed

import subprocess
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()
saved_model_dir = directory_of_script / "fine_tuned_llama-3.2-1B"

generator = pipeline('text-generation', model=str(saved_model_dir))
set_seed(42)
git_diff_result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
git_diff_str = git_diff_result.stdout.decode('utf-8')
input_str = f"{git_diff_str} {END_COMBINED_DIFF_MARKER} "
generated_texts = generator(input_str, num_return_sequences=5, do_sample=False)

print(f"input str: {input_str}")

for generated_text in generated_texts:
    print(f"generated_text: {generated_text['generated_text'][len(input_str):]}")
