from transformers import pipeline, set_seed

import subprocess
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()
saved_model_dir = directory_of_script / "fine_tuned_llama-3.2-1B"

generator = pipeline('text-generation', model=str(saved_model_dir))
set_seed(42)
git_diff_result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
git_diff_str = git_diff_result.stdout.decode('utf-8')
input_str = git_diff_str + " END DIFF "
result = generator(input_str, max_length=1024, num_return_sequences=1, do_sample=False)

print(f"input str: {input_str}")

print(f"generated_text: {result[0]['generated_text'][len(input_str):]}")
