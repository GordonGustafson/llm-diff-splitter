from transformers import pipeline, set_seed

import subprocess
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()
saved_model_dir = directory_of_script / "fine_tuned_gpt2"

generator = pipeline('text-generation', model=str(saved_model_dir))
set_seed(42)
git_diff_result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
git_diff_str = git_diff_result.stdout.decode('utf-8')
print(f"git diff str: {git_diff_str}")
result = generator(git_diff_str, max_length=1024, num_return_sequences=1)
print(result[0]["generated_text"])


