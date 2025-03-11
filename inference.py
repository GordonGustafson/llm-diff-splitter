from transformers import pipeline, set_seed
import tiktoken

import subprocess
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()
saved_model_dir = directory_of_script / "fine_tuned_gpt2"

generator = pipeline('text-generation', model=str(saved_model_dir))
set_seed(42)
git_diff_result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
git_diff_str = git_diff_result.stdout.decode('utf-8')
result = generator(git_diff_str, max_length=1024, num_return_sequences=1, temperature=0.5)

encoding = tiktoken.encoding_for_model("gpt2")
git_diff_num_tokens = len(encoding.encode(git_diff_str))
print(f"git diff str with {git_diff_num_tokens} tokens: {git_diff_str}")

result_num_tokens = [len(encoding.encode(res['generated_text'])) for res in result]
print(f"generated_text with {result_num_tokens[0]} tokens: {result[0]['generated_text']}")


