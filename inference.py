from data.dataset import get_prompt
from diff_analyzer import parse_diff_pair
from train_with_rl import BASE_MODEL_NAME, MODEL_NAME, MAX_TOKEN_LENGTH

import torch
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
from peft import PeftModel, PeftConfig


import subprocess
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()
# saved_model_dir = directory_of_script / "fine_tuned_llama-3.2-1B"

#####################################

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Load the Lora model
model = PeftModel.from_pretrained(model, str(MODEL_NAME))
model = model.cuda()
model.eval()

#####################################

set_seed(42)
git_diff_result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
git_diff_str = git_diff_result.stdout.decode('utf-8')
input_str = get_prompt({"combined_diff": git_diff_str})["prompt"]
tokenizer_output = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=MAX_TOKEN_LENGTH)
with torch.inference_mode():
    generated_tokens = model.generate(input_ids=tokenizer_output.input_ids.cuda(),
                                      attention_mask=tokenizer_output.attention_mask.cuda(),
                                      max_new_tokens=2048,
                                      num_return_sequences=5,
                                      do_sample=True,
                                      top_p=0.9)
generated_texts = tokenizer.batch_decode(generated_tokens.detach().cpu().numpy(), skip_special_tokens=True)

print(f"input str: {input_str}")
# Adjust this to whatever's aesthetically pleasing.
TERMINAL_WIDTH = 239
for generated_text in generated_texts:
    print("-" * TERMINAL_WIDTH)
    print(generated_text[len(input_str):])

for generated_text in generated_texts:
    print(parse_diff_pair(generated_text[len(input_str):]))
