from data.dataset import END_COMBINED_DIFF_MARKER
from train import MODEL_NAME, MAX_TOKEN_LENGTH

import torch
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
from peft import PeftModel, PeftConfig


import subprocess
import pathlib

directory_of_script = pathlib.Path(__file__).parent.resolve()
saved_model_dir = directory_of_script / "fine_tuned_llama-3.2-1B"

#####################################

# Load peft config for pre-trained checkpoint etc.
config = PeftConfig.from_pretrained(str(saved_model_dir))

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the Lora model
model = PeftModel.from_pretrained(model, str(saved_model_dir))
model = model.cuda()
model.eval()

#####################################

set_seed(42)
git_diff_result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
git_diff_str = git_diff_result.stdout.decode('utf-8')
input_str = f"{git_diff_str} {END_COMBINED_DIFF_MARKER} "
input_ids = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=MAX_TOKEN_LENGTH).input_ids.cuda()
with torch.inference_mode():
    generated_tokens = model.generate(input_ids=input_ids, max_new_tokens=2048,  num_return_sequences=5, do_sample=True, top_p=0.9)
generated_texts = tokenizer.batch_decode(generated_tokens.detach().cpu().numpy(), skip_special_tokens=True)

print(f"input str: {input_str}")


for generated_text in generated_texts:
    print(f"generated_text: {generated_text}")
