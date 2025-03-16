from transformers import AutoModelForCausalLM

MODEL_NAME = "fine_tuned_llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.push_to_hub("ggustafson/diff-splitter-llama-3.2-1B-7k-examples")
