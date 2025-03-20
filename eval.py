from data.dataset import load_huggingface_dataset, get_separate_prompt_and_completion
from diff_analyzer import parse_model_output
from train_with_rl import BASE_MODEL_NAME, MODEL_NAME, MAX_TOKEN_LENGTH, tokenize_prompt

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
from peft import PeftModel


import os
from pathlib import Path

directory_of_script = Path(__file__).parent.resolve()
# saved_model_dir = directory_of_script / "fine_tuned_llama-3.2-1B"

PARQUET_DATASET_PATH = Path("data/combined-diffs-less-than-1000-chars.parquet")

#####################################


def run_on_eval_set():
    # load base LLM model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, str(MODEL_NAME))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    set_seed(42)
    dataset = load_huggingface_dataset(PARQUET_DATASET_PATH)
    dataset = dataset.map(get_separate_prompt_and_completion)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    tokenized_datasets = dataset.map(num_proc=os.cpu_count(), function=lambda row: tokenize_prompt(row, tokenizer))
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "completion"])
    eval_dataset = tokenized_datasets["test"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    num_parseable_outputs = 0
    num_unparseable_outputs = 0

    with torch.inference_mode():
        for batch in eval_dataloader:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            outputs = model.generate(batch["input_ids"],
                                     attention_mask=batch["attention_mask"],
                                     return_dict_in_generate=True,
                                     output_scores=True,
                                     max_length=MAX_TOKEN_LENGTH,
                                     do_sample=True,
                                     top_p=0.9)

            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs.sequences[:, input_length:]

            prompt_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)[0].replace('\\n', '\n')
            text_produced_by_model = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].replace('\\n', '\n')
            ground_truth_completion_text  = batch["completion"][0]

            print(f"input str: {prompt_text}")
            # Adjust this to whatever's aesthetically pleasing.
            TERMINAL_WIDTH = 239
            print("-" * TERMINAL_WIDTH)
            print(text_produced_by_model)

            try:
                print(parse_model_output(text_produced_by_model))
                num_parseable_outputs += 1
            except Exception as e:
                print(e)
                num_unparseable_outputs += 1
            print("-" * TERMINAL_WIDTH)

    print(f"{num_parseable_outputs} parseable outputs and {num_unparseable_outputs} unparseable outputs out of {len(eval_dataset)} total outputs")


if __name__ == "__main__":
    run_on_eval_set()
