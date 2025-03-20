from data.dataset import load_huggingface_dataset, get_separate_prompt_and_completion
from diff_analyzer import parse_diff_pair, max_mean_iou_between_diffs, ParseError
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
BATCH_SIZE = 2

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
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = dataset.map(get_separate_prompt_and_completion)

    tokenized_datasets = dataset.map(num_proc=os.cpu_count(),
                                     function=lambda row: tokenize_prompt(row, tokenizer),
                                     batched=True,
                                     batch_size=BATCH_SIZE)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "completion"])
    eval_dataset = tokenized_datasets["test"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    num_parseable_outputs = 0
    num_unparseable_outputs = 0
    total_max_mean_iou = 0.0

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

            prompt_text_batch = [text.replace('\\n', '\n') for text in
                           tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)]
            text_produced_by_model_batch = [text.replace('\\n', '\n') for text in
                                      tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)]
            ground_truth_completion_text_batch  = batch["completion"]

            for prompt_text, text_produced_by_model, ground_truth_completion_text in zip(prompt_text_batch,
                                                                                         text_produced_by_model_batch,
                                                                                         ground_truth_completion_text_batch):
                try:
                    parsed_ground_truth_diff_pair = parse_diff_pair(ground_truth_completion_text)
                except Exception as e:
                    print(f"got error {e} when parsing ground truth diff: {ground_truth_completion_text}")
                    raise e

                print(f"input str: {prompt_text}")
                # Adjust this to whatever's aesthetically pleasing.
                TERMINAL_WIDTH = 239
                print("-" * TERMINAL_WIDTH)
                print(f"model produced: {text_produced_by_model}")
                print("-" * TERMINAL_WIDTH)
                print(f"ground truth: {ground_truth_completion_text}")
                print("-" * TERMINAL_WIDTH)

                try:
                    parsed_diff_pair = parse_diff_pair(text_produced_by_model)
                except ParseError as e:
                    print(f"got error {e} when parsing generated diff")
                    num_unparseable_outputs += 1
                else:
                    num_parseable_outputs += 1
                    max_mean_iou = max_mean_iou_between_diffs(predicted=parsed_diff_pair,
                                                              ground_truth=parsed_ground_truth_diff_pair)
                    total_max_mean_iou += max_mean_iou
                    print(f"max_mean_iou: {max_mean_iou}")
                    print(f"mean_max_iou: {total_max_mean_iou / num_parseable_outputs}")

    print(f"mean_max_iou: {total_max_mean_iou / num_parseable_outputs}")
    print(f"{num_parseable_outputs} parseable outputs and {num_unparseable_outputs} unparseable outputs out of {len(eval_dataset)} total outputs")


if __name__ == "__main__":
    run_on_eval_set()
