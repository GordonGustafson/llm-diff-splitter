from data.dataset import load_huggingface_dataset, get_separate_prompt_and_completion
from diff_analyzer import parse_diff_pair, max_mean_iou_between_diffs, ParseError, DIFF_SEPARATOR
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
BATCH_SIZE = 12

#####################################


def run_on_eval_set():
    # load base LLM model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the Lora model
    model = PeftModel.from_pretrained(model, str(MODEL_NAME))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    set_seed(42)
    dataset = load_huggingface_dataset(PARQUET_DATASET_PATH)["validation"]
    dataset = dataset.map(get_separate_prompt_and_completion)

    tokenized_dataset = dataset.map(function=lambda row: tokenize_prompt(row, tokenizer),
                                    batched=True,
                                    batch_size=BATCH_SIZE)

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "completion"])
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE)

    num_diff_separator_not_found = 0
    num_parseable_outputs = 0
    num_unparseable_outputs = 0
    total_max_mean_iou = 0.0

    num_batches = len(tokenized_dataset) // BATCH_SIZE

    with torch.inference_mode():
        for batch_index, batch in enumerate(eval_dataloader):
            print(f"Batch {batch_index} out of {num_batches}")
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            outputs = model.generate(batch["input_ids"],
                                     attention_mask=batch["attention_mask"],
                                     return_dict_in_generate=True,
                                     # 512 is too short for some examples in the validation set.
                                     max_length=1024,
                                     do_sample=True,
                                     top_p=0.9)

            prompt_text_batch = [text.replace('\\n', '\n') for text in
                                 tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)]
            model_output_with_prompt_batch = [text.replace('\\n', '\n') for text in
                                              tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)]
            ground_truth_completion_text_batch = batch["completion"]

            for prompt_text, model_output_with_prompt, ground_truth_completion_text in zip(prompt_text_batch,
                                                                                           model_output_with_prompt_batch,
                                                                                           ground_truth_completion_text_batch):
                try:
                    parsed_ground_truth_diff_pair = parse_diff_pair(ground_truth_completion_text)
                except Exception as e:
                    print(f"got error {e} when parsing ground truth diff: {ground_truth_completion_text}")
                    raise e
                # Remove the prompt, which ends with DIFF_SEPARATOR.
                try:
                    text_produced_by_model = model_output_with_prompt.split(DIFF_SEPARATOR, 1)[1]
                except IndexError as e:
                    print("Could not find DIFF_SEPARATOR in the model output including the prompt! "
                          "Out best guess is that this is due to the token limit truncating the output. "
                          "Skipping this sample.")
                    num_diff_separator_not_found += 1
                    continue

                print(f"input str: {prompt_text}")
                # Adjust this to whatever's aesthetically pleasing.
                TERMINAL_WIDTH = 239
                print("-" * TERMINAL_WIDTH)
                print(f"model produced (without prompt): {text_produced_by_model}")
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
                    print(f"max_mean_iou for this diff: {max_mean_iou}")
                    print(f"mean max_mean_iou for all parseable diffs so far: {total_max_mean_iou / num_parseable_outputs}")
                    print(f"{num_parseable_outputs} parseable outputs and {num_unparseable_outputs} unparseable outputs out of {len(tokenized_dataset)} total outputs")
                    print(f"Number of times diff separator wasn't found: {num_diff_separator_not_found}")

    print(f"mean max_mean_iou: {total_max_mean_iou / num_parseable_outputs}")
    print(f"{num_parseable_outputs} parseable outputs and {num_unparseable_outputs} unparseable outputs out of {len(tokenized_dataset)} total outputs")
    print(f"Number of times diff separator wasn't found: {num_diff_separator_not_found}")


if __name__ == "__main__":
    run_on_eval_set()
