import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel

from data.dataset import load_huggingface_dataset, get_separate_prompt_and_completion
from diff_analyzer import get_diff_metrics, diff_metrics_to_reward, parse_diff_pair

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "ggustafson/diff-splitter-llama-3.2-1B-7k-examples"
MAX_TOKEN_LENGTH = 1536
PARQUET_DATASET_PATH = Path("data/combined-diffs-less-than-1000-chars.parquet")


def tokenize_prompt(row_dict, tokenizer):
    text = row_dict["prompt"]
    print(f"len(text): {len(text)}")
    result = tokenizer(text, truncation=True, max_length=MAX_TOKEN_LENGTH, padding='longest')
    return result

def compute_loss(transition_scores, prompt_tokens, generated_tokens, ground_truth_completion, tokenizer) -> torch.Tensor:
    prompt_text = tokenizer.batch_decode(prompt_tokens)[0].replace('\\n', '\n')
    generated_text = tokenizer.batch_decode(generated_tokens)[0].replace('\\n', '\n')
    ground_truth_completion_text = ground_truth_completion[0]
    selected_log_probabilities = transition_scores

    diff_metrics = get_diff_metrics(combined_diff=prompt_text, generated_diff=generated_text)
    reward = diff_metrics_to_reward(diff_metrics)
    print(f"prompt text:\n{prompt_text}")
    print("-" * 239)
    print(f"generated_text:\n{generated_text}")
    print("-" * 239)
    print(f"ground_truth_completion_text:\n{ground_truth_completion_text}")
    print("-" * 239)
    try:
        model_output = parse_diff_pair(generated_text)
        print(model_output)
    except Exception as e:
        print(e)
    print(f"diff metrics: {diff_metrics}")
    print(f"rewards: {reward}")

    train_loss_items = - selected_log_probabilities * reward
    total_train_loss = train_loss_items.sum()
    total_train_loss.requires_grad = True
    return total_train_loss

def fine_tune_model(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, model_name)

    dataset = load_huggingface_dataset(PARQUET_DATASET_PATH)
    dataset = dataset.map(get_separate_prompt_and_completion)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    tokenized_datasets = dataset.map(num_proc=os.cpu_count(), function=lambda row: tokenize_prompt(row, tokenizer))
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "completion"])
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=1)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=math.exp(-4),
                                  betas=(0.9, 0.999),
                                  weight_decay=0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

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

        ground_truth_completion = batch["completion"]

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        loss = compute_loss(transition_scores=transition_scores,
                            prompt_tokens=batch["input_ids"],
                            generated_tokens=generated_tokens,
                            ground_truth_completion=ground_truth_completion,
                            tokenizer=tokenizer)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.save_pretrained("./fine_tuned_llama-3.2-1B_rl")
    tokenizer.save_pretrained("./fine_tuned_llama-3.2-1B_rl")

if __name__ == "__main__":
    fine_tune_model(MODEL_NAME)

