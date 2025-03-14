import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import (
    get_peft_model,
    LoraConfig
)

from data.dataset import load_huggingface_dataset
from diff_analyzer import get_diff_metrics, diff_metrics_to_reward

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_TOKEN_LENGTH = 1536
PARQUET_DATASET_PATH = Path("data/combined-diffs-less-than-1000-chars.parquet")


def tokenize_function(row_dict, tokenizer):
    text = row_dict["text"]
    result = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_TOKEN_LENGTH)
    result["labels"] = result["input_ids"]
    return result

def compute_loss(model_outputs) -> torch.Tensor:
    # This is probably the wrong way to extract the logits and generated text.
    logits = model_outputs["logits"]
    generated_text = model_outputs["generated_text"]
    log_probabilities = torch.nn.functional.log_softmax(logits)
    diff_metrics = get_diff_metrics(generated_text)
    reward = diff_metrics_to_reward(diff_metrics)

    train_loss_items = - log_probabilities * reward
    total_train_loss = train_loss_items.sum()
    return total_train_loss

def fine_tune_model(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    dataset = load_huggingface_dataset(PARQUET_DATASET_PATH)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    tokenized_datasets = dataset.map(num_proc=os.cpu_count(), function=lambda row: tokenize_function(row, tokenizer))
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=1)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=math.exp(-4),
                                  betas=(0.9, 0.999),
                                  weight_decay=0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = compute_loss(outputs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.save_pretrained("./fine_tuned_llama-3.2-1B")
    tokenizer.save_pretrained("./fine_tuned_llama-3.2-1B")

if __name__ == "__main__":
    fine_tune_model(MODEL_NAME)

