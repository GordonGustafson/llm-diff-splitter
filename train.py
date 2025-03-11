import os
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from data.dataset import load_huggingface_dataset


PARQUET_DATASET_PATH = Path("data/output.parquet")


def tokenize_function(row_dict, tokenizer):
    text = row_dict["text"]
    result = tokenizer(text, padding="max_length", truncation=True, max_length=1024)
    result["labels"] = result["input_ids"]
    return result


def fine_tune_gpt2(model_name="openai-community/gpt2-medium"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    dataset = load_huggingface_dataset(PARQUET_DATASET_PATH)
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    tokenized_datasets = dataset.map(num_proc=os.cpu_count(), function=lambda row: tokenize_function(row, tokenizer))

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")


if __name__ == "__main__":
    fine_tune_gpt2()

