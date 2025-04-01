import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM

from peft import (
        get_peft_model, 
        LoraConfig
    )

from data.dataset import load_huggingface_dataset, get_combined_prompt_and_completion
from torch.profiler import profile, ProfilerActivity

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_TOKEN_LENGTH = 1536
PARQUET_DATASET_PATH = Path("data/combined-diffs-less-than-1000-chars.parquet")


def tokenize_prompt_and_completion(row_dict, tokenizer):
    text = row_dict["prompt_and_completion"]
    result = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_TOKEN_LENGTH)
    result["labels"] = result["input_ids"]
    return result


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
    del dataset["train_rl"]
    del dataset["test"]
    dataset = dataset.map(get_combined_prompt_and_completion)

    tokenized_datasets = dataset.map(num_proc=os.cpu_count(), function=lambda row: tokenize_prompt_and_completion(row, tokenizer))

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=4000,
        save_strategy="steps",
        save_steps=2000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        train_dataset=tokenized_datasets["train"].select(range(1)),
        eval_dataset=tokenized_datasets["validation"].select(range(1)),
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True) as prof:
        trainer.train()

    prof.export_chrome_trace("trace-train.json")

    trainer.save_model("./fine_tuned_llama-3.2-1B")
    tokenizer.save_pretrained("./fine_tuned_llama-3.2-1B")


if __name__ == "__main__":
    fine_tune_model(MODEL_NAME)

