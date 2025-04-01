import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TopPLogitsWarper, LogitsProcessorList, \
    StoppingCriteriaList

from peft import PeftModel

from data.dataset import load_huggingface_dataset, get_separate_prompt_and_completion
from diff_analyzer import parse_diff_pair, ParseError, max_mean_iou_between_diffs

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_NAME = "ggustafson/diff-splitter-llama-3.2-1B-7k-examples"
MAX_TOKEN_LENGTH = 512
PARQUET_DATASET_PATH = Path("data/combined-diffs-less-than-1000-chars.parquet")


def tokenize_prompt(row_dict, tokenizer):
    text = row_dict["prompt"]
    result = tokenizer(text, truncation=True, max_length=MAX_TOKEN_LENGTH, padding="longest", padding_side="left", return_tensors="pt")
    return result

def compute_loss(transition_scores, prompt_tokens, generated_tokens_without_prompt, ground_truth_completion, tokenizer) -> torch.Tensor:
    prompt_text = tokenizer.batch_decode(prompt_tokens)[0].replace('\\n', '\n')
    generated_text_without_prompt = tokenizer.batch_decode(generated_tokens_without_prompt)[0].replace('\\n', '\n')
    ground_truth_completion_text = ground_truth_completion[0]
    selected_log_probabilities = transition_scores

    print(f"prompt text:\n{prompt_text}")
    print("-" * 239)
    print(f"generated_text_without_prompt:\n{generated_text_without_prompt}")
    print("-" * 239)
    print(f"ground_truth_completion_text:\n{ground_truth_completion_text}")
    print("-" * 239)

    try:
        parsed_ground_truth_diff_pair = parse_diff_pair(ground_truth_completion_text)
    except Exception as e:
        print(f"got error {e} when parsing ground truth diff: {ground_truth_completion_text}")
        raise e

    try:
        parsed_diff_pair = parse_diff_pair(generated_text_without_prompt)
    except ParseError as e:
        print(f"got ParseError {e} when parsing generated diff")
        reward = -1.0
    except Exception as e:
        print(f"got unexpected exception {e} when parsing generated diff")
        reward = -1.0
    else:
        raw_reward = max_mean_iou_between_diffs(predicted=parsed_diff_pair,
                                                ground_truth=parsed_ground_truth_diff_pair)
        baseline_reward = 0.5
        reward = raw_reward - baseline_reward

    print(f"reward: {reward}")

    train_loss_items = - selected_log_probabilities * reward
    total_train_loss = train_loss_items.sum()
    return total_train_loss

def fine_tune_model(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_name, is_trainable=True)

    train_dataset = load_huggingface_dataset(PARQUET_DATASET_PATH)["train_rl"]
    train_dataset = train_dataset.map(get_separate_prompt_and_completion)

    tokenized_train_dataset = train_dataset.map(num_proc=os.cpu_count(),
                                                function=lambda row: tokenize_prompt(row, tokenizer))
    tokenized_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "completion"])
    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=1)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=5e-7,
                                  betas=(0.9, 0.999),
                                  weight_decay=0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    generation_config = GenerationConfig(return_dict_in_generate=True,
                                         output_scores=True,
                                         max_length=MAX_TOKEN_LENGTH,
                                         do_sample=True,
                                         top_p=0.9)
    logits_processor = LogitsProcessorList()
    logits_processor.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))
    # This is needed to call model._get_stopping_criteria
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask=True, device=device)
    stopping_criteria = model._get_stopping_criteria(generation_config=generation_config,
                                                     stopping_criteria=StoppingCriteriaList(),
                                                     tokenizer=None)

    for batch_index, batch in enumerate(train_dataloader):
        print(f"batch {batch_index} out of {len(train_dataloader)}")
        batch["input_ids"] = batch["input_ids"].to(device).squeeze(0)
        batch["attention_mask"] = batch["attention_mask"].to(device).squeeze(0)

        outputs = model._sample(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                logits_processor=logits_processor,
                                stopping_criteria=stopping_criteria,
                                generation_config=generation_config,
                                synced_gpus=False,
                                streamer=None)

        input_length = batch["input_ids"].shape[1]
        generated_tokens_without_prompt = outputs.sequences[:, input_length:]

        ground_truth_completion = batch["completion"]

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        loss = compute_loss(transition_scores=transition_scores,
                            prompt_tokens=batch["input_ids"],
                            generated_tokens_without_prompt=generated_tokens_without_prompt,
                            ground_truth_completion=ground_truth_completion,
                            tokenizer=tokenizer)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_index % 150 == 149:
            model.save_pretrained(f"./fine_tuned_llama-3.2-1B_rl_batch_{batch_index}")

    model.save_pretrained("./fine_tuned_llama-3.2-1B_rl_final")
    tokenizer.save_pretrained("./fine_tuned_llama-3.2-1B_rl_final")

if __name__ == "__main__":
    fine_tune_model(MODEL_NAME)

