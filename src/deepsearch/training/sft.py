from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SFTConfig:
    model_name_or_path: str
    train_file: str
    validation_file: str
    output_dir: str
    learning_rate: float = 1e-5
    epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    use_lora: bool = True


def train(config: SFTConfig) -> None:
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Install training dependencies with `pip install -e '.[train]'`.") from exc

    dataset = load_dataset(
        "json",
        data_files={"train": config.train_file, "validation": config.validation_file},
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    )
    if config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(full_text, truncation=True, max_length=config.max_length)
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=config.max_length,
        )["input_ids"]
        labels = list(encoded["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        encoded["labels"] = labels
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    arguments = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def main(default_task: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for DeepSearch models.")
    parser.add_argument("--task", choices=["filter", "selector"], default=default_task or "selector")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--full-finetune", action="store_true")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(
        SFTConfig(
            model_name_or_path=args.model,
            train_file=args.train_file,
            validation_file=args.validation_file,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            use_lora=not args.full_finetune,
        )
    )
