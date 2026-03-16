"""
train.py
────────
Main training entry point. Fine-tunes Llama 3.2 3B with QLoRA on Alpaca
using TRL's SFTTrainer.

Usage:
  python scripts/train.py
  python scripts/train.py --config configs/train_config.yaml

What happens:
  1. Load config, seed, tokenizer
  2. Load dataset (reproducible split, seed=42)
  3. Load base model in 4-bit NF4 quantization
  4. Attach LoRA adapters (r=16, alpha=32, 7 projection layers)
  5. Train for 1 epoch with SFTTrainer
  6. Save adapter weights to outputs/final_adapter/
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import yaml
import torch
from transformers import set_seed, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model

from src.data_utils import load_and_split, get_formatting_func
from src.model_utils import load_tokenizer, load_base_model, get_lora_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    torch.manual_seed(cfg["data"]["seed"])

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(cfg)

    # ── Dataset ─────────────────────────────────────────────────────────────
    print("Loading dataset...")
    splits = load_and_split(cfg)
    train_ds = splits["train"]
    print(f"  Training on {len(train_ds):,} rows")

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"Loading model: {cfg['model']['name']}")
    model = load_base_model(cfg)

    lora_config = get_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ──────────────────────────────────────────────────
    tc = cfg["training"]
    training_args = TrainingArguments(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        gradient_checkpointing=tc["gradient_checkpointing"],
        optim=tc["optim"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        max_grad_norm=tc["max_grad_norm"],
        logging_steps=tc["logging_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        report_to="none",  # disable wandb/tensorboard by default
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,         # we evaluate separately via evaluate.py
        formatting_func=get_formatting_func(tokenizer),
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=4,
        packing=False,             # cleaner loss signal for instruction tuning
        args=training_args,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("\nStarting training...")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    # ── Save ────────────────────────────────────────────────────────────────
    adapter_dir = tc["final_adapter_dir"]
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTraining complete.")
    print(f"  Time:        {hours}h {minutes}m {seconds}s")
    print(f"  Final loss:  {result.training_loss:.4f}")
    print(f"  Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
