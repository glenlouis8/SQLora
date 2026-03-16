# 06 — `scripts/train.py`

## What this file is

`train.py` is the core training script. It orchestrates the entire fine-tuning process: loading the config, setting random seeds, loading the tokenizer, downloading and splitting the dataset, loading the base model in 4-bit NF4 quantization, attaching LoRA adapters, configuring the training loop via HuggingFace's `TrainingArguments`, running training with TRL's `SFTTrainer`, and saving the resulting adapter weights. On an RTX 4090, this takes roughly 2.5 hours for one epoch over 49,000 training examples.

## How it connects

- **Called by:** the user via `python scripts/train.py`.
- **Calls:** `src.data_utils.load_and_split`, `src.data_utils.get_formatting_func`, `src.model_utils.load_tokenizer`, `src.model_utils.load_base_model`, `src.model_utils.get_lora_config`, `peft.get_peft_model`, `transformers.TrainingArguments`, `trl.SFTTrainer`.
- **Reads:** `configs/train_config.yaml` (via `--config` flag).
- **Writes:** `outputs/checkpoints/` (mid-training checkpoints), `outputs/final_adapter/` (final adapter weights and tokenizer files).
- **Fits in the pipeline as:** step 3 — runs after `prepare_data.py` and `evaluate.py --stage before`, and before `evaluate.py --stage after`.

---

## Full walkthrough

### Imports

```python
import argparse
import time
import yaml
import torch
from transformers import set_seed, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model

from src.data_utils import load_and_split, get_formatting_func
from src.model_utils import load_tokenizer, load_base_model, get_lora_config
```

- **`time`**: used to measure total training duration and print a human-readable elapsed time at the end.
- **`set_seed`**: HuggingFace's function to seed Python's `random`, NumPy, and PyTorch simultaneously with a single call.
- **`TrainingArguments`**: a large dataclass from HuggingFace `transformers` that holds every training hyperparameter. It is passed directly to `SFTTrainer`.
- **`SFTTrainer`**: from the TRL (Transformer Reinforcement Learning) library. A specialized `Trainer` subclass designed for Supervised Fine-Tuning (SFT) — the process of fine-tuning a model on (prompt, response) pairs with proper loss masking.
- **`get_peft_model`**: wraps a base model with LoRA adapters and freezes the base weights.

---

### Setting up the run

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    torch.manual_seed(cfg["data"]["seed"])
```

**`set_seed(cfg["data"]["seed"])`** sets random seeds for Python's `random` module, NumPy, and PyTorch's CPU operations simultaneously. This makes dataset shuffling, weight initialization (if any), and dropout patterns reproducible.

**`torch.manual_seed(cfg["data"]["seed"])`** additionally seeds PyTorch's GPU random number generator. This is separate from `set_seed` and is needed for full GPU reproducibility. Together, these two calls mean: given the same config, hardware, and library versions, two runs will produce identical training trajectories.

---

### Loading the tokenizer

```python
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(cfg)
```

A thin call into `model_utils.py`. See `03_model_utils.md` for the detailed explanation of what `load_tokenizer` does (`pad_token` and `padding_side` setup). The tokenizer is loaded before the model because it is needed to build the formatting function that the trainer will use.

---

### Loading and splitting the dataset

```python
    print("Loading dataset...")
    splits = load_and_split(cfg)
    train_ds = splits["train"]
    print(f"  Training on {len(train_ds):,} rows")
```

Only the training split is extracted here. The test split is not used during training — it is reserved for `evaluate.py`. The print statement confirms the expected row count (~49,401) before the slow model loading begins.

---

### Loading the base model and attaching LoRA

```python
    print(f"Loading model: {cfg['model']['name']}")
    model = load_base_model(cfg)

    lora_config = get_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
```

**`load_base_model(cfg)`** downloads the 3B parameter Llama model and quantizes it to 4-bit NF4 as it loads. This is the most memory-intensive operation in the entire script. On first run, it also downloads ~6GB of model weights to the HuggingFace cache.

**`get_peft_model(model, lora_config)`** is the step that makes this QLoRA rather than full fine-tuning. It modifies the model in three ways:
1. Freezes all original parameters: `requires_grad = False` for all 3.2 billion of them.
2. Inserts two small matrices (A and B) alongside each of the 7 target layer types in each of the model's 28 transformer blocks (3.2B / 28 blocks × 7 modules × 2 matrices each ≈ ~20 million new parameters).
3. Sets only these new matrices to `requires_grad = True`.

**`model.print_trainable_parameters()`** prints something like:
```
trainable params: 20,054,016 || all params: 3,233,349,632 || trainable%: 0.6202
```
This is a critical sanity check — you want to see a small percentage. If you accidentally unfroze the base model or targeted too many modules, this number would be much higher.

The reduction from 3.2B to 20M trainable parameters is what makes it possible to use the 8-bit AdamW optimizer and gradient checkpointing without running out of VRAM.

---

### Configuring training arguments

```python
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
        report_to="none",
    )
```

`tc = cfg["training"]` is just a local alias to avoid writing `cfg["training"]` for every argument — a readability shortcut.

`TrainingArguments` is a large configuration object. It validates all these settings (e.g., ensures `fp16` and `bf16` are not both `True`) and makes them available to the `Trainer`. All of these settings are described in detail in `01_train_config.md`.

**`report_to="none"`** is an important default. By default, HuggingFace `Trainer` tries to log to Weights & Biases (`wandb`) or TensorBoard if they are installed. This can cause unexpected authentication prompts or errors. `"none"` disables all external logging. If you want to enable `wandb` monitoring, change this to `"wandb"` and ensure you have authenticated.

---

### Configuring SFTTrainer

```python
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,
        formatting_func=get_formatting_func(tokenizer),
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )
```

**`SFTTrainer`** is TRL's subclass of HuggingFace `Trainer`, specialized for supervised fine-tuning. The key difference from a plain `Trainer` is that `SFTTrainer` handles:
1. **Loss masking**: when using a chat template with an assistant turn, it computes loss only on the assistant's tokens, not on the system or user tokens. This is correct behavior for instruction fine-tuning — you want the model to learn to generate good responses, not to predict the instruction text.
2. **Formatting**: accepts a `formatting_func` that converts raw dataset rows to strings, so you do not need to pre-tokenize the entire dataset before training.
3. **Sequence packing**: optionally concatenates multiple short examples into a single training sequence to maximize GPU utilization.

**`model=model`**: the PEFT-wrapped model with LoRA adapters. `SFTTrainer` knows how to work with PEFT models.

**`train_dataset=train_ds`**: the raw HuggingFace dataset (not yet tokenized). `SFTTrainer` will apply `formatting_func` and then tokenize on the fly.

**`eval_dataset=None`**: we do not run in-training evaluation here. Evaluation is handled separately by `evaluate.py` with more comprehensive metrics (perplexity + ROUGE-L) than `SFTTrainer`'s default loss-only evaluation. This also speeds up training by avoiding periodic evaluation pauses.

**`formatting_func=get_formatting_func(tokenizer)`**: the closure returned by `data_utils.get_formatting_func()`. Called by `SFTTrainer` on each batch during the data collation step.

**`max_seq_length=cfg["model"]["max_seq_length"]`**: tells `SFTTrainer` to truncate formatted examples to this length. This is applied after `formatting_func` but before tokenization.

**`dataset_num_proc=4`**: uses 4 CPU worker processes to format and tokenize the dataset in parallel. This pre-processes the full dataset once before training begins, caching the tokenized form. This takes a few minutes upfront but makes each training batch very fast to assemble.

**`packing=False`**: sequence packing is disabled. Packing concatenates multiple short examples (separated by EOS tokens) into a single `max_seq_length` sequence to avoid wasting GPU compute on padding. It is more memory-efficient but introduces a subtler loss signal: the model sees multiple instruction-response pairs in one sequence, and the loss is computed across all of them. For instruction fine-tuning, this can blur the boundary between examples and slightly degrade performance. `packing=False` processes each example as its own sequence (padded to the batch's longest example), which gives a cleaner loss signal at the cost of some wasted compute on padding tokens.

---

### Running training

```python
    print("\nStarting training...")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
```

`trainer.train()` is the main training loop. Under the hood it:
1. Iterates over the dataset in mini-batches.
2. For each batch, calls `formatting_func` to get formatted strings, tokenizes them, and assembles the batch tensors.
3. Runs a forward pass through the model, computing the cross-entropy loss on assistant tokens only.
4. Computes gradients via the backward pass (only the LoRA adapter parameters receive gradients, because the base model is frozen).
5. Accumulates gradients for `gradient_accumulation_steps=4` mini-batches.
6. Clips gradients to `max_grad_norm=1.0`.
7. Updates the optimizer (paged AdamW 8-bit) and learning rate scheduler.
8. Logs the loss every 25 steps.
9. Saves a checkpoint every 500 steps.

`result` is a `TrainOutput` object containing `training_loss` (the final average loss over all steps) and other statistics.

`time.time()` records wall-clock time in seconds before and after training.

---

### Saving the adapter

```python
    adapter_dir = tc["final_adapter_dir"]
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTraining complete.")
    print(f"  Time:        {hours}h {minutes}m {seconds}s")
    print(f"  Final loss:  {result.training_loss:.4f}")
    print(f"  Adapter saved to: {adapter_dir}")
```

**`trainer.save_model(adapter_dir)`** — because the model is a PEFT-wrapped model, `save_model` is smart enough to save only the adapter weights (the LoRA `A` and `B` matrices), not the entire base model. The saved adapter is a small file (typically 40-80MB) compared to the full model (~6GB). It saves:
- `adapter_config.json`: the LoRA configuration (rank, alpha, target modules, etc.).
- `adapter_model.safetensors` (or `.bin`): the actual adapter weight tensors.

**`tokenizer.save_pretrained(adapter_dir)`** — saves the tokenizer alongside the adapter. This is important: when someone loads the adapter from HuggingFace Hub, they also need the tokenizer (specifically the `pad_token` and chat template settings). Saving both together ensures they are always distributed as a matching pair.

**`divmod(int(elapsed), 3600)`** — converts seconds to hours, minutes, seconds in a readable format. `divmod(7384, 3600)` gives `(2, 184)` → 2 hours and 184 seconds. Then `divmod(184, 60)` gives `(3, 4)` → 3 minutes and 4 seconds. Result: "2h 3m 4s".

**`result.training_loss`** — the average training loss over all steps. For a well-behaved fine-tuning run on Alpaca, expect this to end up around 1.0-1.5 for a 3B model.

---

## Things to know

**You must run `evaluate.py --stage before` before `train.py`.** The pipeline is designed as: measure baseline → train → measure improvement. If you run `train.py` first and then try to run `evaluate.py --stage before`, you would be measuring the base model without the adapter — which is technically correct for "before" evaluation — but conceptually you have already committed to a training direction. Running `--stage before` first is the correct scientific workflow.

**Training takes 2-3 hours on an RTX 4090. Plan accordingly.** On a free Colab T4, expect 8-12 hours. On CPU, it is not practical for a full epoch. The script does not support multi-GPU training out of the box (Accelerate would be needed for that).

**Checkpoints in `outputs/checkpoints/` are full checkpoint directories.** They include the adapter weights, the optimizer state, the scheduler state, and training metadata. They are used to resume training if it crashes. With `save_total_limit: 2`, only the two most recent checkpoints are kept, so the directory does not grow too large.

**`eval_dataset=None` means no in-training validation loss.** You will only see training loss in the logs. If training loss is decreasing steadily, that is a good sign. If it plateaus or spikes, investigate learning rate or data quality. A separate validation loss check would require either using `eval_dataset=splits["test"]` (which slows training) or running `evaluate.py` manually at interim checkpoints.

**The final adapter does not include the base model weights.** `outputs/final_adapter/` contains only the small LoRA matrices. To run inference, you must load the base model first and then load the adapter on top — exactly what `load_model_for_eval` does.

**`packing=False` means the effective training time is slightly longer** than with packing enabled. With `packing=True`, a batch of 4 sequences of average length 256 tokens would be packed into a single 2048-token sequence, fully utilizing the GPU's parallelism. With `packing=False`, that same batch wastes the tokens beyond 256 on padding. For Alpaca (short examples), this means ~70-80% GPU utilization instead of ~99%. The trade-off (cleaner gradient signal vs. speed) is judged worthwhile here.
