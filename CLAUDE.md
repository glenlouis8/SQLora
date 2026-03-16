# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Validate dataset formatting and token lengths (no GPU needed)
python scripts/prepare_data.py

# Evaluate base model BEFORE training
python scripts/evaluate.py --stage before

# Train (1 epoch, ~2.5hrs on RTX 4090)
python scripts/train.py

# Evaluate fine-tuned model AFTER training
python scripts/evaluate.py --stage after

# Print before/after comparison table
python scripts/evaluate.py --compare

# Push adapter + auto-generated README to HuggingFace Hub
python scripts/push_to_hub.py
```

All scripts accept `--config configs/train_config.yaml` (default).

## Architecture

```
configs/train_config.yaml   ← single source of truth for ALL hyperparameters
src/
  data_utils.py             ← dataset loading, train/eval split (seed=42), chat template formatter
  model_utils.py            ← BnB 4-bit config, LoRA config, model load helpers
  eval_utils.py             ← perplexity (full eval split) + ROUGE-L (200 samples)
scripts/
  prepare_data.py           ← dry-run: preview samples + token length stats
  train.py                  ← SFTTrainer training loop
  evaluate.py               ← before/after eval → writes results/*.json
  push_to_hub.py            ← pushes adapter + injects real numbers into README
results/                    ← committed JSON files (source of truth for README numbers)
outputs/                    ← git-ignored checkpoints and final adapter
```

**Critical invariant:** `data_utils.load_and_split()` must always be called with `seed=42` (set in config). Both `train.py` and `evaluate.py` rely on this to get the identical held-out split. Never change the seed after initial training.

**Model:** `meta-llama/Llama-3.2-3B-Instruct` with 4-bit NF4 QLoRA.
**Adapter:** LoRA r=16 on all 7 projection layers (q/k/v/o/gate/up/down_proj), ~20M trainable params.
**Training lib:** TRL `SFTTrainer` with `packing=False` and `eval_dataset=None` (eval handled separately).

## Environment

Set `HF_TOKEN` in `.env` (copy from `.env.example`) before running any script that downloads the model or pushes to the Hub.

Before first run, accept the Llama 3.2 license at `huggingface.co/meta-llama/Llama-3.2-3B-Instruct` — otherwise model downloads will 401.

Update `hub.repo_id` in `configs/train_config.yaml` to your HuggingFace username before running `push_to_hub.py`.

On machines without CUDA, the model falls back to CPU float32 with a clear warning (4-bit quantization requires CUDA).
