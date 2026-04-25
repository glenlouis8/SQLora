# llama-3.2-3b-sql-qlora

Fine-tuning `meta-llama/Llama-3.2-3B-Instruct` on `b-mc2/sql-create-context` (~78k examples) using QLoRA — 4-bit NF4 quantization + LoRA adapters on all 7 attention/MLP projection layers.

## Results

Evaluated on a held-out 5% split (2,601 examples, seed=42).

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Perplexity | 25.84 | 4.82 | -81.3% |
| ROUGE-L | 0.259 | 0.353 | +36.3% |

Evaluated on 2026-03-17. ROUGE-L computed over 200 sampled examples with greedy decoding.

## Setup

```bash
# Install dependencies
uv pip install -r requirements.txt

# Copy and fill in your HuggingFace token
cp .env.example .env

# Accept the Llama 3.2 license at huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

Update `hub.repo_id` in `configs/train_config.yaml` with your HuggingFace username before pushing.

## Usage

```bash
python scripts/prepare_data.py              # validate dataset + token length stats
python scripts/evaluate.py --stage before   # baseline metrics (perplexity + ROUGE-L)
python scripts/train.py                     # fine-tune (~2.5hrs on RTX 4090)
python scripts/evaluate.py --stage after    # post-training metrics
python scripts/evaluate.py --compare        # print before/after comparison table
python scripts/push_to_hub.py               # push adapter + model card to HF Hub
```

All scripts accept `--config configs/train_config.yaml` (default).

## Architecture

| Component | Detail |
|-----------|--------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Quantization | 4-bit NF4, bfloat16 compute, double quant |
| LoRA rank | r=16, alpha=32 (scale=2.0) |
| LoRA targets | q/k/v/o/gate/up/down_proj (all 7 layers) |
| Trainable params | ~20M |
| Dataset | `b-mc2/sql-create-context` — ~78k Text-to-SQL examples |
| Eval metrics | Perplexity (full eval split) + ROUGE-L (200 samples) |

## Config

All hyperparameters live in [`configs/train_config.yaml`](configs/train_config.yaml) — model, LoRA, data, training, and hub settings in one place.

Key training settings:
- Effective batch size: 16 (4 per device × 4 gradient accumulation steps)
- Optimizer: paged AdamW 8-bit
- LR: 2e-4 with cosine schedule + 3% warmup
- Gradient checkpointing enabled (~40% VRAM reduction)

## Project Structure

```
configs/train_config.yaml   ← all hyperparameters
src/
  data_utils.py             ← dataset loading, train/eval split, SQL prompt formatter
  model_utils.py            ← BnB 4-bit config, LoRA config, model loading
  eval_utils.py             ← perplexity + ROUGE-L evaluation
scripts/
  prepare_data.py           ← dry-run: preview samples + token stats
  train.py                  ← SFTTrainer training loop
  evaluate.py               ← before/after eval → writes results/*.json
  push_to_hub.py            ← push adapter + auto-generated model card
results/                    ← evaluation JSON files (source of truth for model card)
outputs/                    ← git-ignored checkpoints and final adapter
```

## Requirements

- Python 3.10+
- CUDA GPU with 16GB+ VRAM recommended (tested on RTX 4090)
- CPU fallback available (no 4-bit quant, significantly slower)
