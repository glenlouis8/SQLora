"""
prepare_data.py
───────────────
Dry-run script: validate that dataset loading and formatting work correctly
before committing to a long training run.

What it does:
  1. Downloads tatsu-lab/alpaca
  2. Prints 3 formatted sample rows (so you can eyeball the chat template)
  3. Reports token length distribution (p50 / p90 / p99)
  4. Warns if p99 exceeds max_seq_length (would cause silent truncation)

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --config configs/train_config.yaml
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
from transformers import AutoTokenizer

from src.data_utils import load_and_split, format_alpaca_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading and splitting dataset...")
    splits = load_and_split(cfg)
    train_ds = splits["train"]
    test_ds = splits["test"]
    print(f"  Train: {len(train_ds):,} rows")
    print(f"  Eval:  {len(test_ds):,} rows\n")

    # ── Preview 3 formatted samples ────────────────────────────────────────
    print("=" * 70)
    print("SAMPLE FORMATTED ROWS")
    print("=" * 70)
    for i in [0, 100, 500]:
        row = train_ds[i]
        formatted = format_alpaca_row(row, tokenizer)
        print(f"\n--- Row {i} ---")
        print(formatted[:800])
        if len(formatted) > 800:
            print("... [truncated for display]")

    # ── Token length distribution ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOKEN LENGTH DISTRIBUTION (train split, sampled 2000 rows)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(train_ds), size=min(2000, len(train_ds)), replace=False)
    sample = train_ds.select(sample_indices.tolist())

    lengths = []
    for row in sample:
        text = format_alpaca_row(row, tokenizer)
        ids = tokenizer(text, truncation=False)["input_ids"]
        lengths.append(len(ids))

    lengths = np.array(lengths)
    p50 = int(np.percentile(lengths, 50))
    p90 = int(np.percentile(lengths, 90))
    p99 = int(np.percentile(lengths, 99))
    max_len = cfg["model"]["max_seq_length"]

    print(f"  p50: {p50} tokens")
    print(f"  p90: {p90} tokens")
    print(f"  p99: {p99} tokens")
    print(f"  max_seq_length (config): {max_len} tokens")

    if p99 > max_len:
        print(
            f"\n  [WARNING] p99 ({p99}) exceeds max_seq_length ({max_len}).\n"
            f"  ~1% of rows will be silently truncated during training.\n"
            f"  Consider increasing max_seq_length in configs/train_config.yaml."
        )
    else:
        print(f"\n  OK — p99 fits within max_seq_length.")


if __name__ == "__main__":
    main()
