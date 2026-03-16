"""
evaluate.py
───────────
Before/after evaluation runner. Computes Perplexity and ROUGE-L and writes
results to JSON files that become the source of truth for the README.

Usage:
  # Evaluate the base model (run BEFORE training)
  python scripts/evaluate.py --stage before

  # Evaluate the fine-tuned model (run AFTER training)
  python scripts/evaluate.py --stage after

  # Print the before/after comparison table
  python scripts/evaluate.py --compare
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from datetime import date

from src.data_utils import load_and_split
from src.model_utils import load_tokenizer, load_model_for_eval
from src.eval_utils import compute_perplexity, compute_rouge_l

RESULTS_DIR = "results"
BEFORE_FILE = os.path.join(RESULTS_DIR, "before_finetune.json")
AFTER_FILE = os.path.join(RESULTS_DIR, "after_finetune.json")


def run_evaluation(stage, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tokenizer = load_tokenizer(cfg)

    adapter_path = None
    if stage == "after":
        adapter_path = cfg["training"]["final_adapter_dir"]
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at {adapter_path}. Run train.py first."
            )

    print(f"Loading model for '{stage}' evaluation...")
    model = load_model_for_eval(cfg, adapter_path=adapter_path)

    splits = load_and_split(cfg)
    eval_ds = splits["test"]
    print(f"Eval split: {len(eval_ds):,} rows")

    print("Computing perplexity (full eval split)...")
    ppl = compute_perplexity(model, tokenizer, eval_ds, cfg)
    print(f"  Perplexity: {ppl}")

    print(f"Computing ROUGE-L ({cfg['data']['eval_sample_size']} samples)...")
    rouge_l = compute_rouge_l(model, tokenizer, eval_ds, cfg)
    print(f"  ROUGE-L:    {rouge_l}")

    result = {
        "stage": stage,
        "model": cfg["model"]["name"],
        "adapter": adapter_path,
        "eval_split_size": len(eval_ds),
        "perplexity": ppl,
        "rouge_l": rouge_l,
        "rouge_sample_size": cfg["data"]["eval_sample_size"],
        "eval_date": str(date.today()),
    }

    out_file = BEFORE_FILE if stage == "before" else AFTER_FILE
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_file}")
    return result


def print_comparison():
    if not os.path.exists(BEFORE_FILE) or not os.path.exists(AFTER_FILE):
        missing = []
        if not os.path.exists(BEFORE_FILE):
            missing.append(BEFORE_FILE)
        if not os.path.exists(AFTER_FILE):
            missing.append(AFTER_FILE)
        raise FileNotFoundError(
            f"Missing result files: {', '.join(missing)}\n"
            "Run --stage before and --stage after first."
        )

    with open(BEFORE_FILE) as f:
        before = json.load(f)
    with open(AFTER_FILE) as f:
        after = json.load(f)

    def delta_str(b, a, lower_is_better=True):
        pct = (a - b) / b * 100
        sign = "+" if pct > 0 else ""
        better = (pct < 0) == lower_is_better
        indicator = "(better)" if better else "(worse)"
        return f"{sign}{pct:.1f}% {indicator}"

    ppl_delta = delta_str(before["perplexity"], after["perplexity"], lower_is_better=True)
    rouge_delta = delta_str(before["rouge_l"], after["rouge_l"], lower_is_better=False)

    print("\n" + "=" * 60)
    print("BEFORE / AFTER FINE-TUNING COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Before':>10} {'After':>10}  {'Delta'}")
    print("-" * 60)
    print(f"{'Perplexity (↓)':<20} {before['perplexity']:>10.4f} {after['perplexity']:>10.4f}  {ppl_delta}")
    print(f"{'ROUGE-L (↑)':<20} {before['rouge_l']:>10.4f} {after['rouge_l']:>10.4f}  {rouge_delta}")
    print("=" * 60)
    print(f"\nEval split size: {before['eval_split_size']:,} rows")
    print(f"ROUGE-L sample:  {before['rouge_sample_size']} rows")
    print(f"Before date: {before['eval_date']}  |  After date: {after['eval_date']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stage",
        choices=["before", "after"],
        help="Which stage to evaluate",
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="Print before/after comparison table (requires both result files)",
    )
    args = parser.parse_args()

    if args.compare:
        print_comparison()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_evaluation(args.stage, cfg)


if __name__ == "__main__":
    main()
