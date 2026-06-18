"""
exec_eval.py
────────────
Execution accuracy evaluation. No GPU needed — runs entirely on Mac.
Calls Modal API for SQL generation, executes against in-memory SQLite.

Usage:
  python scripts/exec_eval.py --modal-url https://YOUR-MODAL-URL.modal.run
  python scripts/exec_eval.py --modal-url https://... --n-samples 50
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from datetime import date

from src.data_utils import load_and_split
from src.exec_eval_utils import run_exec_eval

RESULTS_DIR = "results"
EXEC_EVAL_FILE = os.path.join(RESULTS_DIR, "exec_eval.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal-url", required=True, help="Modal FastAPI endpoint URL")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    splits = load_and_split(cfg)
    eval_ds = splits["test"]
    print(f"Test split: {len(eval_ds):,} rows — sampling {args.n_samples}\n")

    results = run_exec_eval(
        modal_url=args.modal_url,
        dataset=eval_ds,
        n_samples=args.n_samples,
        seed=cfg["data"]["seed"],
    )
    results["eval_date"] = str(date.today())
    results["model"] = cfg["model"]["name"]
    results["adapter"] = cfg["hub"]["repo_id"]
    results["modal_url"] = args.modal_url

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(EXEC_EVAL_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 55}")
    print("EXECUTION ACCURACY RESULTS")
    print(f"{'=' * 55}")
    print(f"  Execution Accuracy : {results['execution_accuracy']:.1%}  ({results['correct']}/{results['total']})")
    print(f"  Skipped            : {results['skipped']}  (ref SQL error or no schema)")
    print(f"{'=' * 55}")
    print(f"\nSaved → {EXEC_EVAL_FILE}")


if __name__ == "__main__":
    main()
