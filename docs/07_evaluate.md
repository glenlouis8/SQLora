# 07 — `scripts/evaluate.py`

## What this file is

`evaluate.py` is the evaluation script that measures model quality before and after fine-tuning using two metrics: perplexity and ROUGE-L. It is designed to be run twice — once before training to establish a baseline and once after training to measure improvement. The results are saved as JSON files that become the authoritative source of truth for the published model card. It can also print a formatted comparison table showing the before/after delta.

## How it connects

- **Called by:** the user via `python scripts/evaluate.py --stage before` (before training) and `python scripts/evaluate.py --stage after` (after training), and optionally `python scripts/evaluate.py --compare`.
- **Calls:** `src.data_utils.load_and_split`, `src.model_utils.load_tokenizer`, `src.model_utils.load_model_for_eval`, `src.eval_utils.compute_perplexity`, `src.eval_utils.compute_rouge_l`.
- **Reads:** `configs/train_config.yaml`, and `outputs/final_adapter/` when `--stage after` (requires that `train.py` has completed).
- **Writes:** `results/before_finetune.json` (when `--stage before`) and `results/after_finetune.json` (when `--stage after`).
- **Fits in the pipeline as:** steps 2 and 4 — before and after `train.py`.

---

## Full walkthrough

### Constants and file paths

```python
RESULTS_DIR = "results"
BEFORE_FILE = os.path.join(RESULTS_DIR, "before_finetune.json")
AFTER_FILE = os.path.join(RESULTS_DIR, "after_finetune.json")
```

These module-level constants define where results are stored. Using `os.path.join` (rather than `"results/before_finetune.json"`) ensures the paths work correctly on both Unix/Mac (forward slashes) and Windows (backslashes).

The result files at these paths are later read by `push_to_hub.py`. Having module-level constants means both `evaluate.py` and `push_to_hub.py` refer to the same string values — though they are duplicated (both scripts define `BEFORE_FILE` and `AFTER_FILE` independently). If you change the path in one script, you must also change it in the other.

---

### `run_evaluation(stage, cfg)`

```python
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
```

**`os.makedirs(RESULTS_DIR, exist_ok=True)`** — creates the `results/` directory if it does not exist. `exist_ok=True` means no error is raised if the directory already exists (idempotent). Without this, trying to write to `results/before_finetune.json` before the directory exists would raise `FileNotFoundError`.

**`adapter_path = None`** for `--stage before`: `load_model_for_eval` with `adapter_path=None` loads the raw base model with no LoRA adapter. This is exactly what we want for the baseline measurement.

**`adapter_path = cfg["training"]["final_adapter_dir"]`** for `--stage after`: tells `load_model_for_eval` to load the base model and then merge the saved LoRA adapter into it.

**The `FileNotFoundError` check** is a user-friendly guard: if someone accidentally runs `--stage after` before training, they get a clear error message with the exact remedy, rather than a confusing `PeftModel.from_pretrained` exception deep in the HuggingFace stack.

---

### Running the evaluation

```python
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
```

**`load_model_for_eval`** is called before `load_and_split` intentionally — there is no dependency between them, but if the model fails to load (e.g., VRAM insufficient), it is better to fail early before downloading the dataset.

**`splits["test"]`** is the held-out 5% of Alpaca — the same rows in both `--stage before` and `--stage after` runs, because `load_and_split` always uses the same seed.

**The order of metric computation** (perplexity first, ROUGE-L second) reflects their time cost. Perplexity is fast (~3-5 minutes for 2,600 rows on GPU) because it only requires forward passes. ROUGE-L is slow (~20-30 minutes for 200 rows) because it requires generating text token by token.

---

### Saving results

```python
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
```

The result dict captures everything needed to reproduce and interpret the evaluation:
- **`stage`**: which run this was (`"before"` or `"after"`).
- **`model`**: the base model name (for documentation).
- **`adapter`**: `null` for before (JSON serializes Python `None` as `null`), or the adapter path for after.
- **`eval_split_size`**: number of rows in the test split (for documentation).
- **`perplexity`** and **`rouge_l`**: the actual metric values.
- **`rouge_sample_size`**: how many rows ROUGE-L was computed on.
- **`eval_date`**: today's date (e.g., `"2026-03-15"`), recorded so the README can show when the evaluation was run.

**`json.dump(result, f, indent=2)`** writes the dict as pretty-printed JSON. `indent=2` adds 2-space indentation, making the file human-readable. Without `indent`, the entire JSON would be on one line.

**`out_file = BEFORE_FILE if stage == "before" else AFTER_FILE`** — the output file is selected based on the stage. Running `--stage before` overwrites `before_finetune.json`; running `--stage after` overwrites `after_finetune.json`. Overwrites are intentional — if you re-run evaluation (e.g., after updating the model), you want the latest results.

---

### `print_comparison()`

```python
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
```

**The file existence check** lists all missing files explicitly (not just the first one), so the user gets a single error message telling them everything they need to do.

**`delta_str(b, a, lower_is_better)`** is a small inline helper function:
- `pct = (a - b) / b * 100` computes the percentage change from before (`b`) to after (`a`). A negative value means the metric decreased; positive means it increased.
- `sign = "+" if pct > 0 else ""` adds an explicit `+` prefix for positive changes (Python does not add `+` by default for positive numbers in format strings).
- `better = (pct < 0) == lower_is_better` — this is a boolean expression that evaluates to `True` when the direction of change matches the direction of improvement. For perplexity, lower is better, so a decrease (`pct < 0`) is better. For ROUGE-L, lower is worse, so a decrease is worse. The expression `(pct < 0) == lower_is_better` is an elegant two-case check: it is `True` when "decreased and lower is better" or when "increased and higher is better."

**The print formatting** uses Python's f-string alignment syntax:
- `:<20` means left-align in a 20-character-wide field.
- `:>10.4f` means right-align in a 10-character field with 4 decimal places.

This produces a nicely aligned table like:
```
============================================================
BEFORE / AFTER FINE-TUNING COMPARISON
============================================================
Metric               Before          After  Delta
------------------------------------------------------------
Perplexity (↓)        8.2341       6.8123  -17.2% (better)
ROUGE-L (↑)           0.2341       0.3156  +34.8% (better)
============================================================
```

---

### `main()` — argument parsing and dispatch

```python
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
```

**`add_mutually_exclusive_group(required=True)`** ensures that exactly one of `--stage` or `--compare` must be provided. If neither is given, argparse prints a usage error. If both are given, argparse also prints an error. This prevents accidental runs with no effect.

**`choices=["before", "after"]`** restricts `--stage` to exactly these two strings. Any other value triggers a clear argparse error.

**`action="store_true"` for `--compare`** means: if `--compare` is present in the command line arguments, set `args.compare = True`; if absent, set it to `False` by default.

**`if args.compare: print_comparison(); return`** — the comparison mode does not need to load the config or the model. It only reads the JSON files. So we dispatch to `print_comparison()` and return early, skipping the `with open(args.config)` block.

---

## Things to know

**Re-running `--stage before` after training still gives the correct baseline.** The `--stage before` mode loads `adapter_path=None`, so it always evaluates the raw base model — not the fine-tuned one. You could technically run `--stage before` at any time, even after training completes, and get a valid baseline. The only requirement is that the adapter directory does not somehow corrupt the base model loading, which it cannot since the base model is loaded fresh from disk each time.

**The `--compare` flag does not re-run evaluation.** It only reads the JSON files and prints a formatted table. This means you can run `--compare` quickly at any time after both stage files exist, without loading any model.

**Both evaluations load the full model from scratch.** If you run `--stage before` immediately followed by `--stage after` in a loop or shell script, each call loads a new model instance. There is no in-process model caching between calls. For memory efficiency, always let one evaluation complete and the process exit before starting the next.

**The result JSON files do not contain the actual generated text.** Only the aggregate scores are saved. If you want to audit which specific examples the model got right or wrong, you would need to add logging inside `compute_rouge_l`.

**The `eval_date` in the JSON uses `date.today()`, not `datetime.now()`.** This means it records only the date (e.g., `"2026-03-15"`), not the time. If you run before and after evaluation on the same day, both will show the same date. The time of evaluation is not recorded.

**Path constants are duplicated between `evaluate.py` and `push_to_hub.py`.** Both scripts define `BEFORE_FILE = "results/before_finetune.json"` and `AFTER_FILE = "results/after_finetune.json"` independently. In a more mature codebase, these would be defined in a shared constants module. If you change the results directory structure, remember to update both files.
