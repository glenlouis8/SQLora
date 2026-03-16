# 05 — `scripts/prepare_data.py`

## What this file is

`prepare_data.py` is a diagnostic/validation script that you run before committing to a full training run. It downloads the Alpaca dataset, prints three formatted sample rows so you can visually verify the chat template looks correct, and computes the token length distribution of the training split to warn you if your `max_seq_length` setting is too small. Think of it as a pre-flight checklist — it costs a few minutes but can save hours of wasted training time.

## How it connects

- **Called by:** nothing — it is a standalone script run by the user with `python scripts/prepare_data.py`.
- **Calls:** `src.data_utils.load_and_split`, `src.data_utils.format_alpaca_row`, `transformers.AutoTokenizer`.
- **Fits in the pipeline as:** step 1, run before anything else. It reads no saved artifacts and writes no output files — it only prints to stdout and exits.

---

## Full walkthrough

### Module docstring and argument definition

```python
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
```

The docstring serves as user documentation, explaining what to run and what to expect. The "dry-run" framing is important: this script never modifies any files or starts training — it is safe to run at any time.

---

### `main()` — setup

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
```

**`argparse`** is Python's standard library for command-line argument parsing. `--config` is optional — if not provided, it defaults to `"configs/train_config.yaml"`. This means `python scripts/prepare_data.py` works out of the box without any flags.

**`yaml.safe_load(f)`** parses the YAML config file into a nested Python dictionary. `safe_load` (as opposed to `yaml.load`) does not execute arbitrary Python constructors in the YAML file, which is a security best practice.

**Tokenizer loading here is intentionally simpler than `load_tokenizer` in `model_utils.py`** — it sets `pad_token` but not `padding_side`. Since this script only tokenizes text (no batched training), the padding side does not matter.

---

### Loading and splitting the dataset

```python
    print("Loading and splitting dataset...")
    splits = load_and_split(cfg)
    train_ds = splits["train"]
    test_ds = splits["test"]
    print(f"  Train: {len(train_ds):,} rows")
    print(f"  Eval:  {len(test_ds):,} rows\n")
```

This calls the same `load_and_split` function that `train.py` and `evaluate.py` use. The `,` format specifier in `f"{len(train_ds):,}"` adds comma thousands separators, making large numbers easier to read (e.g., `49,401` instead of `49401`).

Running this section confirms two things: the dataset downloaded successfully, and the split produces the expected sizes. If the dataset name in the config is wrong or there is no internet connection, this is where the error will surface — before anything expensive has happened.

---

### Previewing formatted samples

```python
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
```

**`for i in [0, 100, 500]`** — three rows are chosen at positions 0, 100, and 500. These are spread across the dataset to sample different types of examples (the dataset is not sorted by type, but different positions increase variety). The indices are hardcoded because this is a diagnostic preview, not a rigorous sample.

**`formatted[:800]`** — displays only the first 800 characters of each formatted example. Most Alpaca rows are under 800 characters, so this shows the full example. For very long examples, the `... [truncated for display]` suffix signals that more content exists.

**Why eyeball the formatting?** The chat template is model-specific and uses special tokens that are not human-readable. This preview lets you verify:
1. The system, user, and assistant roles are present and in the right order.
2. The instruction and input are combined correctly (two newlines between them).
3. The assistant's answer appears at the end and is followed by the end-of-sequence token.

If any of these look wrong, you catch it here rather than after a 2.5-hour training run.

---

### Token length distribution

```python
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
```

**`rng = np.random.default_rng(42)`** — a fixed seed so the 2000-row sample is always the same, making results reproducible across runs.

**`size=min(2000, len(train_ds))`** — graceful handling for datasets smaller than 2000 rows (uncommon here but good practice).

**`tokenizer(text, truncation=False)`** — intentionally does not truncate. We want to measure the actual token lengths without truncation, because the point is to check whether our `max_seq_length` is large enough. If we applied truncation here, every length would be capped at 2048 and we would never see the true length of long examples.

**Percentile reporting (p50, p90, p99):**

- **p50 (median):** Half the training examples are shorter than this. For Alpaca, expect ~180-250 tokens. This tells you the "typical" example size.
- **p90:** 90% of examples are shorter than this. Expect ~400-600 tokens. This tells you what most examples look like, excluding the longest 10%.
- **p99:** 99% of examples are shorter than this. Expect ~800-1200 tokens. This is the critical number for truncation risk.

**The warning logic:** if p99 exceeds `max_seq_length` (2048 here), then approximately 1% of training examples will be silently truncated when `SFTTrainer` tokenizes them. "Silently" means no error is raised — the text is simply cut off. For Alpaca with `max_seq_length=2048`, truncation is almost never an issue (most examples are well under 1000 tokens), so you should see the `OK` message. For other datasets with longer examples (e.g., code, long documents), this warning is critical.

**Why sample 2000 rows instead of the full ~49,000?** Tokenizing 49,000 rows takes several minutes; 2000 rows takes seconds. A sample of 2000 is statistically representative enough for estimating percentiles — the error on a percentile estimate is proportional to `1/sqrt(n)`, so even 500 rows would give a reasonable estimate.

---

### Entry point

```python
if __name__ == "__main__":
    main()
```

Standard Python convention: the `main()` function is called only when this script is run directly, not when it is imported as a module. This means `import prepare_data` in a test or another script would not automatically run `main()`.

---

## Things to know

**This script does not save any output.** It prints everything to stdout. If you want to keep a record of the distribution, redirect output: `python scripts/prepare_data.py > data_check.txt`.

**The 2000-row sample uses a hardcoded seed of 42.** This is separate from `cfg["data"]["seed"]` and is hardcoded directly in the script. This was probably an oversight — ideally it would use `cfg["data"]["seed"]` for full consistency. In practice, for distribution estimation, the exact sample does not matter much.

**Changing `max_seq_length` in the config also affects training.** The check here is informational — it does not modify the config. If you see a p99 warning, you must manually edit `train_config.yaml` and re-run this script to confirm the warning is gone.

**The three preview rows (0, 100, 500) are from the training split, not the test split.** After `load_and_split`, the split is shuffled and indexed. Row 0 of `train_ds` is not the same as row 0 of the original `tatsu-lab/alpaca` dataset.

**This is the only script that imports `AutoTokenizer` directly** rather than using `load_tokenizer` from `model_utils.py`. This is a minor inconsistency: it sets `pad_token` but not `padding_side`. For this script's purposes (no batching), this does not matter. But if a future developer refactors to add batching in this script, they would need to remember to add `padding_side = "right"`.
