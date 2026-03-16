# 08 — `scripts/push_to_hub.py`

## What this file is

`push_to_hub.py` is the final step in the pipeline. It authenticates with HuggingFace Hub, verifies that all prerequisites exist (the trained adapter, the before/after evaluation results, and an API token), then publishes three things: the merged LoRA adapter weights, the tokenizer, and an auto-generated `README.md` that includes the real evaluation numbers. The README is not written by hand — it is programmatically constructed using the evaluation JSON files so the numbers are always accurate and up-to-date.

## How it connects

- **Called by:** the user via `python scripts/push_to_hub.py`.
- **Calls:** `src.model_utils.load_model_for_eval`, `peft.PeftModel.from_pretrained`, `peft.PeftModel.push_to_hub`, `transformers.AutoTokenizer.push_to_hub`, `huggingface_hub.HfApi.upload_file`, `huggingface_hub.login`, `dotenv.load_dotenv`.
- **Reads:** `configs/train_config.yaml`, `results/before_finetune.json`, `results/after_finetune.json`, `outputs/final_adapter/`, `HF_TOKEN` from environment or `.env` file.
- **Writes:** to HuggingFace Hub (remote only — no local files other than a temporary README).
- **Fits in the pipeline as:** step 5, the last step. Requires that all prior steps have completed successfully.

---

## Full walkthrough

### Imports

```python
import argparse
import json
import os
import tempfile
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer
from peft import PeftModel
from src.model_utils import load_model_for_eval
```

- **`tempfile`**: Python's standard library for creating temporary files. Used to write the README to a temp file before uploading it, then delete the temp file afterward.
- **`dotenv`** (`python-dotenv`): loads key-value pairs from a `.env` file into the process's environment variables. This is the standard pattern for storing secrets (like API tokens) outside of source code.
- **`HfApi`**: HuggingFace Hub's API client. Used here specifically for `upload_file()` to push the README markdown.
- **`login`**: authenticates the current process with HuggingFace Hub using the provided token. After calling this, subsequent `push_to_hub()` calls succeed without needing to pass the token explicitly.
- **`PeftModel`**: needed here (in addition to `model_utils`) because `push_to_hub.py` wraps the base model with the adapter using `PeftModel.from_pretrained()` then calls `peft_model.push_to_hub()`.

---

### `build_readme(cfg, before, after)`

This is the most interesting function in the file. It takes the config dict and the two result dicts, computes delta values, and returns a multi-section markdown string.

```python
def build_readme(cfg, before, after):
    """Render the README string with real evaluation numbers."""
    model_name = cfg["model"]["name"]
    repo_id = cfg["hub"]["repo_id"]
    dataset_name = cfg["data"]["dataset_name"]

    ppl_before = before["perplexity"]
    ppl_after = after["perplexity"]
    ppl_delta = (ppl_after - ppl_before) / ppl_before * 100

    rouge_before = before["rouge_l"]
    rouge_after = after["rouge_l"]
    rouge_delta = (rouge_after - rouge_before) / rouge_before * 100
```

**Extracting values from the result JSONs:** `before` and `after` are Python dicts loaded from `results/before_finetune.json` and `results/after_finetune.json`. The fields match exactly what `evaluate.py` saved.

**`ppl_delta = (ppl_after - ppl_before) / ppl_before * 100`** — percentage change in perplexity. If perplexity dropped from 8.23 to 6.81, this is `(6.81 - 8.23) / 8.23 * 100 = -17.3`. A negative number means improvement (lower perplexity is better).

**`rouge_delta`** — same calculation for ROUGE-L. If ROUGE-L improved from 0.234 to 0.316, this is `(0.316 - 0.234) / 0.234 * 100 = +35.0`. A positive number means improvement.

---

### The README template

```python
    return f"""# {repo_id.split("/")[-1]}

Fine-tuned [{model_name}](https://huggingface.co/{model_name}) on [{dataset_name}](...) using QLoRA for general instruction-following.

## Results

| Metric | Base Model | Fine-tuned | Delta |
|--------|-----------|------------|-------|
| Perplexity (eval split, ↓) | {ppl_before:.4f} | {ppl_after:.4f} | {ppl_delta:+.1f}% |
| ROUGE-L (200 samples, ↑) | {rouge_before:.4f} | {rouge_after:.4f} | {rouge_delta:+.1f}% |
...
```

This is a Python f-string that spans the entire function's return value. Notable formatting choices:

**`repo_id.split("/")[-1]`** — extracts just the model name from the repo ID. For `"yourname/llama-3.2-3b-alpaca-qlora"`, this gives `"llama-3.2-3b-alpaca-qlora"` as the top-level heading.

**`{ppl_delta:+.1f}%`** — the `+` format specifier explicitly shows the sign for both positive and negative numbers. This gives `"-17.3%"` for improvements (perplexity drop) and `"+35.0%"` for improvements (ROUGE-L increase). Without `+`, Python would only show `-` for negative numbers and nothing for positive numbers, which would be confusing in the context of a comparison table.

**`{cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]}`** — the effective batch size is computed inline in the template. This is `4 × 4 = 16`. If the config values change, the README will automatically reflect the correct effective batch size.

**The Quickstart section** includes a full code example showing how to load and use the pushed model. The double braces `{{` and `}}` in the f-string are escaped curly braces — they produce literal `{` and `}` in the output. This is necessary because the code example itself contains Python dict literals with curly braces (e.g., `{"role": "system", ...}`).

**The Methodology section** inlines the actual perplexity and ROUGE-L numbers into the explanation text (`{ppl_before:.1f}`, `{ppl_after:.1f}`, `{rouge_before:.3f}`, `{rouge_after:.3f}`). This makes the README self-explanatory: the numbers in the "What do the numbers mean?" paragraph automatically match the numbers in the results table.

**The Limitations section** is honest about the dataset's origins (GPT-3.5 generated) and the absence of safety alignment. This is important model card practice.

---

### `main()` — prerequisites check

```python
def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Verify prerequisites ────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set. Copy .env.example to .env and fill in your token."
        )

    adapter_path = cfg["training"]["final_adapter_dir"]
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. Run train.py first."
        )

    for path in [BEFORE_FILE, AFTER_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run evaluate.py --stage before and --stage after first."
            )
```

**`load_dotenv()`** reads the `.env` file in the current working directory and loads its contents into `os.environ`. A `.env` file looks like:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
After `load_dotenv()`, `os.environ.get("HF_TOKEN")` returns the token value. This keeps secrets out of source code and out of version control (`.env` should be in `.gitignore`).

**The three prerequisite checks fail fast** with clear, actionable error messages:
1. No HF_TOKEN → tell the user to set it in `.env`.
2. No adapter → tell the user to run `train.py`.
3. Missing result files → tell the user which specific file is missing and which command produces it.

These checks happen before loading any model, so you get immediate feedback if something is wrong, rather than waiting several minutes for a model to load before hitting the error.

---

### Authentication and loading result files

```python
    login(token=hf_token)

    with open(BEFORE_FILE) as f:
        before = json.load(f)
    with open(AFTER_FILE) as f:
        after = json.load(f)
```

**`login(token=hf_token)`** authenticates with HuggingFace Hub. Subsequent library calls (`push_to_hub`, `upload_file`) automatically use this token. The token needs `write` permission on the target repository.

---

### Pushing adapter weights

```python
    repo_id = cfg["hub"]["repo_id"]
    print(f"Pushing adapter to: {repo_id}")

    # ── Push adapter weights ────────────────────────────────────────────────
    model = load_model_for_eval(cfg, adapter_path=None)
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    peft_model.push_to_hub(repo_id, private=cfg["hub"]["private"])
```

**Why load the model again here?** The adapter directory (`outputs/final_adapter/`) only contains the LoRA weight matrices, not the base model. To push the adapter to Hub in the PEFT format (which is what allows others to use it with `PeftModel.from_pretrained`), PEFT needs the base model to attach the adapter to. So we load the base model fresh, then wrap it with the adapter.

**`load_model_for_eval(cfg, adapter_path=None)`** loads the base model without merging any adapter. Unlike `evaluate.py`, here we do not want the merged model — we want the PEFT-wrapped model so PEFT's `push_to_hub` serializes just the adapter weights.

**`PeftModel.from_pretrained(model, adapter_path)`** loads the adapter on top of the base model, creating a `PeftModel` wrapper. This is different from `merge_and_unload()` used in evaluation — here we keep the PEFT structure intact so the Hub repository contains just the small adapter files.

**`peft_model.push_to_hub(repo_id, private=cfg["hub"]["private"])`** uploads the adapter weights to HuggingFace Hub. The repository is created automatically if it does not exist. The uploaded files are `adapter_config.json` and `adapter_model.safetensors` — just 40-80MB total.

---

### Pushing the tokenizer

```python
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.push_to_hub(repo_id)
```

The tokenizer is loaded fresh from the base model (not from `outputs/final_adapter/`). In practice the tokenizer in `outputs/final_adapter/` is identical to the base model's tokenizer (since we did not modify the vocabulary or template), so this distinction does not matter. The tokenizer is pushed so users can do `AutoTokenizer.from_pretrained(repo_id)` instead of needing to separately download the base model's tokenizer.

---

### Pushing the README

```python
    readme_content = build_readme(cfg, before, after)
    api = HfApi()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(readme_content)
        tmp_path = tmp.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    os.unlink(tmp_path)
```

**`build_readme(cfg, before, after)`** renders the full README string with all numbers injected.

**`tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)`** creates a temporary file on disk with a `.md` extension. `delete=False` means the file persists after the `with` block closes — necessary because `upload_file` needs to read it afterward. The `with` block opens and writes the README content to the temp file, then closes the file handle.

**Why a temp file?** `HfApi.upload_file` accepts a file path (`path_or_fileobj`). An alternative would be `api.upload_file(path_or_fileobj=readme_content.encode(), ...)` using an in-memory bytes object — but using a file path is more explicit and avoids encoding concerns.

**`api.upload_file(path_in_repo="README.md", ...)`** uploads the local temp file to `README.md` in the HuggingFace repository. This creates or overwrites the README. `repo_type="model"` specifies that this is a model repository (as opposed to a dataset or space repository).

**`os.unlink(tmp_path)`** deletes the temporary file from disk. `unlink` is the Unix system call for deleting a file (identical to `os.remove()`). Cleanup after temp files is good practice — without it, temp files accumulate in the system's temp directory.

---

### Final output

```python
    print(f"\nDone! Model pushed to:")
    print(f"  https://huggingface.co/{repo_id}")
```

Prints the URL where the model can now be found and used.

---

## Things to know

**You must replace `YOUR_HF_USERNAME` in the config before running this.** The config default is `"YOUR_HF_USERNAME/llama-3.2-3b-alpaca-qlora"`. If you push with this literal string, HuggingFace will try to create a repository under a username that almost certainly does not exist.

**The HF token requires `write` permission.** Read-only tokens cannot create or modify repositories. Generate a token at `https://huggingface.co/settings/tokens` with `write` scope.

**The token should be in `.env`, not hardcoded.** The `.env.example` file (mentioned in the error message) should be a template file showing the expected structure but with a placeholder instead of a real token. Never commit a real token to version control.

**Pushing does not delete previous versions.** HuggingFace Hub repositories are git-backed. Each push to the same `repo_id` adds a new commit to the repository's history. If you re-train and re-push, the old adapter weights are preserved in git history but the latest commit shows the new files.

**The model pushed is the PEFT adapter format, not the merged model.** This means users loading it via `AutoModelForCausalLM.from_pretrained(repo_id)` will get an error. The correct loading pattern (shown in the Quickstart section of the generated README) is to load the base model first, then `PeftModel.from_pretrained(base_model, repo_id)`. This is the standard pattern for LoRA adapters on HuggingFace.

**`load_model_for_eval` is called with `adapter_path=None` here**, then the adapter is re-loaded via `PeftModel.from_pretrained`. This is slightly redundant — `load_model_for_eval(cfg, adapter_path=adapter_path)` with `merge_and_unload()` would also work but would push a merged (non-PEFT) model. The current approach correctly preserves the PEFT format so others can apply the adapter selectively.

**The script loads the full model into memory.** This requires the same GPU VRAM as running `evaluate.py`. If you do not have GPU access for pushing, you would need to modify the script to use CPU mode — though pushing itself is a CPU-bound network operation, the model loading step before it requires VRAM.
