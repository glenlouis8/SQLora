"""
push_to_hub.py
──────────────
Push the fine-tuned LoRA adapter to HuggingFace Hub and auto-generate
a README with real evaluation numbers injected from results/*.json.

Usage:
  python scripts/push_to_hub.py
  python scripts/push_to_hub.py --config configs/train_config.yaml

Requirements:
  - HF_TOKEN must be set in .env or environment
  - outputs/final_adapter/ must exist (run train.py first)
  - results/before_finetune.json and results/after_finetune.json must
    exist (run evaluate.py --stage before and --stage after first)
"""

import argparse
import json
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer
from peft import PeftModel
from src.model_utils import load_model_for_eval


BEFORE_FILE = "results/before_finetune.json"
AFTER_FILE = "results/after_finetune.json"


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

    return f"""# {repo_id.split("/")[-1]}

Fine-tuned [{model_name}](https://huggingface.co/{model_name}) on [{dataset_name}](https://huggingface.co/datasets/{dataset_name}) using QLoRA for Text-to-SQL generation.

## Results

| Metric | Base Model | Fine-tuned | Delta |
|--------|-----------|------------|-------|
| Perplexity (eval split, ↓) | {ppl_before:.4f} | {ppl_after:.4f} | {ppl_delta:+.1f}% |
| ROUGE-L (200 samples, ↑) | {rouge_before:.4f} | {rouge_after:.4f} | {rouge_delta:+.1f}% |

Evaluated on a 5% held-out split of `{dataset_name}` (seed=42, {before["eval_split_size"]:,} examples).

## Training Configuration

| Setting | Value |
|---------|-------|
| Base model | [{model_name}](https://huggingface.co/{model_name}) |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | r={cfg["lora"]["r"]}, alpha={cfg["lora"]["lora_alpha"]} |
| LoRA target modules | q/k/v/o/gate/up/down_proj |
| Trainable parameters | ~20M (~0.67% of model) |
| Dataset | [{dataset_name}](https://huggingface.co/datasets/{dataset_name}) (~78k rows) |
| Epochs | {cfg["training"]["num_train_epochs"]} |
| Effective batch size | {cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]} |
| Learning rate | {cfg["training"]["learning_rate"]} (cosine schedule) |
| Optimizer | paged_adamw_8bit |

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "Given a SQL table schema, write a SQL query that answers the question."}},
    {{"role": "user", "content": "Table: employees (id INT, name TEXT, department TEXT, salary INT)\nQuestion: What is the average salary by department?"}},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Methodology

**What is QLoRA?** QLoRA (Quantized Low-Rank Adaptation) combines two techniques: the base model weights are frozen and compressed to 4-bit precision using NF4 quantization (reducing memory from ~6GB to ~2GB for a 3B model), while a small set of trainable LoRA adapter matrices are added in full precision. This means we only update ~20M parameters instead of 3B, making fine-tuning feasible on a single consumer GPU.

**Why sql-create-context?** The [sql-create-context dataset](https://huggingface.co/datasets/b-mc2/sql-create-context) contains ~78,000 natural language question / SQL answer pairs grounded in explicit `CREATE TABLE` schemas. Each example provides the full table definition, a natural language question, and the correct SQL query — teaching the model to ground its output in real schema structure rather than hallucinate column names or types.

**What do the numbers mean?** Perplexity measures how surprised the model is by the held-out text under teacher-forcing — a drop from {ppl_before:.1f} to {ppl_after:.1f} means the fine-tuned model assigns substantially higher probability to correct SQL responses. ROUGE-L measures the longest common subsequence overlap between generated and reference SQL on 200 sampled examples — an improvement from {rouge_before:.3f} to {rouge_after:.3f} confirms the model generates more structurally accurate queries, not just lower perplexity.

## Limitations

The model is trained for single-table SQL queries matching the style of the sql-create-context dataset. It may struggle with multi-table JOINs, nested subqueries, or dialects (e.g. T-SQL, PL/pgSQL) not represented in the training data. The model has not been RLHF-tuned for safety — use with the same caveats as the base `{model_name}`.

## Citation

```bibtex
@dataset{{sql_create_context,
  author = {{b-mc2}},
  title = {{sql-create-context}},
  year = {{2023}},
  url = {{https://huggingface.co/datasets/b-mc2/sql-create-context}},
}}
```

**Training framework:** PyTorch · Transformers · PEFT · TRL · bitsandbytes
"""


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

    login(token=hf_token)

    with open(BEFORE_FILE) as f:
        before = json.load(f)
    with open(AFTER_FILE) as f:
        after = json.load(f)

    repo_id = cfg["hub"]["repo_id"]
    print(f"Pushing adapter to: {repo_id}")

    # ── Push adapter weights ────────────────────────────────────────────────
    model = load_model_for_eval(cfg, adapter_path=None)
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    peft_model.push_to_hub(repo_id, private=cfg["hub"]["private"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.push_to_hub(repo_id)

    # ── Push README ─────────────────────────────────────────────────────────
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

    print(f"\nDone! Model pushed to:")
    print(f"  https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
