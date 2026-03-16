# 02 — `src/data_utils.py`

## What this file is

`data_utils.py` is the central library for everything related to the Alpaca dataset: downloading it, splitting it reproducibly into train and test halves, and converting raw dataset rows into the formatted text strings the model actually trains on and is evaluated against. It is imported by three different scripts, and having all data logic here ensures that training and evaluation always see the data in exactly the same format.

## How it connects

- **Called by:** `train.py` (imports `load_and_split`, `get_formatting_func`), `evaluate.py` (imports `load_and_split`), `prepare_data.py` (imports `load_and_split`, `format_alpaca_row`), and `eval_utils.py` (imports `format_alpaca_row`, `format_alpaca_prompt_only`).
- **Calls:** `datasets.load_dataset` (HuggingFace Datasets library) and `tokenizer.apply_chat_template` (HuggingFace Transformers tokenizer method).
- **Fits in the pipeline as:** the data layer. Every other module that needs data goes through this file.

---

## Full walkthrough

### Imports

```python
from datasets import load_dataset
```

The `datasets` library (from HuggingFace) is what downloads and caches the Alpaca dataset. It handles HTTP requests, decompression, and local caching automatically. The first time `load_dataset` is called, it downloads the data; subsequent calls use a local cache.

---

### `load_and_split(cfg)`

```python
def load_and_split(cfg):
    """
    Load tatsu-lab/alpaca and return a train/test DatasetDict.
    Always uses cfg.data.seed so splits are reproducible across scripts.
    """
    dataset = load_dataset(cfg["data"]["dataset_name"], split="train")
    split = dataset.train_test_split(
        test_size=cfg["data"]["eval_split_ratio"],
        seed=cfg["data"]["seed"],
    )
    return split  # {"train": ..., "test": ...}
```

**Line 1 — `load_dataset(cfg["data"]["dataset_name"], split="train")`:**

The Alpaca dataset on HuggingFace only has a `"train"` split — there is no official test set. So we load the entire dataset as a single object (52,002 rows). `split="train"` tells the library to load that specific partition; without it, you would get a `DatasetDict` object containing all splits, which would then require further indexing.

**Line 2 — `dataset.train_test_split(...)`:**

This splits the 52,002 rows randomly into two subsets. `test_size=0.05` means 5% goes to the test set (~2,601 rows) and 95% goes to training (~49,401 rows). The `seed=42` argument is critical: it seeds the random number generator so that every call to this function — whether from `train.py`, `evaluate.py`, or `prepare_data.py` — produces the exact same partition. This guarantee is what makes before/after evaluation valid: you are measuring the model's performance on the same test examples it never saw during training.

**Return value:** A Python dict-like object `{"train": Dataset(...), "test": Dataset(...)}`. Scripts access the splits with `splits["train"]` and `splits["test"]`.

Without this function: every script would need its own copy of the splitting logic. If they ever used different seeds by accident, training and evaluation would bleed into each other, and the evaluation results would be meaningless.

---

### `format_alpaca_row(row, tokenizer)`

```python
def format_alpaca_row(row, tokenizer):
    """
    Convert a single Alpaca row into a single training string using the
    model's official chat template.
    ...
    """
    user_content = row["instruction"]
    if row.get("input", "").strip():
        user_content = f"{row['instruction']}\n\n{row['input']}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"]},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
```

**Lines 1-3 — building `user_content`:**

An Alpaca row has three fields: `instruction` (a task description), `input` (optional extra context), and `output` (the expected answer). About a third of Alpaca rows have a non-empty `input`. For example:

```
instruction: "Summarize the following text."
input: "The sun is a star located at the center of our solar system..."
output: "The sun is a central star of our solar system."
```

When `input` is present, we concatenate instruction and input with two newlines between them to make a single user message. When `input` is empty (or just whitespace), we use the instruction alone. `.strip()` guards against rows where `input` is `"  "` (whitespace only), which would otherwise create an ugly double-newline with nothing after it.

**Lines 4-9 — building the messages list:**

Llama models are trained to follow a specific "chat template" — a structured format that marks which parts of the text came from the system, the user, and the assistant. This format uses special tokens like `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>`. The exact format matters: if you format your training data differently than the format the model learned during its pre-training instruction-tuning phase, the model will be confused.

Rather than manually constructing this format (which would be brittle), we use the tokenizer's built-in `apply_chat_template()` method, which knows the exact format for this specific model.

The three messages are:
1. A "system" message setting the persona: `"You are a helpful assistant."` This is prepended to all examples to establish context.
2. A "user" message containing the task.
3. An "assistant" message containing the expected answer.

**Lines 10-14 — `apply_chat_template(...)`:**

`tokenize=False` means: give me the formatted string, not token IDs. We want the raw text so `SFTTrainer` can handle tokenization with its own settings (truncation, padding, etc.).

`add_generation_prompt=False` means: do not append the special marker that would tell the model "now start generating." We include the full assistant response because we are building training examples — we want the model to see both the question and the answer.

The result looks something like:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize the following text.

The sun is a star...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The sun is a central star...<|eot_id|>
```

This function is used by: `prepare_data.py` (to preview formatting), `eval_utils.py`/`compute_perplexity` (for teacher-forced perplexity), and `get_formatting_func` below (for SFTTrainer).

---

### `format_alpaca_prompt_only(row, tokenizer)`

```python
def format_alpaca_prompt_only(row, tokenizer):
    """
    Format a row WITHOUT the assistant turn — used during evaluation to
    produce the prompt we feed to model.generate().
    """
    user_content = row["instruction"]
    if row.get("input", "").strip():
        user_content = f"{row['instruction']}\n\n{row['input']}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
```

This is nearly identical to `format_alpaca_row`, with two differences:

1. **The messages list has only two entries** — system and user. There is no assistant message. We want the model to generate that itself.

2. **`add_generation_prompt=True`** appends the special header token that signals "the assistant's turn begins here." Without this, the model would not know it should start generating a response. The formatted string ends with something like `<|start_header_id|>assistant<|end_header_id|>\n\n`, which is exactly the cue the Llama model needs.

This function is used exclusively by `eval_utils.py`/`compute_rouge_l` — the ROUGE-L evaluation that requires generating model outputs.

Why two separate functions instead of one with a flag? It makes each function's intent unambiguous, and it prevents accidental mistakes where training data is formatted without the answer (making the model compute loss on nothing useful) or evaluation data is formatted with the answer (making the model's generated text contain the reference answer before it even generates anything).

---

### `get_formatting_func(tokenizer)`

```python
def get_formatting_func(tokenizer):
    """
    Return a formatting function closure suitable for SFTTrainer's
    `formatting_func` parameter.
    """
    def formatting_func(row):
        return format_alpaca_row(row, tokenizer)

    return formatting_func
```

`SFTTrainer` (from the TRL library) accepts a `formatting_func` parameter: a function that takes a single dataset row and returns a formatted string. The challenge is that `SFTTrainer` calls this function with only one argument (the row), but `format_alpaca_row` needs two arguments (the row and the tokenizer).

This function solves that with a **closure**. A closure is a function that "captures" variables from its surrounding scope. Here, `formatting_func` is defined inside `get_formatting_func`, so it has access to the `tokenizer` variable from the outer function's scope. When `SFTTrainer` calls `formatting_func(row)`, it automatically uses the tokenizer that was captured.

In `train.py`:
```python
trainer = SFTTrainer(
    ...
    formatting_func=get_formatting_func(tokenizer),
    ...
)
```

Without this closure: either `SFTTrainer` would need to be patched to pass the tokenizer to the formatting function (not possible without modifying the TRL library), or you would need to use a global variable for the tokenizer (bad practice), or you would need a lambda `lambda row: format_alpaca_row(row, tokenizer)` (which works but is less explicit).

---

## Things to know

**The split is deterministic but requires the same version of the `datasets` library.** If the library's internal shuffling algorithm changes between major versions, the same `seed=42` might produce a different partition. When comparing results across machines or environments, ensure the `datasets` library version matches.

**`row.get("input", "")` vs `row["input"]`:** The `.get()` call with a default of `""` protects against rows where the `"input"` key is absent entirely (which can happen in some dataset variants). Pure `row["input"]` would raise a `KeyError`. The `.strip()` further protects against whitespace-only strings.

**`apply_chat_template` is model-specific.** If you change `model.name` in the config to a non-Llama model, the chat template will change. Mistral, Phi, Gemma, and other models all use different special token structures. The tokenizer bundles the correct template for its model, so `apply_chat_template` will automatically adapt — but the actual string format will look different. This is expected behavior.

**`format_alpaca_row` produces the full sequence including the answer for perplexity calculation.** This is correct: perplexity is computed over the entire sequence (question + answer), which is the standard way to measure language model quality on instruction-following datasets. An alternative would be to mask the instruction tokens and compute loss only on the answer, but the current implementation is simpler and gives a reasonable signal.

**There is no explicit caching in this file.** If you call `load_and_split()` multiple times in the same script (e.g., once in training and once in evaluation), it will call `load_dataset()` twice. However, the `datasets` library caches data on disk, so the second call is fast (it reads from the local cache rather than downloading again).
