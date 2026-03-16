"""
data_utils.py
─────────────
Dataset loading, train/eval splitting, and Alpaca → chat template formatting.

Both train.py and evaluate.py import from here to guarantee they always use
the identical split (same seed=42) and identical prompt format.
"""

from datasets import load_dataset


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


def format_alpaca_row(row, tokenizer):
    """
    Convert a single Alpaca row into a single training string using the
    model's official chat template.

    Alpaca fields: instruction, input (optional context), output.

    The full string includes the assistant turn so SFTTrainer can compute
    the causal LM loss only on the assistant tokens (via its loss masking).
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
        add_generation_prompt=True,  # appends the assistant header to trigger generation
    )


def get_formatting_func(tokenizer):
    """
    Return a formatting function closure suitable for SFTTrainer's
    `formatting_func` parameter.
    """
    def formatting_func(row):
        return format_alpaca_row(row, tokenizer)

    return formatting_func
