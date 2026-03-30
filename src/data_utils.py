"""
data_utils.py
─────────────
Dataset loading, train/eval splitting, and SQL context → chat template formatting.

Both train.py and evaluate.py import from here to guarantee they always use
the identical split (same seed=42) and identical prompt format.

Dataset: b-mc2/sql-create-context
Fields used:
  question  — natural language question
  context   — SQL table schema (CREATE TABLE statements)
  answer    — the correct SQL query (training target)
"""

from datasets import load_dataset


def load_and_split(cfg):
    """
    Load sql-create-context and return a train/test DatasetDict.
    Always uses cfg.data.seed so splits are reproducible across scripts.
    """
    dataset = load_dataset(cfg["data"]["dataset_name"], split="train")
    split = dataset.train_test_split(
        test_size=cfg["data"]["eval_split_ratio"],
        seed=cfg["data"]["seed"],
    )
    return split  # {"train": ..., "test": ...}


def format_row(row, tokenizer):
    """
    Convert a single sql-create-context row into a training string using the
    model's official chat template.

    Fields: question (natural language), context (schema), answer (SQL query).

    The full string includes the assistant turn so SFTTrainer can compute
    the causal LM loss only on the assistant tokens (via its loss masking).
    """
    user_content = f"{row['question']}\n\nSchema:\n{row['context']}"

    messages = [
        {"role": "system", "content": "You are a SQL expert. Given a natural language question and a database schema, write a SQL query that answers the question. Return only the SQL query with no explanation."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["answer"]},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def format_prompt_only(row, tokenizer):
    """
    Format a row WITHOUT the assistant turn — used during evaluation to
    produce the prompt we feed to model.generate().
    """
    user_content = f"{row['question']}\n\nSchema:\n{row['context']}"

    messages = [
        {"role": "system", "content": "You are a SQL expert. Given a natural language question and a database schema, write a SQL query that answers the question. Return only the SQL query with no explanation."},
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
        return format_row(row, tokenizer)

    return formatting_func
