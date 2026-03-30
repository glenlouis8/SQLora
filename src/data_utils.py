"""
data_utils.py
─────────────
Dataset loading, train/eval splitting, and Counsel Chat → chat template formatting.

Both train.py and evaluate.py import from here to guarantee they always use
the identical split (same seed=42) and identical prompt format.

Dataset: nbertagnolli/counsel-chat
Fields used:
  questionTitle  — short title of the user's question
  questionText   — full question body (optional, may be empty)
  answerText     — therapist's response (training target)
  upvotes        — community upvotes; used to filter for quality answers
"""

from datasets import load_dataset


def load_and_split(cfg):
    """
    Load counsel-chat and return a train/test DatasetDict.
    Filters to answers with at least 2 upvotes to keep quality high.
    Always uses cfg.data.seed so splits are reproducible across scripts.
    """
    dataset = load_dataset(cfg["data"]["dataset_name"], split="train")
    dataset = dataset.filter(lambda row: row["upvotes"] >= 2)
    split = dataset.train_test_split(
        test_size=cfg["data"]["eval_split_ratio"],
        seed=cfg["data"]["seed"],
    )
    return split  # {"train": ..., "test": ...}


def format_row(row, tokenizer):
    """
    Convert a single Counsel Chat row into a training string using the
    model's official chat template.

    Counsel Chat fields: questionTitle, questionText (optional), answerText.

    The full string includes the assistant turn so SFTTrainer can compute
    the causal LM loss only on the assistant tokens (via its loss masking).
    """
    user_content = row["questionTitle"]
    if (row.get("questionText") or "").strip():
        user_content = f"{row['questionTitle']}\n\n{row['questionText']}"

    messages = [
        {"role": "system", "content": "You are a compassionate mental health counselor. Provide supportive, empathetic, and evidence-based responses to people seeking mental health guidance."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["answerText"]},
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
    user_content = row["questionTitle"]
    if (row.get("questionText") or "").strip():
        user_content = f"{row['questionTitle']}\n\n{row['questionText']}"

    messages = [
        {"role": "system", "content": "You are a compassionate mental health counselor. Provide supportive, empathetic, and evidence-based responses to people seeking mental health guidance."},
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
