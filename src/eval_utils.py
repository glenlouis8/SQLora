"""
eval_utils.py
─────────────
Perplexity and ROUGE-L computation for before/after evaluation.

Both metrics are computed on the held-out test split (seed=42, ~5% of counsel-chat).
  - Perplexity: full test split, batched forward passes
  - ROUGE-L:    200 sampled rows, generation + reference comparison
"""

import math
import torch
import numpy as np
from rouge_score import rouge_scorer

from src.data_utils import format_row, format_prompt_only


def compute_perplexity(model, tokenizer, dataset, cfg):
    """
    Compute perplexity over the full eval dataset using teacher-forced
    forward passes (no generation). Lower perplexity = better.

    Returns: float (perplexity)
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for row in dataset:
            text = format_row(row, tokenizer)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg["model"]["max_seq_length"],
            )
            input_ids = inputs["input_ids"].to(model.device)
            n_tokens = input_ids.shape[1]

            if n_tokens < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            # outputs.loss is mean NLL over tokens; scale back to total NLL
            nll = outputs.loss.item() * n_tokens
            total_nll += nll
            total_tokens += n_tokens

    perplexity = math.exp(total_nll / total_tokens)
    return round(perplexity, 4)


def compute_rouge_l(model, tokenizer, dataset, cfg):
    """
    Compute mean ROUGE-L (rougeLsum) over a sample of the eval dataset.
    Generation is greedy (do_sample=False) for reproducibility.

    Returns: float (mean ROUGE-L score, 0-1)
    """
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

    sample_size = cfg["data"]["eval_sample_size"]
    # Deterministic sample using numpy seed
    rng = np.random.default_rng(cfg["data"]["seed"])
    indices = rng.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)
    sample = dataset.select(indices.tolist())

    scores = []
    max_new_tokens = cfg["eval"]["generation_max_new_tokens"]

    for row in sample:
        prompt = format_prompt_only(row, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg["model"]["max_seq_length"],
        ).to(model.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Strip the prompt tokens to get only the new generated text
        new_tokens = generated_ids[0][prompt_len:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reference_text = row["answer"].strip()

        result = scorer.score(reference_text, generated_text)
        scores.append(result["rougeLsum"].fmeasure)

    mean_rouge_l = round(float(np.mean(scores)), 4)
    return mean_rouge_l
