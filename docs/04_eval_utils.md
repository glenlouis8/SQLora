# 04 — `src/eval_utils.py`

## What this file is

`eval_utils.py` is the library that implements the two evaluation metrics used to measure model quality: perplexity and ROUGE-L. These functions are called twice in the project — once on the base model before training and once on the fine-tuned model after — providing a concrete before/after comparison. Both functions are designed to be deterministic and reproducible given the same model and dataset.

## How it connects

- **Called by:** `evaluate.py` (imports `compute_perplexity`, `compute_rouge_l`). No other file directly imports from here.
- **Calls:** `src.data_utils.format_alpaca_row` (for perplexity), `src.data_utils.format_alpaca_prompt_only` (for ROUGE-L), `model.forward()` implicitly via `model(...)`, `model.generate()`, `rouge_score.rouge_scorer.RougeScorer`.
- **Fits in the pipeline as:** the measurement layer. The two functions here produce the numbers that end up in the JSON result files and ultimately in the published README.

---

## Full walkthrough

### Imports

```python
import math
import torch
import numpy as np
from rouge_score import rouge_scorer

from src.data_utils import format_alpaca_row, format_alpaca_prompt_only
```

- **`math`**: used for `math.exp()` to convert average negative log-likelihood to perplexity.
- **`torch`**: used for `torch.no_grad()` (disables gradient computation during evaluation to save memory and speed up passes).
- **`numpy`**: used for `np.random.default_rng` (deterministic random sampling) and `np.mean`, `np.percentile`.
- **`rouge_score`**: Google's reference implementation of ROUGE metrics. `RougeScorer` computes various ROUGE variants including ROUGE-L.
- **`format_alpaca_row`**: formats a row with both question and answer — used for perplexity's teacher-forced pass.
- **`format_alpaca_prompt_only`**: formats a row with only the question — used for ROUGE-L's generation step.

---

### `compute_perplexity(model, tokenizer, dataset, cfg)`

```python
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
            text = format_alpaca_row(row, tokenizer)
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
```

**`model.eval()`** — sets the model to evaluation mode. In training mode, dropout is active (randomly zeroing neurons), making results stochastic. In eval mode, dropout is disabled, so the same input always produces the same output.

**`total_nll = 0.0` and `total_tokens = 0`** — running accumulators. We are computing a weighted average of the per-token loss across all examples in the dataset. We cannot simply average the per-example losses because examples have different lengths — a 200-token example would count the same as a 20-token example. Instead, we accumulate the total token count and total negative log-likelihood, then divide at the end.

**`with torch.no_grad():`** — disables PyTorch's gradient computation graph. During training, PyTorch records all operations so it can compute gradients during the backward pass. During evaluation, this recording is unnecessary and wastes memory and time. `no_grad()` turns it off.

**`text = format_alpaca_row(row, tokenizer)`** — formats the row with both instruction and answer (the full sequence). This is correct for perplexity: we want to measure how well the model predicts the entire conversation, including the answer tokens.

**`inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=...)`** — converts the text string into token IDs. `return_tensors="pt"` returns PyTorch tensors. `truncation=True` cuts off the sequence at `max_seq_length` (2048) if it is longer. The result has shape `[1, sequence_length]` (a batch of 1).

**`input_ids = inputs["input_ids"].to(model.device)`** — moves the token tensor to the same device as the model (GPU or CPU). If they are on different devices, the forward pass will fail.

**`if n_tokens < 2: continue`** — safety check. A sequence of 0 or 1 tokens cannot produce a meaningful loss (there are no "next tokens" to predict). This guards against empty or near-empty rows.

**`outputs = model(input_ids, labels=input_ids)`** — this is the key line. Passing `labels=input_ids` (setting labels equal to inputs) tells the model to compute the causal language modeling loss: for each position, predict the token at position `i+1` given all tokens up to position `i`. The loss is the mean negative log-likelihood across all token positions in the sequence.

To understand this intuitively: think of the model as a very sophisticated autocomplete. At each position, it estimates a probability for every possible next token. The loss measures how surprised it was by the actual next token: `loss = -log(probability assigned to correct next token)`. If the model was very confident and correct (probability = 0.99), the loss is tiny (-log(0.99) ≈ 0.01). If it was very wrong (probability = 0.01), the loss is large (-log(0.01) ≈ 4.6).

**`nll = outputs.loss.item() * n_tokens`** — `outputs.loss` is already the mean NLL over all tokens in this example. Multiplying by `n_tokens` recovers the total (unnormalized) NLL for this example, so we can accumulate and later normalize by the total token count across all examples.

**`perplexity = math.exp(total_nll / total_tokens)`** — the final calculation. `total_nll / total_tokens` is the mean NLL per token across the entire dataset. Taking the exponential converts from log-probability to probability scale: `exp(NLL)` gives the "effective vocabulary size" the model is choosing from at each step. A perplexity of 5 means the model is on average as uncertain as if it were choosing uniformly from 5 equally likely options.

**`round(..., 4)`** — rounds to 4 decimal places before returning. This keeps the result clean in JSON output.

---

### `compute_rouge_l(model, tokenizer, dataset, cfg)`

```python
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
        prompt = format_alpaca_prompt_only(row, tokenizer)
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
        reference_text = row["output"].strip()

        result = scorer.score(reference_text, generated_text)
        scores.append(result["rougeLsum"].fmeasure)

    mean_rouge_l = round(float(np.mean(scores)), 4)
    return mean_rouge_l
```

**`scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)`** — creates the ROUGE scorer object. We request `"rougeLsum"`, which is the "summary-level" ROUGE-L: it splits both the reference and generated texts into sentences, then computes the longest common subsequence across the sentence structure. This is more forgiving than strict sequence-level ROUGE-L, which is appropriate for multi-sentence answers. `use_stemmer=True` reduces words to their root form before comparison (e.g., "running" and "run" are treated as the same), which reduces unfair penalties for minor morphological differences.

**`rng = np.random.default_rng(cfg["data"]["seed"])`** — creates a reproducible random number generator seeded with the same `seed=42` used elsewhere. This ensures the same 200 rows are selected every time ROUGE-L is computed, so before/after comparisons are apples-to-apples.

**`indices = rng.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)`** — draws 200 unique indices from the range `[0, len(dataset))`. `replace=False` means no row is selected twice. `min(sample_size, len(dataset))` handles the edge case where the dataset is smaller than 200 rows.

**`sample = dataset.select(indices.tolist())`** — the HuggingFace `Dataset.select()` method returns a new dataset containing only the specified row indices, in the specified order.

**`prompt = format_alpaca_prompt_only(row, tokenizer)`** — formats the question without the answer. This is critical: we want the model to generate its answer, not continue from the reference answer.

**`.to(model.device)`** — moves all tokenized tensors (input_ids, attention_mask) to the model's device in one call. This works because `tokenizer(...)` returns a dict-like object, and `to(device)` on that object moves all tensor values.

**`prompt_len = inputs["input_ids"].shape[1]`** — records how many tokens are in the prompt before generation. We will use this to strip the prompt from the model's output.

**`model.generate(...)`** — actually generates new tokens. Key arguments:
- `**inputs`: unpacks `input_ids` and `attention_mask` as keyword arguments to `generate()`.
- `max_new_tokens=256`: stop generating after 256 new tokens regardless of content.
- `do_sample=False`: use greedy decoding — always pick the highest-probability next token. This makes output deterministic and reproducible.
- `pad_token_id=tokenizer.eos_token_id`: tells the generation loop what token to use for padding when batching multiple sequences. We are not batching here (one at a time), but some generation code paths still require this to be set.

The output `generated_ids` has shape `[1, prompt_len + new_tokens]` — it includes both the original prompt tokens and the newly generated tokens.

**`new_tokens = generated_ids[0][prompt_len:]`** — slices off the prompt tokens, keeping only the new ones. `generated_ids[0]` removes the batch dimension. `[prompt_len:]` takes everything after the prompt. Without this step, the "generated text" would include the instruction and system prompt, which would artificially inflate the ROUGE-L score (since the reference text is just the answer).

**`tokenizer.decode(new_tokens, skip_special_tokens=True).strip()`** — converts token IDs back to a readable string. `skip_special_tokens=True` removes special tokens like `<|eot_id|>` from the output. `.strip()` removes leading/trailing whitespace.

**`reference_text = row["output"].strip()`** — the gold reference answer from the Alpaca dataset.

**`result = scorer.score(reference_text, generated_text)`** — computes ROUGE-L between the reference and generated text. The first argument is the reference; the second is the hypothesis (generated text). Returns a dict with a `Score` named-tuple for each requested ROUGE variant.

**`scores.append(result["rougeLsum"].fmeasure)`** — `.fmeasure` is the F1 score, which balances precision (what fraction of the generated text is in the reference) and recall (what fraction of the reference is in the generated text). F1 is the standard choice for ROUGE in most evaluation settings.

**`mean_rouge_l = round(float(np.mean(scores)), 4)`** — averages the 200 individual ROUGE-L scores. `float()` converts from numpy float64 to a Python float so it serializes cleanly to JSON.

---

## Things to know

**Perplexity computation runs on the full 2,600-row test split with no sampling.** This takes longer than a sampled approach but gives a more accurate population estimate. On a modern GPU, each forward pass is fast (milliseconds), so 2,600 passes is typically under 5 minutes.

**ROUGE-L computation generates text for 200 rows one at a time.** Generation is sequential (no batching) because different prompts have different lengths and the code keeps it simple. Batched generation would be faster but requires padding all prompts to the same length and tracking which output tokens belong to which example. For 200 examples, the sequential approach is fast enough (~20-30 minutes on a GPU).

**`rougeLsum` vs `rougeL`:** `rougeLsum` (summary-level) splits text into sentences and computes LCS across sentence structure, while `rougeL` (sequence-level) computes a single LCS over the entire text as one sequence. `rougeLsum` is generally considered more appropriate for longer, multi-sentence responses. For very short Alpaca answers (often one or two sentences), the two will give nearly identical results.

**The ROUGE-L score is influenced by generation quality, not just vocabulary overlap.** A model that generates grammatically correct but semantically wrong answers will score poorly. A model that generates the exact same words in a different order will also score poorly (LCS is order-sensitive). This makes ROUGE-L a reasonable but imperfect proxy for answer quality.

**Both metrics use the same test split.** The `dataset` parameter passed to both functions is `splits["test"]` from `load_and_split()`. Using the same split for both metrics is intentional — they measure different aspects (fluency vs. generation quality) on the same held-out examples.

**`outputs.loss` is already averaged across the sequence.** This is the default behavior of HuggingFace causal LM models when `labels` is provided. The averaging means we must multiply by `n_tokens` to recover the total NLL before accumulating, then divide by total tokens at the end. If we simply averaged `outputs.loss` across examples, a long 200-token example would count the same as a short 20-token example, under-weighting longer (and often harder) examples.
