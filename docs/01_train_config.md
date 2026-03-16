# 01 — `configs/train_config.yaml`

## What this file is

`train_config.yaml` is the single configuration file that controls every tunable setting in the entire project: which model to load, how to quantize it, how to configure the LoRA adapters, how to split the dataset, all training hyperparameters, evaluation behavior, and where to publish the result. Every script reads this file at startup so that changing one number here automatically propagates to all downstream steps.

## How it connects

- **Called by:** `prepare_data.py`, `train.py`, `evaluate.py`, `push_to_hub.py` — all four scripts open this file with `yaml.safe_load()` at the start of `main()`.
- **Calls nothing** — it is a pure data file, not code.
- **Fits in the pipeline as:** the project's "control panel." It sits outside all Python code so that hyperparameters can be changed without touching source files.

---

## Full walkthrough

### Model section

```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  max_seq_length: 2048
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
```

**`name`** is the HuggingFace model identifier. `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()` both use this string to download the model and its tokenizer. Changing this to a different model ID (e.g. `meta-llama/Llama-3.2-1B-Instruct`) would fine-tune a different base model instead.

**`max_seq_length: 2048`** is the maximum number of tokens the model will process in a single forward pass. Any formatted training example longer than this is silently truncated. `prepare_data.py` specifically checks whether the p99 of training lengths exceeds this value and warns you if so. 2048 tokens is generous for Alpaca examples — most are under 400 tokens — so truncation is rare here.

**`load_in_4bit: true`** tells the BitsAndBytes library to compress the base model weights from 16-bit floats to 4-bit NF4 before loading them into GPU memory. This halves memory usage roughly twice over, enabling a 3B model to fit on a GPU with 8GB VRAM.

**`bnb_4bit_compute_dtype: "bfloat16"`** controls the floating-point format used for arithmetic during the forward and backward passes. Even though the weights are stored in 4-bit, the actual matrix multiplications happen in bfloat16 — a 16-bit format that is numerically stable on modern GPUs (Ampere architecture and newer, e.g. RTX 3090, 4090, A100). If your GPU is older (pre-Ampere), you would change this to `"float16"`.

**`bnb_4bit_quant_type: "nf4"`** selects "Normal Float 4" as the 4-bit encoding scheme. NF4 is specifically designed for neural network weights, which tend to follow a bell-curve (normal) distribution. By placing more of its 16 representable values near zero (where most weights live), NF4 preserves more information than a naive 4-bit encoding. The alternative, `fp4`, is a uniform 4-bit float that wastes resolution on large-magnitude values that are rarely needed.

**`bnb_4bit_use_double_quant: true`** enables "nested quantization" — after quantizing the weights to 4-bit NF4, the quantization constants themselves (small numbers that describe the scale of each block) are also quantized. This saves an additional ~0.4 bits per parameter on average, which translates to roughly 200MB saved for a 3B model.

---

### LoRA section

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**`r: 16`** is the LoRA rank — the size of the "bottleneck" in the side-car matrices. Each weight matrix `W` gets a pair of small matrices `A` (shape `d × r`) and `B` (shape `r × d`). The adapted weight is `W + B·A`. With `r=16`, the two side-cars together have `2 × d × 16` parameters. Lower rank (e.g. `r=4`) uses less memory but may underfit; higher rank (e.g. `r=64`) trains more parameters and may overfit. For a 3B model fine-tuned on ~50k examples, `r=16` is a well-tested default.

**`lora_alpha: 32`** is a scaling factor. The adaptation is actually applied as `W + (alpha/r) × B·A`, so with `alpha=32` and `r=16`, the scale factor is 2.0. This means the LoRA update gets amplified by 2× before being added to the frozen weight. The reason for this convention: when you change `r`, you usually want to also adjust the effective learning rate of the adapters. Setting `alpha = 2×r` keeps the scale consistent and is the most common convention.

**`lora_dropout: 0.05`** randomly zeros out 5% of the LoRA activations during each training step. Dropout is a regularization technique — it forces the network not to rely too heavily on any single adapter neuron, which reduces overfitting. 0.05 is mild; a 3B model fine-tuned for 1 epoch on 50k examples is not particularly prone to overfitting, so this is mostly a safety net.

**`bias: "none"`** tells LoRA not to train the bias terms (the additive offset in each linear layer). Training biases with LoRA is rarely beneficial and slightly increases the adapter file size.

**`task_type: "CAUSAL_LM"`** tells the PEFT library that this is a causal language model (one that predicts the next token left-to-right). This controls how LoRA integrates with the model architecture. For encoder models (like BERT) you would use `"SEQ_CLS"` or similar.

**`target_modules`** lists the specific linear layers inside each Transformer block that will receive LoRA adapters. These seven layers cover all the important weight matrices in Llama 3.2:

| Module | What it does |
|--------|-------------|
| `q_proj` | Projects token representations into "queries" for attention |
| `k_proj` | Projects token representations into "keys" for attention |
| `v_proj` | Projects token representations into "values" for attention |
| `o_proj` | Projects the attention output back to the model dimension |
| `gate_proj` | Controls the gating pathway in the SwiGLU feed-forward block |
| `up_proj` | Projects up to the expanded feed-forward dimension |
| `down_proj` | Projects back down from the expanded feed-forward dimension |

Targeting all seven ensures the model can adjust both how it attends to context and how it processes information in each layer. Targeting fewer modules (e.g., only `q_proj` and `v_proj` as in the original LoRA paper) reduces trainable parameters but limits expressiveness.

---

### Data section

```yaml
data:
  dataset_name: "tatsu-lab/alpaca"
  train_split_ratio: 0.95
  eval_split_ratio: 0.05
  eval_sample_size: 200
  seed: 42
```

**`dataset_name`** is the HuggingFace Datasets identifier for the Alpaca dataset (52,002 rows). This string is passed directly to `load_dataset()`.

**`train_split_ratio: 0.95` / `eval_split_ratio: 0.05`** define the 95/5 train-test split. With 52,002 rows: ~49,400 training rows and ~2,602 test rows. `train_split_ratio` is not actually read by any script — only `eval_split_ratio` is used as the `test_size` argument to `train_test_split()`.

**`eval_sample_size: 200`** controls how many rows are used for ROUGE-L evaluation. Generating text is slow (seconds per example on a GPU), so computing ROUGE-L on all 2,600 test rows would take hours. 200 is a pragmatic subsample that gives a statistically meaningful score in a reasonable time (~15-30 minutes depending on hardware).

**`seed: 42`** is used in three places: the train/test split (so the same rows are always in each split), the ROUGE-L sample selection (so the same 200 rows are always used), and the global random seed set in `train.py`. This makes the entire experiment reproducible — running it twice on the same hardware gives the same results.

---

### Training section

```yaml
training:
  output_dir: "outputs/checkpoints"
  final_adapter_dir: "outputs/final_adapter"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.001
  fp16: false
  bf16: true
  max_grad_norm: 1.0
  logging_steps: 25
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 2
```

**`output_dir`** is where HuggingFace's `Trainer` writes checkpoints during training (every 500 steps). These are recovery points in case training crashes.

**`final_adapter_dir`** is where the final adapter weights are saved after training completes. `evaluate.py --stage after` and `push_to_hub.py` both read from here.

**`num_train_epochs: 1`** means the model sees each training example exactly once. For a 50k-row dataset, this is typically sufficient to significantly improve instruction-following without overfitting. More epochs risk the model memorizing the training data instead of generalizing.

**`per_device_train_batch_size: 4`** means 4 examples are processed in parallel per GPU per step. Higher batch sizes are faster but require more VRAM.

**`gradient_accumulation_steps: 4`** is a memory-saving trick. Instead of processing 16 examples at once (which would need more VRAM), we process 4 examples four times in a row and sum up the gradients before updating the weights. The effective batch size is `4 × 4 = 16`. To the optimizer, this looks identical to a batch size of 16.

**`gradient_checkpointing: true`** is another memory-saving technique. Normally, the forward pass saves intermediate activations so the backward pass can use them. With gradient checkpointing, only a subset of activations are saved; the rest are recomputed during the backward pass. This reduces VRAM by roughly 40% at the cost of ~20% slower training (because some computations are done twice).

**`optim: "paged_adamw_8bit"`** uses a special 8-bit version of the AdamW optimizer that additionally "pages" its state (the running mean and variance of gradients) to CPU RAM when GPU VRAM is tight. AdamW normally stores two copies of the model parameters as optimizer state, which would use several GB of VRAM. The paged 8-bit version reduces this to ~1GB.

**`learning_rate: 2.0e-4`** is the peak step size for weight updates. `2e-4` is a commonly effective default for LoRA fine-tuning — large enough to train quickly, small enough not to destroy the base model's capabilities. This is the maximum value; the actual learning rate follows a schedule.

**`lr_scheduler_type: "cosine"`** means the learning rate starts at 0, ramps up to `2e-4` (over `warmup_ratio=0.03` of total steps), then smoothly decreases following a cosine curve to near 0 by the end. This prevents overshooting at the start and allows fine-grained adjustments near the end.

**`warmup_ratio: 0.03`** means the first 3% of training steps are used for the warmup ramp. With ~12,350 total steps (49,400 rows ÷ 16 effective batch size × 4 grad-accum), that is ~371 warmup steps.

**`weight_decay: 0.001`** is L2 regularization — it gently penalizes large adapter weights, helping prevent overfitting. The value is very small here because LoRA adapters are already small and unlikely to grow excessively.

**`fp16: false` / `bf16: true`** together select bfloat16 mixed-precision training. Both cannot be true simultaneously. `fp16` is an older 16-bit format that can cause numerical instability (overflow/underflow). `bfloat16` has a wider dynamic range and is preferred on modern Nvidia GPUs.

**`max_grad_norm: 1.0`** clips the gradient vector so its total magnitude never exceeds 1.0. This prevents a single "bad" batch from causing a huge destructive weight update (the "exploding gradients" problem).

**`logging_steps: 25`** prints training loss to the console every 25 optimizer steps.

**`save_strategy: "steps"` / `save_steps: 500`** saves a checkpoint to `output_dir` every 500 steps.

**`save_total_limit: 2`** keeps only the 2 most recent checkpoints, deleting older ones automatically. This prevents the checkpoints directory from growing indefinitely.

---

### Eval section

```yaml
eval:
  generation_max_new_tokens: 256
  generation_temperature: 0.1
  generation_do_sample: false
```

**`generation_max_new_tokens: 256`** caps how many new tokens the model generates per example during ROUGE-L evaluation. If the model tries to generate more than 256 tokens, it is cut off. Most Alpaca answers fit within 256 tokens. This is a time safeguard.

**`generation_temperature: 0.1`** controls randomness in text generation. Temperature 0 is fully deterministic (always pick the highest-probability token). Temperature 1 samples according to the model's raw probability distribution. 0.1 is nearly deterministic but technically `generation_do_sample: false` overrides this anyway.

**`generation_do_sample: false`** forces greedy decoding — always select the single highest-probability next token. This makes ROUGE-L evaluation perfectly reproducible: running it twice gives identical generated text.

---

### Hub section

```yaml
hub:
  repo_id: "YOUR_HF_USERNAME/llama-3.2-3b-alpaca-qlora"
  private: false
```

**`repo_id`** is the HuggingFace Hub repository identifier where the adapter will be published. You must replace `YOUR_HF_USERNAME` with your actual HuggingFace username before running `push_to_hub.py`.

**`private: false`** controls whether the published repository is publicly visible. Set to `true` if you want to share only with specific users.

---

## Things to know

**`train_split_ratio` is defined but not used.** Only `eval_split_ratio` is passed to `train_test_split()` as `test_size`. The `train_split_ratio` key is present for documentation purposes (so a reader can see that 95% is training data) but the code derives it implicitly as `1 - eval_split_ratio`. Do not change one without knowing the other stays consistent.

**Changing `seed` invalidates cached splits.** HuggingFace Datasets caches results. If you change the seed, delete the `.cache` folder (or use `load_dataset(..., download_mode="force_redownload")`) to ensure a fresh split.

**`bf16: true` requires Ampere-class GPU or newer.** On older GPUs (e.g., GTX 1080, RTX 2080), you must set `bf16: false` and `fp16: true` instead. The `model_utils.py` code already handles CPU fallback, but it does not automatically detect non-Ampere GPUs — you must adjust this manually.

**`paged_adamw_8bit` requires the `bitsandbytes` library.** If you get an import error about the optimizer, make sure `bitsandbytes` is installed (`pip install bitsandbytes`).

**The `final_adapter_dir` path is used by three different scripts.** `train.py` writes to it, `evaluate.py --stage after` reads from it, and `push_to_hub.py` reads from it. If you change this value in the config, all three scripts automatically use the new path — but any already-saved adapters at the old path will not be found.
