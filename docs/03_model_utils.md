# 03 — `src/model_utils.py`

## What this file is

`model_utils.py` is the library responsible for loading the Llama 3.2 model and tokenizer in all the different configurations the project needs: quantized for training, quantized with LoRA adapters merged for evaluation, or on CPU as a fallback. It also builds the BitsAndBytes quantization config and the LoRA adapter config from the YAML settings. By centralizing model loading here, all scripts get the same model setup without duplicating complex configuration code.

## How it connects

- **Called by:** `train.py` (imports `load_tokenizer`, `load_base_model`, `get_lora_config`), `evaluate.py` (imports `load_tokenizer`, `load_model_for_eval`), `push_to_hub.py` (imports `load_model_for_eval`).
- **Calls:** `transformers.AutoModelForCausalLM`, `transformers.AutoTokenizer`, `transformers.BitsAndBytesConfig`, `peft.LoraConfig`, `peft.get_peft_model`, `peft.PeftModel`, and `torch.cuda.is_available()`.
- **Fits in the pipeline as:** the model layer. All model construction is contained here so changing quantization or LoRA settings in the YAML automatically affects all scripts.

---

## Full walkthrough

### Imports

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
```

- **`torch`**: PyTorch, the deep learning framework everything is built on. Used here mainly to check for GPU availability and specify data types.
- **`AutoModelForCausalLM`**: A HuggingFace class that automatically selects the right model architecture based on the model name. "CausalLM" means "Causal Language Model" — a model that predicts the next token, reading left-to-right. Llama is a CausalLM.
- **`AutoTokenizer`**: Similarly auto-selects the right tokenizer for the model. The tokenizer converts raw text into integers (token IDs) that the model understands.
- **`BitsAndBytesConfig`**: A configuration object that tells `AutoModelForCausalLM` how to quantize the weights during loading.
- **`LoraConfig`**: Defines the LoRA adapter architecture (rank, alpha, target modules, etc.).
- **`get_peft_model`**: Takes a regular model and a `LoraConfig` and returns a wrapped model where the specified layers have LoRA adapters attached and frozen.
- **`PeftModel`**: Used to load a saved LoRA adapter back onto a base model during evaluation.

---

### `get_bnb_config(cfg)`

```python
def get_bnb_config(cfg):
    """Build BitsAndBytesConfig for 4-bit NF4 quantization."""
    compute_dtype = (
        torch.bfloat16
        if cfg["model"]["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
    )
```

**Lines 1-4 — dtype resolution:**

The YAML stores the compute dtype as a string (`"bfloat16"`), but `BitsAndBytesConfig` needs an actual PyTorch dtype object (`torch.bfloat16`). This ternary expression converts the string to the appropriate object. If the string is anything other than `"bfloat16"`, it falls back to `torch.float16`.

**Lines 5-10 — constructing the config:**

`BitsAndBytesConfig` is a plain configuration object — it does not load any model. When later passed to `AutoModelForCausalLM.from_pretrained()`, the BitsAndBytes library intercepts the weight loading process and compresses each tensor as it is read from disk.

- `load_in_4bit=True`: compress weights to 4-bit NF4 instead of the default 16-bit.
- `bnb_4bit_quant_type`: `"nf4"` — the NF4 quantization scheme (see `01_train_config.md` for the explanation).
- `bnb_4bit_compute_dtype`: the precision used for actual math (bfloat16 here). Weights live on GPU in 4-bit; for each operation they are temporarily expanded to bfloat16, the math happens, and the result is stored. This is why quantization hurts speed only slightly while saving a lot of memory.
- `bnb_4bit_use_double_quant`: further compresses the quantization scale factors themselves.

Without this function: `load_base_model` and `load_model_for_eval` would each need to duplicate this config construction, and if you wanted to change the quantization type you would need to update it in multiple places.

---

### `get_lora_config(cfg)`

```python
def get_lora_config(cfg):
    """Build LoraConfig from the yaml lora block."""
    return LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )
```

This directly maps YAML config values to a `LoraConfig` object. `LoraConfig` is another plain data object — it describes the desired adapter architecture. It is passed to `get_peft_model()` in `train.py`, which modifies the model in-place by:
1. Freezing all original model parameters (setting `requires_grad=False`).
2. Adding LoRA adapter matrices (`A` and `B` matrices) to each of the seven target module types.
3. Setting only the adapter matrices to `requires_grad=True`.

After `get_peft_model()`, calling `model.print_trainable_parameters()` (as `train.py` does) shows something like: `trainable params: 20,054,016 || all params: 3,233,349,632 || trainable%: 0.6202`. This means only 0.62% of the model's parameters are actually being updated during training — the other 99.38% are frozen.

---

### `load_tokenizer(cfg)`

```python
def load_tokenizer(cfg):
    """
    Load tokenizer and apply required settings for SFT training:
      - pad_token = eos_token  (Llama has no pad token by default)
      - padding_side = "right" (required for correct loss masking in SFTTrainer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
```

**`AutoTokenizer.from_pretrained(...)`** downloads the tokenizer files for the Llama 3.2 model (vocabulary, merge rules for BPE tokenization, special tokens, chat template). `trust_remote_code=True` allows any custom Python code bundled with the tokenizer to execute — some models include custom tokenization logic.

**`tokenizer.pad_token = tokenizer.eos_token`** is a required fix for Llama. When training in batches, all sequences in a batch must be padded to the same length. Padding requires a special "pad token" that fills the empty positions. Llama's tokenizer does not define a pad token by default (it was pre-trained without batched padding). Setting `pad_token = eos_token` repurposes the end-of-sequence token as the padding token. `SFTTrainer` then masks out padding tokens when computing the training loss, so they do not interfere.

**`tokenizer.padding_side = "right"`** means padding tokens are added to the right (end) of sequences that are shorter than the batch's longest sequence. Right-padding is required for correct loss masking in `SFTTrainer`. The alternative, left-padding, would shift the position IDs of the real tokens, which can confuse the model's positional encoding. `SFTTrainer`'s documentation explicitly requires right-padding.

---

### `load_base_model(cfg)`

```python
def load_base_model(cfg):
    """
    Load the base model for training:
      - 4-bit quantization if CUDA is available
      - gradient checkpointing-compatible settings
      - CPU fallback with a clear warning
    """
    model_name = cfg["model"]["name"]

    if torch.cuda.is_available():
        bnb_config = get_bnb_config(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(
            "\n[WARNING] No CUDA GPU found. Running in CPU mode.\n"
            "Training will be extremely slow. Consider using a GPU runtime.\n"
            "4-bit quantization is disabled (requires CUDA).\n"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    # Required for gradient checkpointing + LoRA to work together
    model.config.use_cache = False
    model.enable_input_require_grads()

    return model
```

**`torch.cuda.is_available()`** — checks whether a CUDA-capable GPU (Nvidia) is available. On Google Colab, cloud VMs, or workstations with an Nvidia GPU, this returns `True`. On a Mac or CPU-only machine, it returns `False`.

**GPU path — `quantization_config=bnb_config`:** passes the BitsAndBytes config to load the model in 4-bit NF4. This is what makes QLoRA work.

**`device_map="auto"`:** tells HuggingFace to automatically distribute the model's layers across available devices (GPU, and CPU RAM if GPU VRAM is insufficient). For a 3B model on an 8GB GPU, everything fits on GPU. For a larger model, some layers would overflow to CPU. The "auto" map handles this transparently.

**CPU path — `torch_dtype=torch.float32`:** on CPU, we load the model in full 32-bit precision since bitsandbytes quantization only works with CUDA. This requires ~12GB of RAM and will train extremely slowly (hours per epoch instead of hours for the full 1-epoch run on GPU). The warning message makes this clear.

**`model.config.use_cache = False`** — disables the KV-cache. The KV-cache is an optimization for inference (generating text) that stores intermediate attention keys and values so they do not need to be recomputed at each step. However, gradient checkpointing recomputes activations on purpose during the backward pass, and the KV-cache interferes with this. Setting `use_cache = False` is required when `gradient_checkpointing = True`.

**`model.enable_input_require_grads()`** — a subtle but important call. When LoRA is used with gradient checkpointing, the gradient flow can break at the input embedding layer because the embeddings are not wrapped by LoRA. This call patches the input layer to ensure gradients flow correctly through it into the LoRA adapter layers above. Without this, training runs but the LoRA adapters receive zero gradient and learn nothing.

---

### `load_model_for_eval(cfg, adapter_path=None)`

```python
def load_model_for_eval(cfg, adapter_path=None):
    """
    Load model for evaluation.
    - adapter_path=None  → raw base model (before fine-tuning evaluation)
    - adapter_path=str   → base model with LoRA adapter merged in (after evaluation)

    merge_and_unload() folds the LoRA weights back into the base model so
    generation benchmarks measure the actual merged behavior.
    """
    model_name = cfg["model"]["name"]

    if torch.cuda.is_available():
        bnb_config = get_bnb_config(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Loaded and merged adapter from: {adapter_path}")

    model.eval()
    return model
```

The first section mirrors `load_base_model` — same GPU/CPU branching, same BitsAndBytes config. Notice it does **not** call `model.enable_input_require_grads()` or set `use_cache = False`, because we are not training — we do not need gradients.

**`adapter_path=None`** is used for the "before" evaluation: we want to measure the raw base model before any fine-tuning. The model is returned as-is.

**`adapter_path=str`** is used for the "after" evaluation: we load the saved LoRA adapter on top of the base model.

**`PeftModel.from_pretrained(model, adapter_path)`** reads the saved adapter weights (the `A` and `B` matrices for each target layer) from disk and attaches them to the base model. At this point, you have a two-component model: frozen base weights + small trainable adapter weights.

**`model.merge_and_unload()`** combines both components by mathematically adding `B·A` (scaled by `alpha/r`) into each corresponding base weight matrix, then discards the separate adapter structure. The result is a single standard model with no PEFT wrapper. Why do this? Because merged models generate text slightly faster (one matrix multiplication instead of two), and certain generation functions work more cleanly with a plain model than a PEFT-wrapped one. For evaluation, this is the right representation.

**`model.eval()`** switches the model to evaluation mode, which disables dropout (LoRA has `dropout=0.05`, so without `.eval()`, ROUGE-L scores would vary between runs).

---

## Things to know

**`load_base_model` and `load_model_for_eval` have similar but not identical GPU paths.** `load_base_model` adds `model.config.use_cache = False` and `model.enable_input_require_grads()` — both needed for training but not evaluation. If you accidentally use `load_base_model` for evaluation, you would get a slight speed penalty from the disabled cache, and `enable_input_require_grads` would be a no-op but harmless.

**`trust_remote_code=True` is a security consideration.** This allows HuggingFace to execute Python code bundled in the model repository. For official Meta models on HuggingFace, this is safe. For models from unknown authors, `trust_remote_code=True` could execute malicious code. Only use it with repositories you trust.

**4-bit quantization is irreversible in memory.** Once loaded in 4-bit, there is no in-memory "dequantize" back to full precision. `merge_and_unload()` merges the LoRA adapters into the quantized weights, which means the merged weights are still 4-bit. This is acceptable for evaluation and inference but means the saved merged model (if you were to save it) would also be 4-bit. The adapter weights themselves (saved by `train.py`) are always full-precision bfloat16.

**CPU mode does not support bitsandbytes.** The BitsAndBytes library depends on CUDA GPU kernels that simply do not exist on CPU. The else branch explicitly avoids passing `quantization_config` and uses `float32` instead. Running a 3B model in float32 on CPU requires ~12GB of RAM.

**`device_map="auto"` can be unpredictable on machines with multiple GPUs.** It distributes layers greedily across available devices. If you have two GPUs and want to control exactly which layers go where, you would pass a custom `device_map` dict instead.
