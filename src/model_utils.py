"""
model_utils.py
──────────────
Model and tokenizer loading helpers.

Handles:
  - BitsAndBytes 4-bit quantization config
  - LoRA / QLoRA config
  - Base model loading (for training)
  - Merged model loading (for evaluation)
  - CPU fallback when no CUDA GPU is available
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel


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
