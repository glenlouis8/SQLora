"""
infer.py
────────
Run inference on the fine-tuned model from the command line.

Usage:
  # Single instruction
  python scripts/infer.py --prompt "Write a Python function to reverse a string"

  # Instruction + optional context input
  python scripts/infer.py --instruction "Summarize the following" --input "The quick brown fox..."

  # Interactive REPL (keep model loaded, type prompts one by one)
  python scripts/infer.py --interactive

  # Load from HuggingFace Hub instead of local adapter
  python scripts/infer.py --adapter glen-louis/llama-3.2-3b-alpaca-qlora --prompt "..."

  # Use base model (no adapter) for comparison
  python scripts/infer.py --base --prompt "..."
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.model_utils import load_tokenizer, load_model_for_eval


def build_prompt(instruction, input_text, tokenizer):
    user_content = instruction
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt, cfg, max_new_tokens=None):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["model"]["max_seq_length"],
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    max_new = max_new_tokens or cfg["eval"]["generation_max_new_tokens"]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=cfg["eval"]["generation_do_sample"],
            temperature=cfg["eval"]["generation_temperature"] if cfg["eval"]["generation_do_sample"] else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(cfg, adapter, use_base):
    if use_base:
        adapter_path = None
        print("Loading base model (no adapter)...")
    elif adapter:
        adapter_path = adapter
        print(f"Loading model with adapter: {adapter_path}")
    else:
        adapter_path = cfg["training"]["final_adapter_dir"]
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at '{adapter_path}'.\n"
                "Run train.py first, or pass --adapter <hub-repo-id> to load from the Hub."
            )
        print(f"Loading model with local adapter: {adapter_path}")

    tokenizer = load_tokenizer(cfg)
    model = load_model_for_eval(cfg, adapter_path=adapter_path)
    return model, tokenizer


def run_interactive(model, tokenizer, cfg):
    print("\nInteractive mode — type your instruction and press Enter.")
    print("Optionally prefix with 'input:' on a second line for context.")
    print("Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            instruction = input(">> Instruction: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        input_text = input(">> Input (optional, press Enter to skip): ").strip()

        prompt = build_prompt(instruction, input_text, tokenizer)
        print("\n--- Response ---")
        print(generate(model, tokenizer, prompt, cfg))
        print("----------------\n")


def main():
    parser = argparse.ArgumentParser(description="Inference on the fine-tuned Llama QLoRA model")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Path to adapter dir OR HuggingFace repo ID (overrides config default)")
    parser.add_argument("--base", action="store_true",
                        help="Load base model without any adapter (for comparison)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override generation max_new_tokens from config")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prompt", type=str, help="Single instruction prompt")
    mode.add_argument("--interactive", action="store_true", help="Interactive REPL mode")

    parser.add_argument("--input", type=str, default="",
                        help="Optional context/input to accompany --prompt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model(cfg, args.adapter, args.base)

    if args.interactive:
        run_interactive(model, tokenizer, cfg)
    else:
        if not args.prompt:
            parser.error("Provide --prompt <text> or use --interactive mode.")
        prompt = build_prompt(args.prompt, args.input, tokenizer)
        response = generate(model, tokenizer, prompt, cfg, max_new_tokens=args.max_new_tokens)
        print("\n--- Response ---")
        print(response)
        print("----------------")


if __name__ == "__main__":
    main()
