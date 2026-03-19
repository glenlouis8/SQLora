"""
demo.py
───────
Gradio web demo for the fine-tuned Llama QLoRA model.

Usage:
  pip install gradio
  python scripts/demo.py

  # Load from HuggingFace Hub
  python scripts/demo.py --adapter glen-louis/llama-3.2-3b-alpaca-qlora

  # Show base model side-by-side with fine-tuned
  python scripts/demo.py --compare

  # Make accessible over the network (share link)
  python scripts/demo.py --share
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
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt, cfg, max_new_tokens, do_sample, temperature):
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
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_single_demo(model, tokenizer, cfg):
    import gradio as gr

    def respond(instruction, input_text, max_new_tokens, do_sample, temperature):
        if not instruction.strip():
            return "Please enter an instruction."
        prompt = build_prompt(instruction, input_text, tokenizer)
        return generate(model, tokenizer, prompt, cfg, int(max_new_tokens), do_sample, temperature)

    with gr.Blocks(title="Llama 3.2 QLoRA Demo") as demo:
        gr.Markdown("## Llama 3.2-3B Alpaca QLoRA — Fine-tuned Demo")
        gr.Markdown(
            "Model fine-tuned on [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) "
            "using 4-bit NF4 QLoRA (LoRA r=16, 1 epoch)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                instruction = gr.Textbox(label="Instruction", lines=3,
                                         placeholder="e.g. Write a Python function to reverse a string")
                input_text = gr.Textbox(label="Input (optional context)", lines=2,
                                        placeholder="Leave blank if not needed")
                with gr.Row():
                    max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="Max new tokens")
                    do_sample = gr.Checkbox(label="Sampling", value=False)
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                submit = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                output = gr.Textbox(label="Response", lines=15, interactive=False)

        submit.click(
            fn=respond,
            inputs=[instruction, input_text, max_new_tokens, do_sample, temperature],
            outputs=output,
        )

        gr.Examples(
            examples=[
                ["Write a Python function to check if a number is prime.", ""],
                ["Explain the concept of gradient descent in simple terms.", ""],
                ["Summarize the following text.", "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration, converting nutrients into energy the cell can use."],
                ["Translate the following sentence to French.", "The weather is beautiful today."],
                ["Give three tips for staying productive while working from home.", ""],
            ],
            inputs=[instruction, input_text],
        )

    return demo


def make_compare_demo(base_model, ft_model, tokenizer, cfg):
    import gradio as gr

    def respond(instruction, input_text, max_new_tokens, do_sample, temperature):
        if not instruction.strip():
            return "Please enter an instruction.", "Please enter an instruction."
        prompt = build_prompt(instruction, input_text, tokenizer)
        base_out = generate(base_model, tokenizer, prompt, cfg, int(max_new_tokens), do_sample, temperature)
        ft_out = generate(ft_model, tokenizer, prompt, cfg, int(max_new_tokens), do_sample, temperature)
        return base_out, ft_out

    with gr.Blocks(title="Llama 3.2 QLoRA — Before vs After") as demo:
        gr.Markdown("## Llama 3.2-3B — Base vs Fine-tuned Comparison")

        with gr.Row():
            instruction = gr.Textbox(label="Instruction", lines=3,
                                     placeholder="e.g. Write a Python function to reverse a string")
            input_text = gr.Textbox(label="Input (optional)", lines=3,
                                    placeholder="Leave blank if not needed")

        with gr.Row():
            max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="Max new tokens")
            do_sample = gr.Checkbox(label="Sampling", value=False)
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")

        submit = gr.Button("Generate Both", variant="primary")

        with gr.Row():
            base_output = gr.Textbox(label="Base Model", lines=15, interactive=False)
            ft_output = gr.Textbox(label="Fine-tuned Model", lines=15, interactive=False)

        submit.click(
            fn=respond,
            inputs=[instruction, input_text, max_new_tokens, do_sample, temperature],
            outputs=[base_output, ft_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for the fine-tuned Llama QLoRA model")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--adapter", default=None,
                        help="Adapter path or HuggingFace repo ID (default: outputs/final_adapter)")
    parser.add_argument("--compare", action="store_true",
                        help="Load both base and fine-tuned model for side-by-side comparison")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    adapter_path = args.adapter or cfg["training"]["final_adapter_dir"]
    if not args.adapter and not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found at '{adapter_path}'.\n"
            "Run train.py first, or pass --adapter <hub-repo-id> to load from the Hub."
        )

    tokenizer = load_tokenizer(cfg)

    if args.compare:
        print("Loading base model...")
        base_model = load_model_for_eval(cfg, adapter_path=None)
        print(f"Loading fine-tuned model from: {adapter_path}")
        ft_model = load_model_for_eval(cfg, adapter_path=adapter_path)
        demo = make_compare_demo(base_model, ft_model, tokenizer, cfg)
    else:
        print(f"Loading fine-tuned model from: {adapter_path}")
        model = load_model_for_eval(cfg, adapter_path=adapter_path)
        demo = make_single_demo(model, tokenizer, cfg)

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
