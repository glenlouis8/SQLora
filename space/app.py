"""
app.py — HuggingFace Spaces entry point
────────────────────────────────────────
Loads the fine-tuned Llama 3.2-3B QLoRA adapter from the Hub and serves
a Gradio demo. No local files required.

The HF_TOKEN secret must be set in the Space settings (Settings → Secrets)
so the gated Llama 3.2 model can be downloaded.
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_REPO = "glen-louis/llama-3.2-3b-alpaca-qlora"
HF_TOKEN     = os.environ.get("HF_TOKEN")

MAX_SEQ_LENGTH   = 2048
DEFAULT_MAX_NEW  = 256


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model...")
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
        )
    else:
        print("[WARNING] No GPU found — running on CPU (slow).")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            token=HF_TOKEN,
        )

    print(f"Loading LoRA adapter from: {ADAPTER_REPO}")
    model = PeftModel.from_pretrained(model, ADAPTER_REPO, token=HF_TOKEN)
    model = model.merge_and_unload()
    model.eval()
    print("Model ready.")
    return model, tokenizer


model, tokenizer = load_model()


# ── Inference ─────────────────────────────────────────────────────────────────
def build_prompt(instruction, input_text):
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{instruction.strip()}\n\n{input_text.strip()}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def respond(instruction, input_text, max_new_tokens, do_sample, temperature):
    if not instruction.strip():
        return "Please enter an instruction."

    prompt = build_prompt(instruction, input_text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Llama 3.2 QLoRA Demo") as demo:
    gr.Markdown("# Llama 3.2-3B Alpaca QLoRA")
    gr.Markdown(
        "Fine-tuned on [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) "
        "using 4-bit NF4 QLoRA (LoRA r=16, all 7 projection layers, 1 epoch). "
        f"Adapter: [{ADAPTER_REPO}](https://huggingface.co/{ADAPTER_REPO})"
    )

    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(
                label="Instruction",
                lines=4,
                placeholder="e.g. Write a Python function to check if a number is prime.",
            )
            input_text = gr.Textbox(
                label="Input (optional context)",
                lines=3,
                placeholder="Leave blank if your instruction needs no extra context.",
            )
            with gr.Row():
                max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="Max new tokens")
                do_sample = gr.Checkbox(label="Sampling (creative)", value=False)
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            submit = gr.Button("Generate", variant="primary")
            gr.Examples(
                examples=[
                    ["Write a Python function to check if a number is prime.", ""],
                    ["Explain gradient descent as if I'm five years old.", ""],
                    ["Write a haiku about machine learning.", ""],
                    ["Summarize the following paragraph.", "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration, converting nutrients into energy the cell can use."],
                    ["Translate to French.", "The weather is beautiful today."],
                    ["Give three tips for staying productive while working from home.", ""],
                ],
                inputs=[instruction, input_text],
            )

        with gr.Column():
            output = gr.Textbox(label="Response", lines=20, interactive=False)

    submit.click(
        fn=respond,
        inputs=[instruction, input_text, max_new_tokens, do_sample, temperature],
        outputs=output,
    )
    instruction.submit(
        fn=respond,
        inputs=[instruction, input_text, max_new_tokens, do_sample, temperature],
        outputs=output,
    )

demo.launch()
