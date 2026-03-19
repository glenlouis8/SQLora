---
title: Llama 3.2-3B Alpaca QLoRA
emoji: 🦙
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: llama3.2
---

# Llama 3.2-3B Alpaca QLoRA Demo

Fine-tuned on [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) using 4-bit NF4 QLoRA.

- **Base model:** meta-llama/Llama-3.2-3B-Instruct
- **Adapter:** LoRA r=16 on all 7 projection layers (~20M trainable params)
- **Training:** 1 epoch, SFTTrainer, paged_adamw_8bit
