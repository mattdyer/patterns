#!/usr/bin/env python3
"""
Template Qwen3-8B QLoRA fine-tuning script using TRL SFTTrainer.
Designed for 8GB VRAM — uses 4-bit quantization + LoRA.

Usage:
    source .venv/bin/activate
    python train.py
"""

import json
import os
import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

# ── Model & Data ────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/main"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")

# ── 4-bit quantization config ──────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── LoRA config ────────────────────────────────────────────────────────
# Only ~1-2% of parameters updated; keeps VRAM low
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
}

# ── Training hyperparameters ───────────────────────────────────────────
TRAIN_ARGS = {
    # Data
    "max_seq_length": 512,          # Keep low for 8GB VRAM

    # Optimization
    "learning_rate": 2.0e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "optim": "adamw_8bit",           # 8-bit optimizer saves VRAM

    # Scheduling
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",

    # Precision & memory
    "bf16": False,
    "fp16": True,
    "gradient_checkpointing": True,

    # Logging
    "logging_steps": 10,
    "output_dir": OUTPUT_DIR,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "report_to": "tensorboard",
}


def load_tokenizer():
    """Load the Qwen3 tokenizer."""
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )


def load_model():
    """Load Qwen3-8B in 4-bit quantized mode."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )


def load_data(path: str) -> Dataset:
    """Load instruction-tuning data from a JSON file."""
    with open(path, "r") as f:
        items = json.load(f)

    # Convert to the format TRL expects
    formatted = []
    for item in items:
        prompt = f"### Instruction: {item['instruction']}"
        if item.get("input"):
            prompt += f"\n\n### Input: {item['input']}"
        prompt += f"\n\n### Answer: {item['output']}"
        formatted.append({"text": prompt})

    return Dataset.from_list(formatted)


def train():
    """Run the full QLoRA fine-tuning pipeline."""
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (4-bit QLoRA)...")
    model = load_model()

    print("Loading data...")
    dataset = load_data(DATA_FILE)

    print("Configuring SFT trainer...")
    sft_args = SFTConfig(**TRAIN_ARGS)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
        peft_config=LORA_CONFIG,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapters
    output_path = os.path.join(OUTPUT_DIR, "lora-adapter")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved adapter to: {output_path}")


if __name__ == "__main__":
    train()
