#!/usr/bin/env python3
"""
Run inference with a fine-tuned Qwen3-8B model.

Usage:
    source .venv/bin/activate
    python run.py "Your prompt here"
"""

import argparse
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Qwen3-8B model")
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to generate a response for",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "lora-adapter"),
        help="Path to the LoRA adapter directory (default: output/lora-adapter)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, use 0.0 for greedy)",
    )
    return parser.parse_args()


def load_model_and_adapter():
    MODEL_PATH = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/main"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    global args
    args = parse_args()

    model, tokenizer = load_model_and_adapter()

    print(f"\nPrompt: {args.prompt}\n")
    response = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
