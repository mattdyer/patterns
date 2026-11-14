#!/usr/bin/env python3
"""
Test the model on pattern-related questions.
Run before training (on base model) and after training (with adapter)
to see the difference in responses.

Usage:
    # Base model (before training)
    source .venv/bin/activate
    python test_model.py

    # Trained model (after training)
    source .venv/bin/activate
    python test_model.py --adapter output/lora-adapter
"""

import argparse
import json
import os
import torch
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ── Test questions ────────────────────────────────────────────────────
# These cover the different pattern types the model should learn.
TEST_QUESTIONS = [
    # --- Number sequences: explain ---
    {
        "category": "number_explain",
        "prompt": "### Instruction: Explain the pattern\n### Input: 1, 4, 9, 16, 25\n### Answer:",
        "expected_hint": "square",
    },
    {
        "category": "number_explain",
        "prompt": "### Instruction: Explain the pattern\n### Input: 1, 2, 3, 5, 8, 13\n### Answer:",
        "expected_hint": "fibonacci",
    },
    {
        "category": "number_explain",
        "prompt": "### Instruction: Explain the pattern\n### Input: 2, 3, 5, 7, 11, 13\n### Answer:",
        "expected_hint": "prime",
    },
    {
        "category": "number_explain",
        "prompt": "### Instruction: Explain the pattern\n### Input: 1, 8, 27, 64, 125\n### Answer:",
        "expected_hint": "cube",
    },
    {
        "category": "number_explain",
        "prompt": "### Instruction: Explain the pattern\n### Input: 1, 2, 4, 8, 16\n### Answer:",
        "expected_hint": "power",
    },

    # --- Number sequences: next item ---
    {
        "category": "number_next",
        "prompt": "### Instruction: Find the next item in the pattern\n### Input: 1, 3, 6, 10, 15\n### Answer:",
        "expected_hint": "21",
    },
    {
        "category": "number_next",
        "prompt": "### Instruction: Find the next item in the pattern\n### Input: 1, 2, 4, 8, 16\n### Answer:",
        "expected_hint": "32",
    },
    {
        "category": "number_next",
        "prompt": "### Instruction: Find the next item in the pattern\n### Input: 1, 1, 2, 3, 5\n### Answer:",
        "expected_hint": "8",
    },
    {
        "category": "number_next",
        "prompt": "### Instruction: Find the next item in the pattern\n### Input: 1, 4, 10, 20, 35\n### Answer:",
        "expected_hint": "56",
    },
    {
        "category": "number_next",
        "prompt": "### Instruction: Find the next item in the pattern\n### Input: 1, 2, 6, 24, 120\n### Answer:",
        "expected_hint": "720",
    },

    # --- Pattern breaks ---
    {
        "category": "pattern_break",
        "prompt": "### Instruction: Where does the pattern break and why\n### Input: 1, 2, 4, 8, 15\n### Answer:",
        "expected_hint": "16",
    },
    {
        "category": "pattern_break",
        "prompt": "### Instruction: Where does the pattern break and why\n### Input: 2, 4, 6, 9, 10\n### Answer:",
        "expected_hint": "odd",
    },
    {
        "category": "pattern_break",
        "prompt": "### Instruction: Where does the pattern break and why\n### Input: 1, 4, 9, 15, 25\n### Answer:",
        "expected_hint": "16",
    },
    {
        "category": "pattern_break",
        "prompt": "### Instruction: Where does the pattern break and why\n### Input: 1, 2, 3, 5, 7, 12\n### Answer:",
        "expected_hint": "not prime",
    },
    {
        "category": "pattern_break",
        "prompt": "### Instruction: Where does the pattern break and why\n### Input: 1, 3, 5, 8, 9\n### Answer:",
        "expected_hint": "even",
    },

    # --- Code patterns ---
    {
        "category": "code_pattern",
        "prompt": "### Instruction: Explain the code pattern\n### Input: function getUser() { ... }\nfunction createUser() { ... }\nfunction deleteUser() { ... }\nfunction updateUser() { ... }\n### Answer:",
        "expected_hint": "CRUD",
    },
    {
        "category": "code_pattern",
        "prompt": "### Instruction: Explain the code pattern\n### Input: router.get('/users', handler)\nrouter.post('/users', handler)\nrouter.put('/users/:id', handler)\nrouter.delete('/users/:id', handler)\n### Answer:",
        "expected_hint": "REST",
    },
    {
        "category": "code_pattern",
        "prompt": "### Instruction: Explain the code pattern\n### Input: function validateEmail(email) {\n  if (!email) return false;\n  const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n  return regex.test(email);\n}\n### Answer:",
        "expected_hint": "validation",
    },
    {
        "category": "code_pattern",
        "prompt": "### Instruction: Where does the code pattern break and why\n### Input: function getUser(id) { ... }\nfunction createUser(user) { ... }\nfunction deletePost(id) { ... }\nfunction updateUser(id, data) { ... }\n### Answer:",
        "expected_hint": "post",
    },

    # --- Writing patterns ---
    {
        "category": "writing_pattern",
        "prompt": "### Instruction: Explain the writing pattern\n### Input: The cat sat on the mat. The cat jumped off the mat. The cat ran across the yard.\n### Answer:",
        "expected_hint": "active",
    },
    {
        "category": "writing_pattern",
        "prompt": "### Instruction: Where does the writing pattern break and why\n### Input: The cat sat on the mat. The cat jumped off the mat. On the mat sat the cat.\n### Answer:",
        "expected_hint": "passive",
    },
    {
        "category": "writing_pattern",
        "prompt": "### Instruction: Explain the writing pattern\n### Input: She was tired. She was hungry. She was ready to go home.\n### Answer:",
        "expected_hint": "repetition",
    },

    # --- Real-world patterns ---
    {
        "category": "realworld_pattern",
        "prompt": "### Instruction: Explain the pattern\n### Input: Monday, Tuesday, Wednesday, Thursday, Friday\n### Answer:",
        "expected_hint": "weekday",
    },
    {
        "category": "realworld_pattern",
        "prompt": "### Instruction: Where does the pattern break and why\n### Input: January, February, March, June, May\n### Answer:",
        "expected_hint": "swapped",
    },
    {
        "category": "realworld_pattern",
        "prompt": "### Instruction: Explain the pattern\n### Input: Red, Orange, Yellow, Green, Blue, Indigo, Violet\n### Answer:",
        "expected_hint": "rainbow",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test the model on pattern-related questions")
    parser.add_argument(
        "--adapter",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output", "lora-adapter"),
        help="Path to LoRA adapter (default: output/lora-adapter)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a JSON file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max tokens per response (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Low temperature for consistent answers (default: 0.1)",
    )
    parser.add_argument(
        "--use-trained",
        action="store_true",
        help="Use trained adapter (otherwise uses base model)",
    )
    return parser.parse_args()


def load_model_and_tokenizer(adapter_path=None):
    """Load the model, optionally with a trained adapter."""
    MODEL_PATH = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/main"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.eval()

    if adapter_path and os.path.isdir(adapter_path):
        print(f"Loading trained adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("Using base model (no adapter loaded)")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH if adapter_path is None else adapter_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_test(model, tokenizer, question, max_new_tokens, temperature):
    """Run one test question and return the response."""
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    gen_kwargs = {"do_sample": temperature > 0.0, "pad_token_id": tokenizer.eos_token_id}
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the answer part (after the prompt)
    response = response.replace(question, "").strip()
    return response


def main():
    args = parse_args()

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        adapter_path=args.adapter if args.use_trained else None
    )

    print(f"Running {len(TEST_QUESTIONS)} tests...")
    print()

    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        response = run_test(
            model, tokenizer, q["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        result = {
            "index": i,
            "category": q["category"],
            "prompt": q["prompt"],
            "response": response,
            "expected_hint": q["expected_hint"],
        }
        results.append(result)

        print(f"[{i:3d}/{len(TEST_QUESTIONS)}] {q['category']:20s} | {response[:100]}...")

    # ── Save results ──────────────────────────────────────────────────
    if args.output is None:
        tag = "trained" if args.use_trained else "base"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(os.path.dirname(__file__), "test_results", f"{tag}_{ts}.json")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print(f"Total: {len(results)} tests run")


if __name__ == "__main__":
    main()
