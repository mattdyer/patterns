# Qwen3-8B Post-Training Environment

## Setup Complete

- **Python**: 3.14.9
- **Virtual Environment**: `.venv/` (activate with `source .venv/bin/activate`)
- **PyTorch**: 2.11.0 + CUDA 12.8
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8.5GB VRAM)
- **Model**: Qwen3-8B (downloaded, cached locally in `~/.cache/huggingface/hub/`)

## Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.11.0+cu128 | Deep learning framework with CUDA |
| transformers | 5.22.2 | Model loading, tokenizers, quantization |
| accelerate | 1.15.0 | Training orchestration, device mapping |
| peft | 0.21.2 | Parameter-efficient fine-tuning (LoRA/QLoRA) |
| bitsandbytes | 0.50.2 | 4-bit/8-bit quantization |
| trl | 1.17.1 | Transformer Reinforcement Learning (SFT trainer) |
| datasets | 5.2.0 | Hugging Face dataset loading |
| safetensors | 0.8.0 | Safe model weight serialization |
| tensorboard | latest | Training visualization |

## Hardware Notes

- **8GB VRAM** means QLoRA (4-bit) is required for fine-tuning.
- Use small batch sizes (`per_device_train_batch_size=1`), gradient accumulation,
  and modest sequence lengths (512-1024) to stay within memory.
- `flash-attn` is not available due to CUDA version mismatch; training will use standard attention.

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt   # if starting fresh
python train.py                   # run the template script

source .venv/bin/activate
python run.py "What is machine learning?"
```

## Model Cache Location

```
~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/
```

Re-downloading is not required once cached. To clear: `rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/`
