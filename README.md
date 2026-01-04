<center><h1>🇨🇿 SLiM-CZ-V1</h1></center>

**Slavic Linguistic integrated Micro-model for Czechia**

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/filipsedivy/SLiM-CZ-V1)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/filipsedivy/SLiM-CZ-V1)
[![Follow us on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-sm.svg)](https://huggingface.co/filipsedivy)

A small Czech language model based on Transformer architecture with Multi-Head Attention mechanism.
---

## 📋 Project Overview

SLiM-CZ-V1 is a complete implementation of a small language model for Czech text, featuring:

- **Multi-Head Attention** mechanism with 8 heads
- **Transformer architecture** (GPT-style)
- **Complete data preprocessing** pipeline
- **Training infrastructure** with checkpointing
- **Flexible model sizes** from Tiny (~3M params) to Large (~125M params)

---

## 📦 Installation

### Using uv (recommended)

```bash
# Clone repository
git clone https://github.com/filipsedivy/SLiM-CZ-V1
cd SLiM-CZ-V1

# Install with uv
uv pip install -e .

# Or install with extras
uv pip install -e ".[all]"
```

### Using pip

```bash
pip install -e .
```

---

## 🚀 Quick Start

### Prepare Your Data

```bash
python src/prepare_data.py \
    --input ./raw_texts \
    --output ./processed_data
```

### Train with Configuration

```bash
# Train with default config (small)
python src/train.py --data-dir ./processed_data

# Train with specific config
python src/train.py --data-dir ./processed_data --config tiny
python src/train.py --data-dir ./processed_data --config medium

# Override config parameters
python src/train.py --data-dir ./processed_data --config small --epochs 20
```

### Run Inference

```bash
# Interactive mode
python src/inference_cli.py --model ./models/best_model.pt

# One-time generation
python src/inference_cli.py \
    --model ./models/best_model.pt \
    --prompt "Praha je" \
    --max-tokens 100
```

---

## ⚙️ Configuration System

SLiM-CZ-V1 uses YAML configuration files:

### Available Configurations

| Config | Parameters | Size | Use Case | File |
|--------|-----------|------|----------|------|
| **Tiny** | ~3M | ~12 MB | Testing, debugging | `cfg/tiny.yaml` |
| **Small** | ~14M | ~56 MB | Development (default) | `cfg/small.yaml` |
| **Medium** | ~56M | ~224 MB | Production | `cfg/medium.yaml` |
| **Large** | ~125M | ~500 MB | High-end | `cfg/large.yaml` |

### Using Configurations

```bash
# Use predefined config
python src/train.py --data-dir ./data --config tiny
python src/train.py --data-dir ./data --config small
python src/train.py --data-dir ./data --config medium
python src/train.py --data-dir ./data --config large

# Use custom config file
python src/train.py --data-dir ./data --config path/to/custom.yaml

# Override specific parameters
python src/train.py --data-dir ./data --config small \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 0.0001
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Prepare your text data
python src/prepare_data.py \
    --input ./raw_texts \
    --output ./processed_data

# 2. Train with configuration
python src/train.py \
    --data-dir ./processed_data \
    --config small \
    --output-dir ./models

# 3. Test with CLI
python src/inference_cli.py --model ./models/best_model.pt
```

---

## 🔧 Data Preparation Pipeline

### Supported File Formats

`.txt`, `.md`, `.rst`, `.py`, `.js`, `.html`, `.css`, `.json`, `.xml`, `.csv`, `.log`, `.c`, `.cpp`, `.java`

### Pipeline Steps

1. **File Collection** - Recursively scan directories
2. **Text Cleaning** - Remove URLs, emails, normalize whitespace
3. **Tokenization** - Character-level tokenizer (can be replaced with BPE)
4. **Sequence Creation** - Create overlapping sequences
5. **Dataset Split** - Split into train/val/test

### Example

```bash
python prepare_data.py \
    --input ./my_texts \
    --output ./processed \
    --vocab-size 30000 \
    --seq-len 1024 \
    --stride 512 \
    --lowercase
```

---

## 🎮 CLI Inference

```bash
# Interactive mode
python src/inference_cli.py --model model.pt --tokenizer tokenizer.json

# Commands in interactive mode:
#   - Enter prompt to generate
#   - 'settings' to change parameters
#   - 'quit' to exit

# One-shot generation
python src/inference_cli.py \
    --model model.pt \
    --prompt "Your prompt here" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-k 50
```

---

## 📖 Configuration Reference

### Model Parameters

| Parameter | Description | Tiny | Small | Medium | Large |
|-----------|-------------|------|-------|--------|-------|
| `vocab_size` | Vocabulary size | 10k | 10k | 30k | 50k |
| `d_model` | Embedding dimension | 128 | 256 | 512 | 768 |
| `num_heads` | Attention heads | 4 | 8 | 8 | 12 |
| `num_layers` | Transformer layers | 4 | 6 | 8 | 12 |
| `d_ff` | FFN dimension | 512 | 1024 | 2048 | 3072 |
| `max_seq_len` | Max sequence length | 256 | 512 | 1024 | 2048 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size | 32 |
| `learning_rate` | Learning rate | 0.0001 |
| `epochs` | Number of epochs | 10 |
| `scheduler` | LR scheduler | cosine |
| `gradient_clip` | Gradient clipping | 1.0 |
| `patience` | Early stopping | 5 |

---

## 🎯 Use Cases

- **Text Generation**: Generate Czech text
- **Language Modeling**: Train on Czech corpus
- **Fine-tuning**: Adapt to specific domains
- **Research**: Experiment with transformer architectures
- **Education**: Learn about language models

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👨‍💻 Contributing

Contributions welcome! This is a reference implementation.

---

**Happy modeling! 🚀🇨🇿**