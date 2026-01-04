# 🇨🇿 SLiM-CZ-V1

**Slavic Linguistic integrated Micro-model for Czechia**

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/filipsedivy/SLiM-CZ-V1)

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