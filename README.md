# 🇨🇿 SLiM-CZ-V1

**Slavic Linguistic integrated Micro-model for Czechia**

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/filipsedivy/SLiM-CZ-V1)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/filipsedivy/SLiM-CZ-V1)
[![Follow us on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-sm.svg)](https://huggingface.co/filipsedivy)

A compact Czech language model based on Transformer architecture optimized for small datasets (2-5M tokens) with Multi-Head Attention mechanism and research-backed optimizations.

---

## 📋 Project Overview

SLiM-CZ-V1 is a complete implementation of a small language model for Czech text, featuring:

- **Multi-Head Attention** mechanism (6-8 heads)
- **GPT-style Transformer architecture** with weight tying
- **Complete data preprocessing** pipeline with SentencePiece tokenizer
- **Training infrastructure** with TensorBoard, checkpointing, and early stopping
- **Flexible model sizes** from Tiny (~3.6M params) to Medium (~19.8M params)
- **Chinchilla scaling laws** for optimal parameter/data matching
- **Research optimizations** for small datasets (enhanced dropout, weight decay, label smoothing)

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/filipsedivy/SLiM-CZ-V1
cd SLiM-CZ-V1
```

### Install with uv (Recommended)

```bash
# Install with all features (monitoring, analysis, dev tools)
uv pip install -e ".[all]"

# Or install only what you need
uv pip install -e ".[monitoring]"  # TensorBoard, tqdm
uv pip install -e .                # Core only (torch, sentencepiece, pyyaml)
```

### Install with pip

```bash
# Install with all features
pip install -e ".[all]"

# Or install core dependencies only
pip install -e .

# Or install specific feature sets
pip install -e ".[monitoring]"     # TensorBoard, tqdm
pip install -e ".[analysis]"       # pandas, scikit-learn
pip install -e ".[dev]"           # pytest, black, ruff
```

### Manual Installation (if not using package)

```bash
pip install torch>=2.0.0 sentencepiece>=0.1.99 pyyaml>=6.0.0
pip install tqdm tensorboard  # Optional: for progress bars and visualization
```

**Core dependencies:**
- torch >= 2.0.0
- sentencepiece >= 0.1.99
- pyyaml >= 6.0.0
- numpy >= 1.21.0

**Optional dependencies:**
- `[monitoring]`: tqdm, tensorboard
- `[analysis]`: pandas, scikit-learn
- `[dev]`: pytest, black, ruff
- `[all]`: All of the above

---

## 🔧 Command Line Interface (CLI)

After installing with `pip install -e .` (or `uv pip install -e .`), these commands are available:

| Command | Purpose |
|---------|---------|
| `slim-extract-text` | Extract + clean corpus from TXT/PDF/EPUB |
| `slim-train-tokenizer` | Train SentencePiece tokenizer |
| `slim-tokenize-parallel` | Convert corpus text to token IDs |
| `slim-train` | Train Transformer model on token IDs |
| `slim-inference` | Generate text from trained checkpoint |
| `slim-recommend` | Recommend model config from token count |

---

## 🚀 Quick Start (Guaranteed Sequential Pipeline)

This is the recommended **compatible** order:

1. **Preprocessing** (`slim-extract-text`)
2. **Tokenizer training** (`slim-train-tokenizer`)
3. **Tokenization** (`slim-tokenize-parallel`)
4. **Training** (`slim-train`)

### 1) Preprocess raw files

```bash
slim-extract-text \
  --input ./raw_data \
  --output ./artifacts/processed \
  --output-corpus ./artifacts/corpus.txt \
  --max-workers 8
```

Supported input formats: `.txt`, `.pdf`, `.epub`.

### 2) Train tokenizer

```bash
slim-train-tokenizer \
  --input ./artifacts/corpus.txt \
  --output ./artifacts/tokenizer \
  --model-prefix tokenizer \
  --vocab-size 16000
```

Produces:
- `./artifacts/tokenizer/tokenizer.model`
- `./artifacts/tokenizer/tokenizer.vocab`

### 3) Tokenize corpus to integer IDs

```bash
slim-tokenize-parallel \
  --input ./artifacts/corpus.txt \
  --model ./artifacts/tokenizer/tokenizer.model \
  --output ./artifacts/tokens.txt \
  --workers 8
```

Produces:
- `./artifacts/tokens.txt`
- `./artifacts/tokens.txt.meta.json` (contains tokenizer vocab metadata for compatibility checks)

### 4) Train model

```bash
slim-train \
  --config ./cfg/slim_cz_v1_default.yaml \
  --tokens ./artifacts/tokens.txt \
  --output ./output
```

> `slim-train` now performs an early compatibility check between tokenizer vocabulary (from `tokens.txt.meta.json`) and `model.vocab_size` in your YAML config. If they do not match, training stops with a clear error before wasting GPU/CPU time.

### 5) Inference

```bash
slim-inference \
  --checkpoint ./output/best_model.pt \
  --tokenizer ./artifacts/tokenizer/tokenizer.model
```

---

## ⚙️ Configuration System

SLiM-CZ-V1 uses YAML configuration files optimized for different dataset sizes based on Chinchilla scaling laws.

### Available Configurations

| Config | Parameters | Memory | Dataset Size | Vocab | Use Case | File |
|--------|-----------|--------|--------------|-------|----------|------|
| **Tiny** | ~3.6M | ~14 MB | < 2M tokens | 12k | Testing, limited data | `slim_cz_v1_tiny.yaml` |
| **Default** | ~7.2M | ~29 MB | 2-5M tokens | 16k | Standard (recommended) | `slim_cz_v1_default.yaml` |
| **Medium** | ~19.8M | ~79 MB | 5-15M tokens | 24k | Production, augmented data | `slim_cz_v1_medium.yaml` |

**Note:** All configurations use weight tying (saves ~30-40% parameters). Memory estimates for float32.

### Configuration Structure

Each YAML file contains:

```yaml
model:
  vocab_size: 16000       # Vocabulary size
  d_model: 256            # Embedding dimension
  num_heads: 8            # Attention heads
  num_layers: 4           # Transformer layers
  d_ff: 1024             # Feed-forward dimension
  max_seq_len: 512       # Max sequence length
  dropout: 0.25          # Dropout rate
  weight_tying: true     # Tie embeddings

train:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.05     # Aggressive for small data
  epochs: 30
  warmup_steps: 500
  scheduler: cosine
  gradient_clip: 1.0
  label_smoothing: 0.1   # Prevents overconfidence
  
generation:
  max_new_tokens: 100
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  repetition_penalty: 1.2
  
data:
  train_split: 0.90
  val_split: 0.05
  test_split: 0.05
  seq_len: 512
  stride: 256
```

### Using Configurations

```bash
# Use predefined config (recommended)
slim-train \
    --config ./cfg/slim_cz_v1_tiny.yaml \
    --tokens ./artifacts/tokens.txt \
    --output ./output

slim-train \
    --config ./cfg/slim_cz_v1_default.yaml \
    --tokens ./artifacts/tokens.txt \
    --output ./output

slim-train \
    --config ./cfg/slim_cz_v1_medium.yaml \
    --tokens ./artifacts/tokens.txt \
    --output ./output

# Use custom config file
slim-train \
    --config path/to/custom.yaml \
    --tokens ./artifacts/tokens.txt \
    --output ./output
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Preprocess source documents
slim-extract-text \
    --input ./czech_texts \
    --output ./artifacts/processed \
    --output-corpus ./artifacts/corpus.txt

# 2. Train tokenizer
slim-train-tokenizer \
    --input ./artifacts/corpus.txt \
    --output ./artifacts/tokenizer \
    --model-prefix tokenizer \
    --vocab-size 16000 \
    --character-coverage 0.9999

# 3. Tokenize corpus with trained tokenizer
slim-tokenize-parallel \
    --input ./artifacts/corpus.txt \
    --model ./artifacts/tokenizer/tokenizer.model \
    --output ./artifacts/tokens.txt \
    --workers 8

# 4. Train with selected configuration
slim-train \
    --config ./cfg/slim_cz_v1_default.yaml \
    --tokens ./artifacts/tokens.txt \
    --output ./output

# 5. Monitor training (optional)
tensorboard --logdir ./output/tensorboard

# 6. Test generation
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./artifacts/tokenizer/tokenizer.model \
    --prompt "Dnes je krásný den"
```

---

## 📖 Detailed Configuration Reference

### Model Architecture Comparison

| Parameter | Tiny | Default | Medium |
|-----------|------|---------|--------|
| **vocab_size** | 12,000 | 16,000 | 24,000 |
| **d_model** | 192 | 256 | 384 |
| **num_heads** | 6 | 8 | 8 |
| **num_layers** | 3 | 4 | 6 |
| **d_ff** | 768 | 1024 | 1536 |
| **max_seq_len** | 256 | 512 | 512 |
| **dropout** | 0.3 | 0.25 | 0.2 |
| **Parameters** | 3.6M | 7.2M | 19.8M |
| **Memory (fp32)** | 14 MB | 29 MB | 79 MB |

### Training Configuration

| Parameter | Tiny | Default | Medium |
|-----------|------|---------|--------|
| **batch_size** | 64 | 32 | 32 |
| **learning_rate** | 3e-4 | 1e-4 | 5e-5 |
| **epochs** | 40 | 30 | 25 |
| **weight_decay** | 0.1 | 0.05 | 0.03 |
| **warmup_steps** | 300 | 500 | 1000 |

### Chinchilla Scaling Guidance

The configurations follow research-backed scaling laws:

- **Rule**: ~100-150k parameters per 1M tokens for optimal training
- **Tiny** (3.6M params): Optimal for 1-2M tokens
- **Default** (7.2M params): Optimal for 2-5M tokens
- **Medium** (19.8M params): Optimal for 5-15M tokens

Using a model too large for your dataset leads to overfitting. Use `slim-recommend` to find the optimal match.

---

## 🔧 Data Preparation Pipeline

### Pipeline Steps

1. **File Collection** - Recursively scan directories for supported formats
2. **Text Cleaning** - Remove URLs, emails, normalize whitespace
3. **Tokenizer Training** - Train SentencePiece BPE tokenizer on processed corpus
4. **ID Tokenization** - Convert corpus lines to integer token IDs
5. **Compatibility Metadata** - Save tokenizer/model compatibility metadata for training checks

### Advanced Options

```bash
slim-extract-text \
    --input ./texts \
    --output ./artifacts/processed \
    --output-corpus ./artifacts/corpus.txt \
    --min-line-length 10

slim-train-tokenizer \
    --input ./artifacts/corpus.txt \
    --output ./artifacts/tokenizer \
    --model-prefix tokenizer \
    --vocab-size 16000

slim-tokenize-parallel \
    --input ./artifacts/corpus.txt \
    --model ./artifacts/tokenizer/tokenizer.model \
    --output ./artifacts/tokens.txt \
    --workers 8
```

---

## 🛠️ Diagnostic Tools

### 1. Configuration Recommendation

```bash
slim-recommend ./results/SLiM-CZ-V1-Dataset ./cfg
```

Analyzes your dataset and provides:
- Optimal model size based on token count
- Predicted validation loss and perplexity
- Data efficiency score
- Training time estimate
- Custom config generation if needed

### 2. Pipeline Compatibility Check

`slim-train` now validates compatibility before training starts:

- ✓ reads `tokens.txt.meta.json` generated by `slim-tokenize-parallel`
- ✓ compares tokenizer vocabulary size with `model.vocab_size` in config
- ✓ fails early with actionable message on mismatch

---

## 🎮 Inference Options

### Interactive Mode

```bash
slim-inference \
    --checkpoint model.pt \
    --tokenizer tokenizer.model
```

Commands:
- Enter any Czech text to generate continuation
- `settings` - Adjust temperature, top_k, max_tokens
- `quit` - Exit

### Batch Generation

```bash
slim-inference \
    --checkpoint model.pt \
    --tokenizer tokenizer.model \
    --prompt "Praha je" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-k 50
```

### Generation Parameters

- `--max-tokens` - Maximum tokens to generate (default: 100)
- `--temperature` - Sampling temperature, lower = more focused (0.1-2.0)
- `--top-k` - Top-k sampling, limits vocabulary (10-100)
- `--top-p` - Nucleus sampling (0.8-0.95)
- `--repetition-penalty` - Penalize repetition (1.0-1.5)

---

## 🔬 Research Optimizations

SLiM-CZ-V1 implements several research-backed optimizations for small datasets:

1. **Weight Tying**
   - Ties input and output embeddings
   - Saves ~30-40% parameters
   - No quality loss for small models

2. **Enhanced Dropout**
   - Higher rates (0.2-0.3) for small data
   - Prevents overfitting
   - Applied to attention and FFN

3. **Aggressive Weight Decay**
   - L2 regularization (0.03-0.1)
   - Stronger for smaller datasets
   - Improves generalization

4. **Label Smoothing** 
   - Smoothing factor 0.1-0.15
   - Prevents overconfidence
   - Better calibration on limited data

5. **Extended Training**
   - 25-40 epochs vs standard 10
   - Smaller datasets need more epochs
   - Early stopping prevents overfitting

6. **Chinchilla Scaling**
   - Match parameters to dataset size
   - Optimal: 1 param : 100-150 tokens
   - Prevents under/over-parameterization

7. **SentencePiece BPE**
   - Optimized for Czech morphology
   - Handles diacritics correctly
   - Better subword segmentation

---

## 🎯 Use Cases

- **Text Generation**: Generate coherent Czech text
- **Language Modeling**: Train on Czech corpus
- **Fine-tuning**: Adapt to specific domains (legal, medical, technical)
- **Research**: Experiment with transformer architectures
- **Education**: Learn about language models and training
- **Prototyping**: Quick experiments with Czech NLP

---

## 📊 Expected Performance

Based on Chinchilla scaling and empirical results:

### Tiny (3.6M params, < 2M tokens)
- **Validation Loss**: 2.5-3.0
- **Perplexity**: 12-20
- **Quality**: Basic Czech text, limited coherence
- **Use**: Testing, prototyping

### Default (7.2M params, 2-5M tokens)
- **Validation Loss**: 2.0-2.5
- **Perplexity**: 7-12
- **Quality**: Good Czech text, mostly correct grammar
- **Use**: Standard applications

### Medium (19.8M params, 5-15M tokens)
- **Validation Loss**: 1.8-2.2
- **Perplexity**: 6-9
- **Quality**: High-quality text, good coherence
- **Use**: Production, high-quality applications

---

## 🚨 Troubleshooting

### Model generates only unknown tokens (⇧)

**Diagnosis:**
- Check that `tokens.txt.meta.json` exists.
- Check that `model.vocab_size` in YAML matches tokenizer vocabulary.

**Common causes:**
- Model not trained (epoch 0)
- Wrong tokenizer
- Validation loss too high (> 5.0)

**Solutions:**
1. Train for full epochs (20-30)
2. Use smaller model for your dataset size
3. Check tokenizer was created correctly
4. Verify data quality

### Out of Memory

**Solutions:**
1. Reduce batch size: modify config file or use smaller batch
2. Use smaller model: `slim_cz_v1_tiny.yaml`
3. Reduce sequence length in config
4. Use gradient accumulation

### Overfitting (val loss increases)

**Solutions:**
1. Increase dropout (0.3-0.4)
2. Increase weight decay (0.1-0.2)
3. Use smaller model
4. Add more training data
5. Enable early stopping (default)

### Poor generation quality

**Solutions:**
1. Train longer (50+ epochs)
2. Use larger model if you have enough data
3. Collect more training data (target 2-5M tokens)
4. Adjust generation parameters (temperature, top_k)
5. Re-run tokenization and ensure compatible `vocab_size`

---

## 🔗 Links

- **Hugging Face Model**: [filipsedivy/SLiM-CZ-V1](https://huggingface.co/filipsedivy/SLiM-CZ-V1)
- **Hugging Face Dataset**: [filipsedivy/SLiM-CZ-V1](https://huggingface.co/datasets/filipsedivy/SLiM-CZ-V1)
- **Author**: [Filip Šedivy](https://huggingface.co/filipsedivy)

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👨‍💻 Contributing

Contributions welcome! This is a reference implementation for Czech language modeling with small datasets.

---

**Happy modeling! 🚀🇨🇿**
