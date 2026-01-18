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

After installing the package with `pip install -e .` (or `uv pip install -e .`), five CLI commands become available in your environment. These provide a convenient way to work with SLiM-CZ-V1 without directly calling Python scripts.

### Available Commands

| Command | Purpose |
|---------|---------|
| `slim-prepare-data` | Prepare and tokenize training data |
| `slim-recommend` | Get optimal configuration recommendations |
| `slim-train` | Train the language model |
| `slim-inference` | Generate text with trained model |
| `slim-diagnose` | Diagnose model and training issues |

### Usage Examples

#### 1. Data Preparation

```bash
# Using CLI command (recommended after installation)
slim-prepare-data \
    --input ./raw_texts \
    --output ./data \
    --vocab-size 16000 \
    --seq-len 512

# Or using Python script directly
python prepare_data.py --input ./raw_texts --output ./data --vocab-size 16000 --seq-len 512
```

**Options:**
- `--input, -i` (required): Input directory with text files
- `--output, -o` (required): Output directory for processed data
- `--vocab-size`: Vocabulary size (default: 16000)
- `--seq-len`: Sequence length (default: 512)
- `--stride`: Stride for sequences (default: 256)
- `--train-split`: Training split ratio (default: 0.90)
- `--val-split`: Validation split ratio (default: 0.05)
- `--test-split`: Test split ratio (default: 0.05)
- `--remove-urls`: Remove URLs from text (default: enabled)
- `--remove-emails`: Remove emails from text (default: enabled)
- `--min-line-length`: Minimum line length (default: 10)

#### 2. Configuration Recommendation

```bash
# Using CLI command
slim-recommend --data-dir ./data

# Or using Python script
python recommend_config.py ./data
```

**Arguments:**
- `data_dir` (required): Path to prepared data directory
- `config_dir` (optional): Directory for config files (default: current directory)

#### 3. Model Training

```bash
# Using CLI command
slim-train \
    --config slim_cz_v1_default.yaml \
    --data-dir ./data \
    --output-dir ./output \
    --tokenizer ./data/tokenizer.model

# Or using Python script
python train.py --config slim_cz_v1_default.yaml --data-dir ./data --output-dir ./output
```

**Options:**
- `--config` (required): Path to configuration YAML file
- `--data-dir` (required): Path to prepared data directory
- `--output-dir` (required): Output directory for checkpoints and logs
- `--tokenizer`: Path to tokenizer for sample generation (optional)
- `--no-tensorboard`: Disable TensorBoard logging (optional)

#### 4. Text Inference/Generation

```bash
# Interactive mode (recommended)
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./data/tokenizer.model

# Single prompt generation
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./data/tokenizer.model \
    --prompt "Praha je" \
    --max-tokens 100 \
    --temperature 0.8

# Batch generation from file
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./data/tokenizer.model \
    --prompts-file ./prompts.txt \
    --output ./results.txt

# Or using Python script
python inference.py --checkpoint ./output/best_model.pt --tokenizer ./data/tokenizer.model
```

**Options:**
- `--checkpoint` (required): Path to model checkpoint (.pt file)
- `--tokenizer` (required): Path to tokenizer model (.model file)
- `--prompt`: Single prompt for generation (optional)
- `--prompts-file`: File with prompts, one per line (optional)
- `--output`: Output file for batch generation (optional)
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-k`: Top-k sampling (default: 50)
- `--top-p`: Nucleus sampling threshold (default: 0.95)
- `--repetition-penalty`: Repetition penalty (default: 1.2)
- `--no-sample`: Use greedy decoding instead of sampling
- `--device`: Device to use: cuda or cpu (default: auto-detect)

#### 5. Model Diagnosis

```bash
# Using CLI command
slim-diagnose ./output/best_model.pt ./data/tokenizer.model ./data

# Or using Python script
python diagnose.py ./output/best_model.pt ./data/tokenizer.model ./data
```

**Arguments:**
- `checkpoint_path` (required): Path to model checkpoint
- `tokenizer_path` (required): Path to tokenizer model
- `data_dir` (optional): Path to data directory for validation

### CLI vs Direct Python Scripts

Both approaches work identically, but CLI commands offer advantages:

**CLI Commands (`slim-*`):**
- ✅ Available system-wide after installation
- ✅ Shorter, cleaner syntax
- ✅ Tab completion support (in some shells)
- ✅ No need to remember script locations
- ✅ Professional deployment ready

**Direct Python Scripts (`python *.py`):**
- ✅ Works without package installation
- ✅ Easier for development/debugging
- ✅ More transparent about execution

**Recommendation:** Use CLI commands for production workflows and direct scripts for development.

---

## 🚀 Quick Start

### 1. Prepare Your Data

```bash
slim-prepare-data \
    --input ./raw_texts \
    --output ./data \
    --vocab-size 16000 \
    --seq-len 512
```

**Supported file formats:** `.txt`, `.md`, `.rst`, `.py`, `.js`, `.html`, `.css`, `.json`, `.xml`, `.csv`, `.log`

This creates:
- `data/train.json` - Training sequences
- `data/val.json` - Validation sequences
- `data/tokenizer.model` - SentencePiece tokenizer
- `data/stats.json` - Dataset statistics

### 2. Recommend Optimal Configuration

```bash
slim-recommend --data-dir ./data
```

This tool analyzes your dataset and recommends the best model configuration based on:
- Dataset size (number of tokens)
- Chinchilla scaling laws
- Predicted validation loss
- Data efficiency score

### 3. Train with Configuration

```bash
# Train with recommended config
slim-train \
    --data-dir ./data \
    --config slim_cz_v1_default.yaml \
    --output-dir ./output
```

The training script:
- Loads data and configuration
- Initializes model with specified architecture
- Trains with AdamW optimizer and cosine LR schedule
- Saves checkpoints and best model
- Logs metrics to TensorBoard (optional)
- Implements early stopping

### 4. Run Inference

```bash
# Interactive mode
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./data/tokenizer.model

# One-time generation
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./data/tokenizer.model \
    --prompt "Praha je" \
    --max-tokens 100
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
    --data-dir ./data \
    --config slim_cz_v1_tiny.yaml \
    --output-dir ./output

slim-train \
    --data-dir ./data \
    --config slim_cz_v1_default.yaml \
    --output-dir ./output

slim-train \
    --data-dir ./data \
    --config slim_cz_v1_medium.yaml \
    --output-dir ./output

# Use custom config file
slim-train \
    --data-dir ./data \
    --config path/to/custom.yaml \
    --output-dir ./output
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Prepare your text data (Czech text files)
slim-prepare-data \
    --input ./czech_texts \
    --output ./data \
    --vocab-size 16000 \
    --seq-len 512

# 2. Analyze dataset and get recommendation
slim-recommend --data-dir ./data

# 3. Train with recommended configuration
slim-train \
    --data-dir ./data \
    --config slim_cz_v1_default.yaml \
    --output-dir ./output

# 4. Monitor training (in another terminal)
tensorboard --logdir ./output/tensorboard

# 5. Test generation
slim-inference \
    --checkpoint ./output/best_model.pt \
    --tokenizer ./data/tokenizer.model \
    --prompt "Dnes je krásný den"

# 6. Diagnose model quality
slim-diagnose \
    ./output/best_model.pt \
    ./data/tokenizer.model \
    ./data
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
3. **Tokenization** - Train SentencePiece BPE tokenizer optimized for Czech
4. **Sequence Creation** - Create overlapping sequences with configurable stride
5. **Dataset Split** - Split into train/validation/test sets

### Advanced Options

```bash
slim-prepare-data \
    --input ./texts \
    --output ./data \
    --vocab-size 16000 \
    --seq-len 512 \
    --stride 256 \
    --train-split 0.9 \
    --val-split 0.05 \
    --test-split 0.05 \
    --min-line-length 10
```

---

## 🛠️ Diagnostic Tools

### 1. Configuration Recommendation

```bash
slim-recommend --data-dir ./data
```

Analyzes your dataset and provides:
- Optimal model size based on token count
- Predicted validation loss and perplexity
- Data efficiency score
- Training time estimate
- Custom config generation if needed

### 2. Model Diagnosis

```bash
slim-diagnose \
    ./output/best_model.pt \
    ./data/tokenizer.model \
    ./data
```

Performs comprehensive checks:
- ✓ Checkpoint integrity (epoch, loss, weights)
- ✓ Tokenizer quality (vocab size, Czech encoding)
- ✓ Training data statistics
- ✓ Generation test with sample prompts
- ✓ Identifies common issues (untrained model, high loss, etc.)

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
```bash
slim-diagnose model.pt tokenizer.model data/
```

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
5. Check tokenizer quality with `slim-diagnose`

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