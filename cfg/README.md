# SLiM-CZ-V1 Configuration Files

**Slavic Linguistic integrated Micro-model for Czechia**

This directory contains YAML configuration files for different model sizes optimized for Czech language modeling with small datasets (2-5M tokens).

---

## 📋 Available Configurations

| File | Model Size | Parameters | Memory | Dataset Size | Use Case |
|------|------------|------------|--------|--------------|----------|
| `slim_cz_v1_tiny.yaml` | Tiny | ~3.6M | ~14 MB | < 2M tokens | Testing, debugging, resource-constrained |
| `slim_cz_v1_default.yaml` | Default | ~7.2M | ~29 MB | 2-5M tokens | Standard training (recommended) |
| `slim_cz_v1_medium.yaml` | Medium | ~19.8M | ~79 MB | 5-15M tokens | Production, augmented data |

**Note**: All configurations use weight tying to reduce parameters by ~30-40%. Memory estimates are for float32 precision.

---

## 🎯 Usage

### From Command Line

```bash
# Use predefined config (recommended approach)
python train.py --data-dir ./data --config slim_cz_v1_tiny.yaml
python train.py --data-dir ./data --config slim_cz_v1_default.yaml
python train.py --data-dir ./data --config slim_cz_v1_medium.yaml

# Default config is used if not specified
python train.py --data-dir ./data
```

### From Python

```python
import yaml
from model import SLiM_CZ_V1

# Load config
with open('slim_cz_v1_default.yaml') as f:
    config = yaml.safe_load(f)

# Access configuration
vocab_size = config['model']['vocab_size']
batch_size = config['train']['batch_size']

# Create model
model = SLiM_CZ_V1(**config['model'])
```

---

## 📁 Configuration Structure

Each YAML file contains four main sections:

### 1. Model Architecture

```yaml
model:
  vocab_size: 16000      # Vocabulary size (SentencePiece BPE)
  d_model: 256           # Embedding dimension
  num_heads: 8           # Number of attention heads
  num_layers: 4          # Number of transformer layers
  d_ff: 1024            # Feed-forward dimension (typically 4x d_model)
  max_seq_len: 512      # Maximum sequence length
  dropout: 0.25         # Dropout rate (higher for regularization)
  weight_tying: true    # Tie input/output embeddings (saves ~30% params)
```

### 2. Training Parameters

```yaml
train:
  # Optimization
  batch_size: 32                # Training batch size
  learning_rate: 0.0001        # Initial learning rate
  weight_decay: 0.05           # L2 regularization (aggressive for small data)
  betas: [0.9, 0.98]          # Adam optimizer betas
  eps: 1.0e-9                 # Adam epsilon
  
  # Learning rate schedule
  warmup_steps: 500            # Warmup steps
  scheduler: cosine            # LR scheduler type
  min_lr: 1.0e-6              # Minimum learning rate
  
  # Regularization
  gradient_clip: 1.0           # Gradient clipping threshold
  label_smoothing: 0.1         # Label smoothing (prevents overconfidence)
  
  # Training duration
  epochs: 30                   # Number of training epochs
  max_steps: 100000           # Maximum training steps
  
  # Logging & Checkpointing
  log_every: 50               # Log every N steps
  eval_every: 500             # Evaluate every N steps
  save_every: 2000            # Save checkpoint every N steps
  
  # Early Stopping
  patience: 8                 # Early stopping patience
  min_delta: 0.0005          # Minimum improvement delta
  
  # Optional monitoring
  use_tensorboard: false      # Enable TensorBoard logging
```

### 3. Generation Parameters

```yaml
generation:
  max_new_tokens: 100         # Maximum tokens to generate
  temperature: 0.8            # Sampling temperature (lower = more focused)
  top_k: 50                   # Top-k sampling
  top_p: 0.95                # Nucleus (top-p) sampling
  repetition_penalty: 1.2    # Repetition penalty
  do_sample: true            # Use sampling (vs greedy decoding)
```

### 4. Data Configuration

```yaml
data:
  # Dataset splits
  train_split: 0.90          # Training data ratio
  val_split: 0.05            # Validation data ratio
  test_split: 0.05           # Test data ratio
  
  # Sequence configuration
  seq_len: 512              # Sequence length for training
  stride: 256               # Stride for overlapping sequences
  
  # Text Processing
  lowercase: false           # Convert to lowercase
  remove_urls: true         # Remove URLs
  remove_emails: true       # Remove email addresses
  min_line_length: 10       # Minimum line length
  min_frequency: 2          # Minimum token frequency
```

---

## 🔧 Creating Custom Configuration

### Option 1: Copy and Modify

```bash
# Copy existing config
cp slim_cz_v1_default.yaml my_config.yaml

# Edit with your parameters
nano my_config.yaml

# Use it
python train.py --data-dir ./data --config my_config.yaml
```

### Option 2: Create from Scratch

```yaml
# my_config.yaml
model:
  vocab_size: 20000
  d_model: 320
  num_heads: 8
  num_layers: 5
  d_ff: 1280
  max_seq_len: 512
  dropout: 0.2
  weight_tying: true

train:
  batch_size: 32
  learning_rate: 0.00008
  weight_decay: 0.03
  betas: [0.9, 0.98]
  eps: 1.0e-9
  warmup_steps: 500
  scheduler: cosine
  min_lr: 1.0e-6
  gradient_clip: 1.0
  label_smoothing: 0.1
  epochs: 30
  max_steps: 100000
  log_every: 50
  eval_every: 500
  save_every: 2000
  patience: 8
  min_delta: 0.0005
  use_tensorboard: false

generation:
  max_new_tokens: 100
  temperature: 0.8
  top_k: 50
  top_p: 0.95
  repetition_penalty: 1.2
  do_sample: true

data:
  train_split: 0.90
  val_split: 0.05
  test_split: 0.05
  seq_len: 512
  stride: 256
  lowercase: false
  remove_urls: true
  remove_emails: true
  min_line_length: 10
  min_frequency: 2
```

---

## 💡 Configuration Tips

### Model Size Guidelines

**Tiny** (`slim_cz_v1_tiny.yaml`) - Use when:
- Dataset < 2M tokens
- Testing code changes
- Debugging training pipeline
- Quick experiments
- Limited GPU memory (< 4GB)
- Fast iteration needed

**Default** (`slim_cz_v1_default.yaml`) - Use when:
- Dataset 2-5M tokens (recommended)
- Starting development
- Standard Czech text generation
- 4-8GB GPU memory
- Balanced quality/speed

**Medium** (`slim_cz_v1_medium.yaml`) - Use when:
- Dataset 5-15M tokens
- After data augmentation
- Production deployment
- Higher quality needed
- 8-16GB GPU memory

### Chinchilla Scaling Laws

The configurations follow Chinchilla scaling recommendations:
- **Rule of thumb**: ~100-150k parameters per 1M tokens
- **Tiny**: 3.6M params → optimal for 1-2M tokens
- **Default**: 7.2M params → optimal for 2-5M tokens
- **Medium**: 19.8M params → optimal for 5-15M tokens

### Parameter Relationships

**d_model and num_heads:**
- `d_model` must be divisible by `num_heads`
- Each head gets `d_model / num_heads` dimensions
- Standard: 8 heads, 256 dimensions = 32 dim/head

**d_ff (Feed-Forward):**
- Typically 4x `d_model`
- Can range from 2x to 8x
- Larger = more capacity but slower

**seq_len and stride:**
- `stride = seq_len / 2` is common (50% overlap)
- Smaller stride = more sequences (slower, more data)
- Larger stride = fewer sequences (faster, less data)

**batch_size and GPU memory:**
- Larger model → smaller batch size needed
- Longer sequences → smaller batch size needed
- Rule of thumb: batch_size ∝ 1 / (params × seq_len)

**dropout and regularization:**
- Small data → higher dropout (0.25-0.3)
- Large data → lower dropout (0.1-0.15)
- weight_decay should be higher for small datasets

---

## 🎯 Recommended Configurations

### For Learning / Testing

```bash
# Quick test with tiny model
python train.py --data-dir ./data --config slim_cz_v1_tiny.yaml --epochs 5
```

### For Development

```bash
# Standard development with default config
python train.py --data-dir ./data --config slim_cz_v1_default.yaml --epochs 30
```

### For Production

```bash
# Production with medium model (requires more data)
python train.py --data-dir ./data --config slim_cz_v1_medium.yaml --epochs 25
```

---

## 📊 Configuration Comparison

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
| **batch_size** | 64 | 32 | 32 |
| **learning_rate** | 3e-4 | 1e-4 | 5e-5 |
| **epochs** | 40 | 30 | 25 |
| **weight_decay** | 0.1 | 0.05 | 0.03 |
| **Recommended Data** | < 2M tokens | 2-5M tokens | 5-15M tokens |

---

## 🔬 Research Optimizations Applied

All configurations implement research-backed optimizations for small datasets:

1. **Weight Tying**: Saves ~30-40% parameters without quality loss
2. **Enhanced Dropout**: Higher rates (0.2-0.3) prevent overfitting on small data
3. **Aggressive Weight Decay**: L2 regularization (0.03-0.1) for better generalization
4. **Label Smoothing**: (0.1-0.15) prevents overconfidence on limited data
5. **Extended Training**: More epochs (25-40) to fully utilize small datasets
6. **SentencePiece BPE**: Optimized for Czech morphology
7. **Chinchilla Scaling**: Parameter count matched to dataset size

---

## 🛠️ Diagnostic Tools

### Check Configuration

```bash
# View configuration
python -c "
import yaml
with open('slim_cz_v1_default.yaml') as f:
    config = yaml.safe_load(f)
    print(yaml.dump(config, default_flow_style=False))
"
```

### Calculate Parameters

```bash
# Calculate exact parameter count
python -c "
import yaml
with open('slim_cz_v1_default.yaml') as f:
    cfg = yaml.safe_load(f)['model']
    
vocab_emb = cfg['vocab_size'] * cfg['d_model']
per_layer = (cfg['d_model'] * cfg['d_model'] * 4 + 
             cfg['d_model'] * cfg['d_ff'] * 2 + 
             cfg['d_model'] * 4)
total = vocab_emb + per_layer * cfg['num_layers'] + cfg['d_model'] * 2
print(f'Parameters: {total:,} (~{total/1e6:.1f}M)')
"
```

### Diagnose Model

```bash
# Run full diagnostic
python diagnose.py output/best_model.pt data/tokenizer.model data/
```

---

## 📝 Notes

- **Default is recommended** for most Czech language modeling tasks with 2-5M tokens
- **Tiny** is best for experimentation and testing with limited data/compute
- **Medium** requires augmented data (back-translation, synthesis) to reach its potential
- All configs use **weight tying** - do not disable unless you have a specific reason
- For datasets > 15M tokens, consider creating a custom larger configuration

---

## 🆘 Common Issues

### Out of Memory

```bash
# Reduce batch size or use smaller model
python train.py --config slim_cz_v1_tiny.yaml
# or
python train.py --config slim_cz_v1_default.yaml  # but edit batch_size to 16
```

### Overfitting

- Increase dropout (0.3-0.4)
- Increase weight_decay (0.1-0.2)
- Use smaller model (tiny instead of default)
- Add more training data

### Poor Quality

- Train longer (50+ epochs)
- Use larger model (medium instead of default)
- Collect more data (target 2-5M tokens)
- Check tokenizer quality with `diagnose.py`