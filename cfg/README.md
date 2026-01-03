# SLiM-CZ-V1 Configuration Files

**Slavic Linguistic integrated Micro-model for Czechia**

This directory contains YAML configuration files for different model sizes, similar to the ultralytics approach.

---

## 📋 Available Configurations

| File | Model Size | Parameters | Memory | Use Case |
|------|------------|------------|--------|----------|
| `tiny.yaml` | Tiny | ~3M | ~12 MB | Testing, debugging |
| `small.yaml` | Small | ~14M | ~56 MB | Development (default) |
| `medium.yaml` | Medium | ~56M | ~224 MB | Production |
| `large.yaml` | Large | ~125M | ~500 MB | High-end production |
| `default.yaml` | Default | ~14M | ~56 MB | Same as small |

---

## 🎯 Usage

### From Command Line

```bash
# Use predefined config
python train.py --data-dir ./data --config tiny
python train.py --data-dir ./data --config small
python train.py --data-dir ./data --config medium
python train.py --data-dir ./data --config large

# Default (small) is used if no config specified
python train.py --data-dir ./data
```

### From Python

```python
from config_loader import load_config

# Load config
config = load_config('small')

# Access configuration
vocab_size = config['model']['vocab_size']
batch_size = config['train']['batch_size']

# Create model
from slim_cz_v1 import SLiM_CZ_V1
model = SLiM_CZ_V1(**config['model'])
```

---

## 📝 Configuration Structure

Each YAML file contains four main sections:

### 1. Model Architecture

```yaml
model:
  vocab_size: 10000      # Vocabulary size
  d_model: 256           # Embedding dimension
  num_heads: 8           # Number of attention heads
  num_layers: 6          # Number of transformer layers
  d_ff: 1024            # Feed-forward dimension
  max_seq_len: 512      # Maximum sequence length
  dropout: 0.1          # Dropout rate
```

### 2. Training Parameters

```yaml
train:
  batch_size: 32                # Training batch size
  learning_rate: 0.0001        # Learning rate
  weight_decay: 0.01           # Weight decay for regularization
  epochs: 10                   # Number of training epochs
  scheduler: "cosine"          # Learning rate scheduler
  gradient_clip: 1.0           # Gradient clipping threshold
  
  # Logging & Checkpointing
  log_every: 100              # Log every N steps
  eval_every: 1000            # Evaluate every N steps
  save_every: 5000            # Save checkpoint every N steps
  
  # Early Stopping
  patience: 5                 # Early stopping patience
  min_delta: 0.001           # Minimum improvement delta
```

### 3. Generation Parameters

```yaml
generation:
  max_new_tokens: 100         # Maximum tokens to generate
  temperature: 0.8            # Sampling temperature
  top_k: 50                   # Top-k sampling
  top_p: 0.95                # Nucleus (top-p) sampling
  repetition_penalty: 1.2    # Repetition penalty
  do_sample: true            # Use sampling (vs greedy)
```

### 4. Data Configuration

```yaml
data:
  train_split: 0.90          # Training data ratio
  val_split: 0.05            # Validation data ratio
  test_split: 0.05           # Test data ratio
  seq_len: 512              # Sequence length
  stride: 256               # Stride for sequence creation
  
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
cp cfg/small.yaml cfg/my_config.yaml

# Edit with your parameters
nano cfg/my_config.yaml

# Use it
python train.py --data-dir ./data --config cfg/my_config.yaml
```

### Option 2: Create from Scratch

```yaml
# cfg/custom.yaml
model:
  vocab_size: 20000
  d_model: 384
  num_heads: 8
  num_layers: 8
  d_ff: 1536
  max_seq_len: 1024
  dropout: 0.1

train:
  batch_size: 24
  learning_rate: 0.00008
  epochs: 15
  scheduler: "cosine"
  gradient_clip: 1.0

generation:
  max_new_tokens: 150
  temperature: 0.8
  top_k: 50

data:
  train_split: 0.90
  val_split: 0.05
  test_split: 0.05
  seq_len: 1024
  stride: 512

info:
  name: "SLiM-CZ-V1-Custom"
  description: "My custom configuration"
```

---

## 💡 Configuration Tips

### Model Size Guidelines

**Tiny** - Use when:
- Testing code changes
- Debugging training pipeline
- Quick experiments
- Limited GPU memory

**Small** - Use when:
- Starting development
- Learning the system
- Medium-sized datasets
- 4-8GB GPU memory

**Medium** - Use when:
- Production deployment
- Large datasets
- Quality is important
- 12-16GB GPU memory

**Large** - Use when:
- Research projects
- Maximum quality needed
- Very large datasets
- 24GB+ GPU memory

### Parameter Relationships

**d_model and num_heads:**
- d_model must be divisible by num_heads
- Each head gets d_model / num_heads dimensions

**d_ff (Feed-Forward):**
- Typically 4x d_model
- Can range from 2x to 8x

**seq_len and stride:**
- stride = seq_len / 2 is common
- Smaller stride = more sequences
- Larger stride = faster processing

**batch_size and GPU memory:**
- Larger model → smaller batch size
- Longer sequences → smaller batch size
- More memory → larger batch size

---

## 🎯 Recommended Configurations

### For Learning

```yaml
# Use tiny or small
python train.py --data-dir ./data --config tiny --epochs 5
```

### For Development

```yaml
# Use small with overrides
python train.py --data-dir ./data --config small --epochs 10
```

### For Production

```yaml
# Use medium or large
python train.py --data-dir ./data --config medium --epochs 30
```

---

## 📊 Configuration Comparison

| Parameter | Tiny | Small | Medium | Large |
|-----------|------|-------|--------|-------|
| **vocab_size** | 10k | 10k | 30k | 50k |
| **d_model** | 128 | 256 | 512 | 768 |
| **num_heads** | 4 | 8 | 8 | 12 |
| **num_layers** | 4 | 6 | 8 | 12 |
| **d_ff** | 512 | 1024 | 2048 | 3072 |
| **max_seq_len** | 256 | 512 | 1024 | 2048 |
| **batch_size** | 64 | 32 | 16 | 8 |
| **epochs** | 5 | 10 | 20 | 30 |

---

## 🔍 Viewing Configurations

```bash
# View all available configs
python config_loader.py

# Load and display specific config
python -c "
from config_loader import load_config, ConfigLoader

loader = ConfigLoader()
config = load_config('small')
loader.print_config(config)
"
```