"""
Training script for SLiM-CZ-V1
Slavic Linguistic integrated Micro-model for Czechia

Version: 0.2.0
Optimized for small Czech datasets (2-5M tokens) based on research recommendations.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from model import SLiM_CZ_V1
from dataloader import load_preprocessed_data, create_dataloaders


# ============================================================================
# ANSI Color Codes
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ============================================================================
# Progress Bar
# ============================================================================

class ProgressBar:
    """Simple progress bar for console output."""

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, n: int = 1, loss: float = None):
        """Update progress bar."""
        self.current += n
        current_time = time.time()

        # Update every 0.1 seconds to avoid flickering
        if current_time - self.last_update < 0.1 and self.current < self.total:
            return

        self.last_update = current_time
        self._draw(loss)

    def _draw(self, loss: float = None):
        """Draw progress bar."""
        if self.total == 0:
            return

        percent = min(self.current / self.total, 1.0)
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s" if eta > 0 else "ETA: 0s"
        else:
            eta_str = "ETA: --"

        loss_str = f"loss: {loss:.4f} " if loss is not None else ""

        sys.stdout.write(
            f'\r{self.desc} |{bar}| {self.current}/{self.total} '
            f'({percent*100:.0f}%) {loss_str}{eta_str}'
        )
        sys.stdout.flush()

        if self.current >= self.total:
            print()  # New line when complete

    def close(self):
        """Complete the progress bar."""
        self.current = self.total
        self._draw()


# ============================================================================
# Print Utilities
# ============================================================================

def print_header(text: str):
    """Print formatted header."""
    width = 80
    print(f"\n{'=' * width}")
    print(f"{text:^{width}}")
    print(f"{'=' * width}")


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 80)


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


# ============================================================================
# Loss Function
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing for regularization."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        log_probs = F.log_softmax(pred, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return (-true_dist * log_probs).sum(dim=-1).mean()


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0
):
    """
    Create cosine learning rate schedule with warmup.

    Based on research recommendations:
    - Linear warmup for first warmup_steps
    - Cosine decay to min_lr_ratio * initial_lr
    """

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)).item())
        )

    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Training manager for SLiM-CZ-V1."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config['train']
        self.epochs = train_cfg['epochs']
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.patience = train_cfg.get('patience', 5)
        self.min_delta = train_cfg.get('min_delta', 0.0005)

        # Loss function
        label_smoothing = train_cfg.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(smoothing=label_smoothing)
            print_info(f"Using label smoothing: {label_smoothing}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print_warning("Label smoothing disabled (recommended: 0.1)")

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.01),
            betas=tuple(train_cfg.get('betas', [0.9, 0.98])),
            eps=train_cfg.get('eps', 1e-9)
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs
        warmup_steps = train_cfg.get('warmup_steps', 500)
        min_lr = train_cfg.get('min_lr', 1e-6)
        min_lr_ratio = min_lr / train_cfg['learning_rate']

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )

        print_info(f"Scheduler: Cosine with warmup ({warmup_steps} steps)")
        print_info(f"Total training steps: {total_steps:,}")

        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD and train_cfg.get('use_tensorboard', False):
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
            print_success("TensorBoard logging enabled")

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rate': []
        }

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        # Progress tracking
        if HAS_TQDM:
            iterator = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.epochs}",
                ncols=100
            )
        else:
            print_info(f"Training epoch {epoch}/{self.epochs}...")
            pbar = ProgressBar(
                num_batches,
                desc=f"  Epoch {epoch}/{self.epochs}",
                width=40
            )
            iterator = self.train_loader

        for batch_idx, (input_ids, labels) in enumerate(iterator):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress
            if HAS_TQDM:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            else:
                pbar.update(1, loss=loss.item())

        if not HAS_TQDM:
            pbar.close()

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        if not HAS_TQDM:
            print_info("Evaluating...")
            pbar = ProgressBar(num_batches, desc="  Validation", width=40)

        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()

            if not HAS_TQDM:
                pbar.update(1, loss=loss.item())

        if not HAS_TQDM:
            pbar.close()

        return total_loss / num_batches

    def train(self):
        """Main training loop."""
        print_section("🚀 Starting Training")
        print_info(f"Training for {self.epochs} epochs")
        print_info(f"Device: {self.device}")
        print_info(f"Output directory: {self.output_dir}")
        print_info(f"Early stopping patience: {self.patience} epochs")
        print()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.evaluate()

            # Calculate perplexity
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            self.history['learning_rate'].append(lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Perplexity/train', train_ppl, epoch)
                self.writer.add_scalar('Perplexity/val', val_ppl, epoch)
                self.writer.add_scalar('Learning_Rate', lr, epoch)

            # Print epoch summary
            elapsed = time.time() - epoch_start
            print()
            print(f"{Colors.BOLD}Epoch {epoch}/{self.epochs} Summary{Colors.ENDC} (Time: {elapsed:.1f}s)")
            print("-" * 80)
            print(f"  Train Loss:      {train_loss:.4f}  |  Perplexity: {train_ppl:.2f}")
            print(f"  Validation Loss: {val_loss:.4f}  |  Perplexity: {val_ppl:.2f}")
            print(f"  Learning Rate:   {lr:.2e}")

            # Check for improvement
            if val_loss < self.best_val_loss - self.min_delta:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  {Colors.GREEN}✓ New best model! (↓ {improvement:.4f}){Colors.ENDC}")
            else:
                self.patience_counter += 1
                print(f"  {Colors.YELLOW}⚠ No improvement ({self.patience_counter}/{self.patience}){Colors.ENDC}")

                if self.patience_counter >= self.patience:
                    print()
                    print_warning(f"Early stopping triggered after {epoch} epochs")
                    print_info(f"Best validation loss: {self.best_val_loss:.4f}")
                    break

            print()

        # Save final model
        self.save_checkpoint('final_model.pt')
        self.save_history()

        if self.writer:
            self.writer.close()

        # Print final summary
        print_header("✅ Training Completed")
        print()
        print(f"  Total epochs trained: {len(self.history['train_loss'])}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Best validation PPL:  {torch.exp(torch.tensor(self.best_val_loss)):.2f}")
        print(f"  Models saved in:      {self.output_dir}")
        print()
        print_success("Training finished successfully!")
        print("=" * 80)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'global_step': self.global_step
        }
        torch.save(checkpoint, self.output_dir / filename)

    def save_history(self):
        """Save training history."""
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print_success("Training history saved: history.json")


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config(config: Dict[str, Any]):
    """
    Validate configuration against research recommendations.
    Print warnings for suboptimal settings.
    """
    print_section("🔍 Configuration Validation")

    warnings_found = False

    # Model parameters
    model_cfg = config['model']
    train_cfg = config['train']

    # Dropout
    dropout = model_cfg.get('dropout', 0.1)
    if dropout < 0.2:
        print_warning(f"Dropout ({dropout}) is low. Recommended: 0.2-0.3 for small datasets")
        warnings_found = True
    else:
        print_success(f"Dropout: {dropout} (optimal for small datasets)")

    # Weight decay
    weight_decay = train_cfg.get('weight_decay', 0.01)
    if weight_decay < 0.05:
        print_warning(f"Weight decay ({weight_decay}) is low. Recommended: 0.05-0.1")
        warnings_found = True
    else:
        print_success(f"Weight decay: {weight_decay} (aggressive regularization)")

    # Label smoothing
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    if label_smoothing < 0.1:
        print_warning(f"Label smoothing ({label_smoothing}) is low. Recommended: 0.1")
        warnings_found = True
    else:
        print_success(f"Label smoothing: {label_smoothing}")

    # Epochs
    epochs = train_cfg.get('epochs', 10)
    if epochs < 20:
        print_warning(f"Epochs ({epochs}) is low. Recommended: 20-50 for small datasets")
        warnings_found = True
    else:
        print_success(f"Epochs: {epochs} (sufficient for small datasets)")

    # Weight tying
    weight_tying = model_cfg.get('weight_tying', False)
    if not weight_tying:
        print_warning("Weight tying disabled. Recommended: enabled (saves ~30% params)")
        warnings_found = True
    else:
        print_success("Weight tying: enabled (optimal parameter efficiency)")

    if not warnings_found:
        print_success("All parameters optimally configured!")

    print()


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: Path, args) -> Dict[str, Any]:
    """Load and merge configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override with CLI arguments
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
    if args.dropout:
        config['model']['dropout'] = args.dropout
    if args.weight_decay:
        config['train']['weight_decay'] = args.weight_decay

    return config


def main():
    parser = argparse.ArgumentParser(
        description='🇨🇿 SLiM-CZ-V1 Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --data-dir ./data --config config.yaml
  python train.py --data-dir ./data --config config.yaml --epochs 30
  python train.py --data-dir ./data --config config.yaml --batch-size 64 --dropout 0.25
        """
    )

    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with preprocessed data (train.json, val.json, stats.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration YAML file'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for models and logs (default: ./output)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cuda, cpu (default: auto-detect)'
    )

    # Training overrides
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--dropout', type=float, help='Override dropout rate')
    parser.add_argument('--weight-decay', type=float, help='Override weight decay')

    args = parser.parse_args()

    # Print header
    print_header("🇨🇿 SLiM-CZ-V1 Training")
    print(f"{Colors.BOLD}Slavic Linguistic integrated Micro-model for Czechia{Colors.ENDC}".center(80))
    print(f"{Colors.BOLD}Optimized for small Czech datasets (2-5M tokens){Colors.ENDC}".center(80))

    # Load configuration
    print_section("📋 Loading Configuration")
    config_path = Path(args.config)
    if not config_path.exists():
        print_error(f"Configuration file not found: {args.config}")
        return

    config = load_config(config_path, args)
    print_success(f"Configuration loaded: {args.config}")

    # Validate configuration
    validate_config(config)

    # Load data
    print_section("📦 Loading Dataset")
    print_info(f"Data directory: {args.data_dir}")

    try:
        train_seqs, val_seqs, stats = load_preprocessed_data(args.data_dir)
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return

    print_success(f"Training sequences:   {len(train_seqs):,}")
    print_success(f"Validation sequences: {len(val_seqs):,}")
    print_info(f"Vocabulary size:      {stats['vocab_size']:,}")
    print_info(f"Sequence length:      {stats['seq_len']}")

    # Calculate dataset size
    total_tokens = len(train_seqs) * stats['seq_len']
    print_info(f"Total training tokens: {total_tokens:,} (~{total_tokens/1e6:.1f}M)")

    # Chinchilla scaling recommendations
    optimal_params = int(total_tokens / 1e6 * 125000)  # 100-150k per 1M tokens
    print_info(f"Optimal params (Chinchilla): ~{optimal_params/1e6:.1f}M")
    print()

    # Create dataloaders
    print_section("🔄 Creating DataLoaders")
    batch_size = config['train']['batch_size']

    train_loader, val_loader = create_dataloaders(
        train_seqs,
        val_seqs,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    print_success(f"Batch size: {batch_size}")
    print_info(f"Training batches:   {len(train_loader):,}")
    print_info(f"Validation batches: {len(val_loader):,}")
    print()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_section("💻 Device Configuration")
    print_info(f"Device: {device}")
    if device.type == 'cuda':
        print_info(f"GPU: {torch.cuda.get_device_name(0)}")
        print_info(f"CUDA version: {torch.version.cuda}")
    print()

    # Create model
    print_section("🏗️  Building Model")
    print_info("Creating SLiM-CZ-V1...")

    model = SLiM_CZ_V1(
        vocab_size=stats['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=stats['seq_len'],
        dropout=config['model']['dropout'],
        weight_tying=config['model'].get('weight_tying', True)
    ).to(device)

    params = model.count_parameters()
    total_params = params['total']

    print_success("Model created successfully!")
    print()
    print(f"  Architecture:")
    print(f"    Layers:      {config['model']['num_layers']}")
    print(f"    Heads:       {config['model']['num_heads']}")
    print(f"    Embedding:   {config['model']['d_model']}")
    print(f"    FFN:         {config['model']['d_ff']}")
    print(f"    Dropout:     {config['model']['dropout']}")
    print()
    print(f"  Parameters:")
    print(f"    Total:       {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"    Trainable:   {params['trainable']:,}")

    if model.weight_tying:
        print(f"    Saved (tying): {params['saved']:,} ({params['saved']/total_params*100:.1f}%)")
        print_success("Weight tying enabled (efficient!)")

    # Check if model size is reasonable
    print()
    if total_params > optimal_params * 10:
        print_warning(
            f"Model ({total_params/1e6:.1f}M params) may be too large for dataset "
            f"({total_tokens/1e6:.1f}M tokens). Consider reducing layers/dimensions."
        )
    elif total_params > optimal_params * 3:
        print_info(
            f"Model size acceptable with aggressive regularization. "
            f"Monitor for overfitting."
        )
    else:
        print_success(f"Model size optimal for dataset!")
    print()

    # Print research-based optimizations
    print_section("✨ Research-Based Optimizations")
    print(f"   {Colors.GREEN}✓{Colors.ENDC} SentencePiece BPE tokenizer (Czech-optimized)")
    print(f"   {Colors.GREEN}✓{Colors.ENDC} Label smoothing (prevents overconfidence)")
    print(f"   {Colors.GREEN}✓{Colors.ENDC} Cosine LR schedule with warmup")
    print(f"   {Colors.GREEN}✓{Colors.ENDC} Gradient clipping (stability)")
    print(f"   {Colors.GREEN}✓{Colors.ENDC} Early stopping (prevents overfitting)")
    if config['model'].get('weight_tying', False):
        print(f"   {Colors.GREEN}✓{Colors.ENDC} Weight tying (parameter efficiency)")
    if config['model']['dropout'] >= 0.2:
        print(f"   {Colors.GREEN}✓{Colors.ENDC} Aggressive dropout (regularization)")
    if config['train'].get('weight_decay', 0) >= 0.05:
        print(f"   {Colors.GREEN}✓{Colors.ENDC} Strong weight decay (regularization)")
    print()

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=Path(args.output_dir)
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print()
        print_warning("Training interrupted by user")
        trainer.save_checkpoint('interrupted.pt')
        print_success("Model checkpoint saved: interrupted.pt")
    except Exception as e:
        print()
        print_error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()