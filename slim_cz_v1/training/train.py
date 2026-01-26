"""
Core training module for SLiM-CZ-V1 language model.

Provides training infrastructure including:
- Trainer class with TensorBoard support
- Data loading utilities
- Training/validation loops
- Checkpoint management
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..utils import console

try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ============================================================
# TENSORBOARD LOGGER
# ============================================================

class TensorBoardLogger:
    """TensorBoard logger for training metrics."""

    def __init__(self, log_dir: Path, enabled: bool = True):
        self.enabled = enabled and HAS_TENSORBOARD
        self.writer = None

        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        if not self.enabled:
            return

        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)

    def log_lr(self, lr: float, step: int):
        if self.enabled:
            self.writer.add_scalar('Training/learning_rate', lr, step)

    def log_config(self, config: Dict):
        if self.enabled:
            import yaml
            config_text = yaml.dump(config, default_flow_style=False)
            self.writer.add_text('Config/full_config', f"```yaml\n{config_text}\n```", 0)

    def close(self):
        if self.enabled and self.writer:
            self.writer.close()


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for tokenized sequences."""

    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]


# ============================================================
# DATA LOADING
# ============================================================

def load_tokenized_data(
        tokens_path: Path,
        seq_len: int,
        vocab_size: int,
        val_split: float = 0.1
) -> Tuple[SequenceDataset, SequenceDataset, Dict]:
    """Load tokenized data and prepare sequences."""

    console.verbose(f"Loading tokenized data from: {tokens_path}")

    all_tokens = []
    line_count = 0

    with open(tokens_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = [int(t) for t in line.strip().split()]
            all_tokens.extend(tokens)
            line_count += 1

            if line_count % 10000 == 0:
                console.heartbeat(f"Read {line_count} lines...")

    console.verbose(f"Loaded {len(all_tokens):,} tokens from {line_count:,} lines")

    num_sequences = len(all_tokens) // seq_len
    tokens_trimmed = all_tokens[:num_sequences * seq_len]

    sequences = torch.tensor(tokens_trimmed, dtype=torch.long).reshape(num_sequences, seq_len)

    val_size = int(num_sequences * val_split)
    train_size = num_sequences - val_size

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)

    stats = {
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'total_tokens': len(all_tokens),
        'train_tokens': len(train_sequences) * seq_len,
        'val_tokens': len(val_sequences) * seq_len,
    }

    console.verbose(f"Train: {stats['train_sequences']:,} | Val: {stats['val_sequences']:,} sequences")

    return train_dataset, val_dataset, stats


def create_dataloaders(
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset,
        batch_size: int,
        num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""

    # Auto-adjust for notebook environments
    if console.in_notebook or console.in_kaggle:
        if num_workers > 0:
            console.verbose(f"Notebook detected - setting num_workers=0")
            num_workers = 0

    console.verbose(f"Creating dataloaders: batch_size={batch_size}, num_workers={num_workers}")

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0
    }

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )

    console.verbose(f"Dataloaders: {len(train_loader)} train / {len(val_loader)} val batches")

    return train_loader, val_loader


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """
    Model trainer with TensorBoard logging.

    Features:
    - Automatic device selection
    - Learning rate warmup
    - Gradient clipping
    - Checkpoint management
    - TensorBoard logging
    - Verbose mode support via console

    Verbose mode can be enabled via:
    - console.set_verbose(True) before training
    - Environment variable: SLIM_VERBOSE=1
    - CLI flag: slim-train --verbose
    """

    def __init__(
            self,
            config: Dict,
            output_dir: Path,
            enable_tensorboard: bool = True,
            use_tqdm: bool = True
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            output_dir: Output directory for checkpoints
            enable_tensorboard: Enable TensorBoard logging
            use_tqdm: Use tqdm progress bars (auto-disabled in verbose mode)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_tqdm = use_tqdm and HAS_TQDM

        # TensorBoard
        self.tb_logger = TensorBoardLogger(
            self.output_dir,
            enabled=enable_tensorboard and config.get('train', {}).get('use_tensorboard', True)
        )

        # Training parameters
        train_cfg = config.get('train', {})
        self.epochs = train_cfg.get('epochs', 30)
        self.batch_size = train_cfg.get('batch_size', 32)
        self.learning_rate = train_cfg.get('learning_rate', 0.0001)
        self.warmup_steps = train_cfg.get('warmup_steps', 500)
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.log_every = train_cfg.get('log_every', 50)

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        console.verbose(f"Trainer initialized on device: {self.device}")
        console.env_info()

    def build_model(self, vocab_size: int, seq_len: int) -> nn.Module:
        """Build model from configuration."""
        from ..model import SLiM_CZ_V1

        model_cfg = self.config.get('model', {})

        console.verbose(f"Building model: d_model={model_cfg.get('d_model', 256)}, "
                       f"layers={model_cfg.get('num_layers', 4)}")

        model = SLiM_CZ_V1(
            vocab_size=vocab_size,
            d_model=model_cfg.get('d_model', 256),
            num_heads=model_cfg.get('num_heads', 8),
            num_layers=model_cfg.get('num_layers', 4),
            d_ff=model_cfg.get('d_ff', 1024),
            max_seq_len=seq_len,
            dropout=model_cfg.get('dropout', 0.1),
            weight_tying=model_cfg.get('weight_tying', True)
        ).to(self.device)

        self.tb_logger.log_config(self.config)

        return model

    def train_epoch(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            epoch: int,
            global_step: int
    ) -> Tuple[float, int]:
        """Train one epoch."""

        model.train()
        total_loss = 0
        num_batches = len(train_loader)

        console.verbose(f"Epoch {epoch}/{self.epochs}: {num_batches} batches")

        # Use tqdm only if enabled AND not in verbose mode
        use_progress_bar = self.use_tqdm and not console.is_verbose

        if use_progress_bar:
            try:
                pbar = tqdm_auto(train_loader, desc=f"Epoch {epoch}/{self.epochs}",
                                leave=True, mininterval=1.0)
            except Exception:
                pbar = train_loader
                use_progress_bar = False
        else:
            pbar = train_loader

        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            # TensorBoard
            if global_step % self.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                self.tb_logger.log_lr(current_lr, global_step)
                self.tb_logger.log_metrics({
                    'loss': loss.item(),
                    'perplexity': torch.exp(loss).item(),
                }, global_step, prefix='Training')

            # Progress reporting
            if use_progress_bar:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ppl': f'{torch.exp(loss).item():.2f}'
                })
            else:
                # Verbose mode: batch-level logging
                console.batch(batch_idx, num_batches, loss.item(), {
                    'ppl': torch.exp(loss).item(),
                    'lr': scheduler.get_last_lr()[0]
                })

        avg_loss = total_loss / num_batches
        return avg_loss, global_step

    def evaluate(
            self,
            model: nn.Module,
            val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model on validation set."""

        model.eval()
        total_loss = 0
        num_batches = len(val_loader)

        console.verbose(f"Validation: {num_batches} batches")

        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(val_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                total_loss += loss.item()

                # Heartbeat during validation
                if batch_idx % 50 == 0:
                    console.heartbeat()

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        console.verbose(f"Validation complete: loss={avg_loss:.4f}, ppl={perplexity:.2f}")

        return avg_loss, perplexity

    def save_checkpoint(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            global_step: int,
            is_best: bool = False
    ):
        """Save model checkpoint."""

        console.verbose(f"Saving checkpoint epoch {epoch} (best={is_best})")

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
        }

        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def train(
            self,
            tokens_path: Path,
            seq_len: int,
            vocab_size: int
    ) -> Iterator[Dict[str, Any]]:
        """
        Main training loop.

        Yields progress dictionaries for each epoch.
        """

        console.header("SLiM-CZ-V1 Training")
        console.info("Slavic Linguistic integrated Micro-model for Czechia")

        console.section("Configuration")
        console.table({
            'Device': self.device,
            'Epochs': self.epochs,
            'Batch size': self.batch_size,
            'Learning rate': self.learning_rate,
            'Warmup steps': self.warmup_steps,
        })

        # Load data
        with console.status("Loading data"):
            train_dataset, val_dataset, stats = load_tokenized_data(
                tokens_path=tokens_path,
                seq_len=seq_len,
                vocab_size=vocab_size,
                val_split=self.config.get('train', {}).get('val_split', 0.1)
            )

        console.section("Data Statistics")
        console.table({
            'Train sequences': stats['train_sequences'],
            'Val sequences': stats['val_sequences'],
            'Sequence length': stats['seq_len'],
            'Vocabulary size': stats['vocab_size'],
        })

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.get('train', {}).get('num_workers', 4)
        )

        # Build model
        with console.status("Building model"):
            model = self.build_model(stats['vocab_size'], stats['seq_len'])
            params = model.count_parameters()

        console.info(f"Model parameters: {params['total']:,} ({params['trainable']:,} trainable)")

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get('train', {}).get('weight_decay', 0.01)
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Training loop
        global_step = 0
        training_start = time.time()

        console.section("Training Progress")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, global_step = self.train_epoch(
                model, train_loader, optimizer, scheduler, epoch, global_step
            )

            # Evaluate
            val_loss, val_ppl = self.evaluate(model, val_loader)

            # TensorBoard
            self.tb_logger.log_metrics({
                'train_loss': train_loss,
                'train_perplexity': torch.exp(torch.tensor(train_loss)).item(),
            }, epoch, prefix='Epoch')

            self.tb_logger.log_metrics({
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
            }, epoch, prefix='Epoch')

            # Best check
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_perplexity = val_ppl

            epoch_time = time.time() - epoch_start

            # Checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(model, optimizer, epoch, global_step, is_best)

            # Epoch summary
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'time': f"{epoch_time:.1f}s"
            }
            if is_best:
                metrics[''] = '★ BEST'
            console.epoch(epoch, self.epochs, metrics)

            # Yield progress
            yield {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'is_best': is_best,
                'epoch_time': epoch_time,
            }

        # Final
        self.save_checkpoint(model, optimizer, self.epochs, global_step, is_best=False)
        self.tb_logger.close()

        total_time = time.time() - training_start

        console.header("Training Complete")
        console.table({
            'Total time': f"{total_time/60:.1f} minutes",
            'Best val loss': self.best_val_loss,
            'Best val PPL': self.best_val_perplexity,
            'Output': str(self.output_dir),
        })

        yield {
            'final': True,
            'stats': stats,
            'params': params,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
            'total_time': total_time,
            'output_dir': self.output_dir,
        }