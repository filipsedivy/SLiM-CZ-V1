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
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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


# ============================================================
# TENSORBOARD LOGGER
# ============================================================

class TensorBoardLogger:
    """
    TensorBoard logger for training metrics.

    Provides logging for:
    - Scalar metrics (loss, perplexity, learning rate)
    - Model architecture
    - Configuration
    """

    def __init__(self, log_dir: Path, enabled: bool = True):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Enable/disable logging
        """
        self.enabled = enabled and HAS_TENSORBOARD
        self.writer = None

        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log scalar metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
            prefix: Optional prefix for metric names
        """
        if not self.enabled:
            return

        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)

    def log_lr(self, lr: float, step: int):
        """Log learning rate."""
        if self.enabled:
            self.writer.add_scalar('Training/learning_rate', lr, step)

    def log_config(self, config: Dict):
        """Log configuration."""
        if self.enabled:
            import yaml
            config_text = yaml.dump(config, default_flow_style=False)
            self.writer.add_text('Config/full_config', f"```yaml\n{config_text}\n```", 0)

    def close(self):
        """Close writer."""
        if self.enabled and self.writer:
            self.writer.close()


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for tokenized sequences.

    Provides input/target pairs for language modeling:
    - Input: sequence[:-1]
    - Target: sequence[1:]
    """

    def __init__(self, sequences: torch.Tensor):
        """
        Initialize dataset.

        Args:
            sequences: Tensor of shape (num_sequences, seq_len)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get sequence pair for language modeling.

        Returns:
            Tuple of (input_ids, labels)
        """
        seq = self.sequences[idx]
        input_ids = seq[:-1]
        labels = seq[1:]
        return input_ids, labels


# ============================================================
# DATA LOADING
# ============================================================

def load_tokenized_data(
        tokens_path: Path,
        seq_len: int,
        vocab_size: int,
        val_split: float = 0.1
) -> Tuple[SequenceDataset, SequenceDataset, Dict]:
    """
    Load tokenized data and prepare sequences.

    Process:
    1. Read space-separated token IDs from file
    2. Create fixed-length sequences
    3. Split into train/validation sets

    Args:
        tokens_path: Path to tokenized data file
        seq_len: Sequence length
        vocab_size: Vocabulary size (for validation)
        val_split: Validation split ratio

    Returns:
        Tuple of (train_dataset, val_dataset, statistics)

    Statistics contains:
        - vocab_size: Vocabulary size
        - seq_len: Sequence length
        - train_sequences: Number of training sequences
        - val_sequences: Number of validation sequences
        - total_tokens: Total number of tokens
        - train_tokens: Training tokens
        - val_tokens: Validation tokens
    """
    # Read all tokens
    all_tokens = []
    with open(tokens_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = [int(t) for t in line.strip().split()]
            all_tokens.extend(tokens)

    # Create sequences
    num_sequences = len(all_tokens) // seq_len
    tokens_trimmed = all_tokens[:num_sequences * seq_len]

    sequences = torch.tensor(tokens_trimmed, dtype=torch.long).reshape(num_sequences, seq_len)

    # Split train/val
    val_size = int(num_sequences * val_split)
    train_size = num_sequences - val_size

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    # Create datasets
    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)

    # Statistics
    stats = {
        'vocab_size': vocab_size,
        'seq_len': seq_len,
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'total_tokens': len(all_tokens),
        'train_tokens': len(train_sequences) * seq_len,
        'val_tokens': len(val_sequences) * seq_len,
    }

    return train_dataset, val_dataset, stats


def create_dataloaders(
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset,
        batch_size: int,
        num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """
    Model trainer with TensorBoard logging.

    Features:
    - Automatic device selection (CUDA/CPU)
    - Learning rate warmup
    - Gradient clipping
    - Checkpoint management
    - TensorBoard logging
    - Progress tracking
    """

    def __init__(self, config: Dict, output_dir: Path, enable_tensorboard: bool = True):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            output_dir: Output directory for checkpoints
            enable_tensorboard: Enable TensorBoard logging
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.eval_every = train_cfg.get('eval_every', 500)
        self.log_every = train_cfg.get('log_every', 50)
        self.save_every = train_cfg.get('save_every', 1000)

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self, vocab_size: int, seq_len: int) -> nn.Module:
        """
        Build model from configuration.

        Args:
            vocab_size: Vocabulary size
            seq_len: Sequence length

        Returns:
            Initialized model on device
        """
        from ..model import SLiM_CZ_V1

        model_cfg = self.config.get('model', {})

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

        # Log config
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
        """
        Train one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            global_step: Global step counter

        Returns:
            Tuple of (average_loss, updated_global_step)
        """
        model.train()
        total_loss = 0
        num_batches = len(train_loader)

        if HAS_TQDM:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        else:
            pbar = train_loader

        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % self.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                self.tb_logger.log_lr(current_lr, global_step)
                self.tb_logger.log_metrics({
                    'loss': loss.item(),
                    'perplexity': torch.exp(loss).item(),
                }, global_step, prefix='Training')

            # Update progress bar
            if HAS_TQDM:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ppl': f'{torch.exp(loss).item():.2f}'
                })

        avg_loss = total_loss / num_batches
        return avg_loss, global_step

    def evaluate(
            self,
            model: nn.Module,
            val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation set.

        Mathematical formulation:
        - Loss: L = -Σ log P(y|x) / N
        - Perplexity: PPL = exp(L)

        where:
        - P(y|x) is model probability
        - N is number of tokens

        Args:
            model: Model to evaluate
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, perplexity)
        """
        model.eval()
        total_loss = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def save_checkpoint(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            global_step: int,
            is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            global_step: Global step
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def train(
            self,
            tokens_path: Path,
            seq_len: int,
            vocab_size: int
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            tokens_path: Path to tokenized data
            seq_len: Sequence length
            vocab_size: Vocabulary size

        Returns:
            Training statistics dictionary
        """
        # Load data
        train_dataset, val_dataset, stats = load_tokenized_data(
            tokens_path=tokens_path,
            seq_len=seq_len,
            vocab_size=vocab_size,
            val_split=self.config.get('train', {}).get('val_split', 0.1)
        )

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.get('train', {}).get('num_workers', 4)
        )

        # Build model
        model = self.build_model(stats['vocab_size'], stats['seq_len'])

        # Get model parameters
        params = model.count_parameters()

        # Optimizer and scheduler
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

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, global_step = self.train_epoch(
                model, train_loader, optimizer, scheduler, epoch, global_step
            )

            # Evaluate
            val_loss, val_ppl = self.evaluate(model, val_loader)

            # Log epoch metrics
            self.tb_logger.log_metrics({
                'train_loss': train_loss,
                'train_perplexity': torch.exp(torch.tensor(train_loss)).item(),
            }, epoch, prefix='Epoch')

            self.tb_logger.log_metrics({
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
            }, epoch, prefix='Epoch')

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_perplexity = val_ppl

            epoch_time = time.time() - epoch_start

            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(model, optimizer, epoch, global_step, is_best)

            # Yield progress (for CLI to display)
            yield {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'is_best': is_best,
                'epoch_time': epoch_time,
            }

        # Final save
        self.save_checkpoint(model, optimizer, self.epochs, global_step, is_best=False)

        # Close TensorBoard
        self.tb_logger.close()

        # Return final statistics
        total_time = time.time() - training_start

        return {
            'stats': stats,
            'params': params,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
            'total_time': total_time,
            'output_dir': self.output_dir,
        }