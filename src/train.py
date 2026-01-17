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
from typing import Dict, Any, Optional

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
    print("[WARNING] TensorBoard not available. Install with: pip install tensorboard")

from model import SLiM_CZ_V1
from dataloader import load_preprocessed_data, create_dataloaders


class TensorBoardLogger:
    """Enhanced TensorBoard logger with comprehensive metrics."""

    def __init__(self, log_dir: Path, enabled: bool = True):
        self.enabled = enabled and HAS_TENSORBOARD
        self.writer = None

        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
            print(f"✓ TensorBoard logging enabled: {log_dir / 'tensorboard'}")
            print(f"  View with: tensorboard --logdir {log_dir / 'tensorboard'}")
        else:
            if not HAS_TENSORBOARD:
                print("⚠ TensorBoard not installed - logging disabled")
            else:
                print("ℹ TensorBoard logging disabled")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log scalar metrics."""
        if not self.enabled:
            return

        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)

    def log_lr(self, lr: float, step: int):
        """Log learning rate."""
        if not self.enabled:
            return
        self.writer.add_scalar('Training/learning_rate', lr, step)

    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient statistics."""
        if not self.enabled:
            return

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

                # Log per-parameter gradients
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, step)

        total_norm = total_norm ** 0.5
        self.writer.add_scalar('Gradients/total_norm', total_norm, step)

    def log_weights(self, model: nn.Module, step: int):
        """Log model weights."""
        if not self.enabled:
            return

        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param.data, step)

    def log_text_samples(self, prompts: list, generated: list, step: int):
        """Log generated text samples."""
        if not self.enabled:
            return

        text = "\n\n".join([
            f"**Prompt:** {p}\n**Generated:** {g}"
            for p, g in zip(prompts, generated)
        ])
        self.writer.add_text('Generations/samples', text, step)

    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        """Log model graph."""
        if not self.enabled:
            return

        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"⚠ Could not log model graph: {e}")

    def log_config(self, config: Dict):
        """Log configuration as text."""
        if not self.enabled:
            return

        config_text = yaml.dump(config, default_flow_style=False)
        self.writer.add_text('Config/full_config', f"```yaml\n{config_text}\n```", 0)

    def log_dataset_info(self, stats: Dict, step: int = 0):
        """Log dataset statistics."""
        if not self.enabled:
            return

        info_text = f"""
**Dataset Statistics:**
- Vocabulary size: {stats.get('vocab_size', 'N/A'):,}
- Sequence length: {stats.get('seq_len', 'N/A')}
- Training sequences: {stats.get('train_sequences', 'N/A'):,}
- Validation sequences: {stats.get('val_sequences', 'N/A'):,}
- Total tokens: {stats.get('train_sequences', 0) * stats.get('seq_len', 0):,}
"""
        self.writer.add_text('Dataset/info', info_text, step)

    def close(self):
        """Close writer."""
        if self.enabled and self.writer:
            self.writer.close()


class Trainer:
    """Enhanced trainer with TensorBoard logging."""

    def __init__(self, config: Dict, output_dir: Path, enable_tensorboard: bool = True):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard
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
        self.generate_samples_every = train_cfg.get('generate_samples_every', 1000)

        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def build_model(self, vocab_size: int, seq_len: int) -> nn.Module:
        """Build model from config."""
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

        # Log model info
        params = model.count_parameters()
        print(f"\nModel Parameters:")
        print(f"  Total: {params['total']:,} ({params['total']/1e6:.2f}M)")
        print(f"  Trainable: {params['trainable']:,}")
        if model.weight_tying:
            print(f"  Saved by tying: {params['saved']:,}")

        # Log to TensorBoard
        self.tb_logger.log_config(self.config)

        return model

    def train_epoch(
        self,
        model: nn.Module,
        train_loader,
        optimizer,
        scheduler,
        epoch: int,
        tokenizer=None
    ) -> float:
        """Train one epoch."""
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

            # Global step
            global_step = (epoch - 1) * num_batches + batch_idx

            # Logging
            if batch_idx % self.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']

                # Log to TensorBoard
                self.tb_logger.log_metrics({
                    'batch_loss': loss.item(),
                    'batch_perplexity': torch.exp(loss).item()
                }, global_step, prefix='Training')

                self.tb_logger.log_lr(current_lr, global_step)

                # Log gradients periodically
                if batch_idx % (self.log_every * 5) == 0:
                    self.tb_logger.log_gradients(model, global_step)

                if HAS_TQDM:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{current_lr:.2e}"
                    })

            # Generate samples
            if tokenizer and batch_idx % self.generate_samples_every == 0 and batch_idx > 0:
                self.generate_and_log_samples(model, tokenizer, global_step)

            # Log weights occasionally
            if batch_idx % (self.save_every // 2) == 0 and batch_idx > 0:
                self.tb_logger.log_weights(model, global_step)

        return total_loss / num_batches

    def evaluate(self, model: nn.Module, val_loader) -> tuple:
        """Evaluate model."""
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

    def generate_and_log_samples(
        self,
        model: nn.Module,
        tokenizer,
        step: int,
        prompts: Optional[list] = None
    ):
        """Generate and log text samples."""
        if prompts is None:
            prompts = [
                "Praha je",
                "Dnes je krásný",
                "Česká republika"
            ]

        model.eval()
        generated_texts = []

        with torch.no_grad():
            for prompt in prompts:
                # Encode
                input_ids = tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

                # Generate
                generated = model.generate(
                    input_tensor,
                    max_length=50,
                    temperature=0.8,
                    top_k=50
                )

                # Decode
                generated_text = tokenizer.decode(generated[0].tolist())
                generated_texts.append(generated_text)

        model.train()

        # Log to TensorBoard
        self.tb_logger.log_text_samples(prompts, generated_texts, step)

        # Print samples
        print("\n" + "=" * 80)
        print("GENERATED SAMPLES:")
        for prompt, gen in zip(prompts, generated_texts):
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {gen}")
        print("=" * 80 + "\n")

    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")

    def train(self, data_dir: str, tokenizer_path: Optional[str] = None):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("SLiM-CZ-V1 Training with TensorBoard")
        print("=" * 80)

        # Load data
        print("\nLoading data...")
        train_sequences, val_sequences, stats = load_preprocessed_data(data_dir)

        vocab_size = stats['vocab_size']
        seq_len = stats['seq_len']

        print(f"  Train sequences: {len(train_sequences):,}")
        print(f"  Val sequences: {len(val_sequences):,}")
        print(f"  Vocabulary: {vocab_size:,}")
        print(f"  Sequence length: {seq_len}")

        # Log dataset info
        self.tb_logger.log_dataset_info(stats)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_sequences,
            val_sequences,
            batch_size=self.batch_size
        )

        # Build model
        model = self.build_model(vocab_size, seq_len)

        # Log model graph
        sample_input = torch.randint(0, vocab_size, (1, seq_len)).to(self.device)
        self.tb_logger.log_model_graph(model, sample_input)

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

        # Load tokenizer for sample generation
        tokenizer = None
        if tokenizer_path:
            try:
                import sentencepiece as spm
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.load(tokenizer_path)
                print(f"✓ Tokenizer loaded for sample generation")
            except Exception as e:
                print(f"⚠ Could not load tokenizer: {e}")

        # Training loop
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Output directory: {self.output_dir}")
        print(f"TensorBoard: tensorboard --logdir {self.output_dir / 'tensorboard'}")
        print()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(
                model, train_loader, optimizer, scheduler, epoch, tokenizer
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

            # Print summary
            elapsed = time.time() - epoch_start
            print()
            print(f"Epoch {epoch}/{self.epochs} Summary ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}  |  Train PPL: {torch.exp(torch.tensor(train_loss)):.2f}")
            print(f"  Val Loss:   {val_loss:.4f}  |  Val PPL:   {val_ppl:.2f}")
            if is_best:
                print(f"  ✓ New best model!")
            print()

            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(model, optimizer, epoch, is_best)

        # Final save
        self.save_checkpoint(model, optimizer, self.epochs, is_best=False)

        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best perplexity: {self.best_val_perplexity:.2f}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nView TensorBoard:")
        print(f"  tensorboard --logdir {self.output_dir / 'tensorboard'}")
        print("=" * 80)

        # Close TensorBoard
        self.tb_logger.close()


def main():
    parser = argparse.ArgumentParser(description='Train SLiM-CZ-V1 with TensorBoard')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to prepared data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--tokenizer', type=str, help='Path to tokenizer for sample generation')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard')

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = Trainer(
        config,
        args.output_dir,
        enable_tensorboard=not args.no_tensorboard
    )

    # Train
    trainer.train(args.data_dir, args.tokenizer)


if __name__ == "__main__":
    main()
