#!/usr/bin/env python3
"""
CLI tool for training SLiM-CZ-V1 language model.

Trains transformer model on tokenized sequences with TensorBoard monitoring.
"""

import argparse
import sys
from pathlib import Path

import yaml

from ..training.train import Trainer


# ============================================================
# CLI OUTPUT FORMATTING
# ============================================================

class CLIFormatter:
    """
    CLI output formatter with ANSI color support.

    Follows terminal UI/UX best practices:
    - Plaintext only (no Markdown)
    - Clear visual hierarchy
    - ANSI color codes for emphasis
    - Unicode symbols for quick scanning
    """

    # ANSI color codes
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def info(message: str):
        """Print info message."""
        print(f"{CLIFormatter.BLUE}[INFO]{CLIFormatter.RESET}    {message}")

    @staticmethod
    def success(message: str):
        """Print success message."""
        print(f"{CLIFormatter.GREEN}[SUCCESS]{CLIFormatter.RESET} ✔ {message}")

    @staticmethod
    def warning(message: str):
        """Print warning message."""
        print(f"{CLIFormatter.YELLOW}[WARNING]{CLIFormatter.RESET} ⚠ {message}")

    @staticmethod
    def error(message: str):
        """Print error message."""
        print(f"{CLIFormatter.RED}[ERROR]{CLIFormatter.RESET}   ✖ {message}")

    @staticmethod
    def header(title: str):
        """Print section header."""
        separator = "=" * 70
        print(f"\n{CLIFormatter.BOLD}{separator}{CLIFormatter.RESET}")
        print(f"{CLIFormatter.BOLD}{title}{CLIFormatter.RESET}")
        print(f"{CLIFormatter.BOLD}{separator}{CLIFormatter.RESET}\n")

    @staticmethod
    def section(title: str):
        """Print subsection."""
        print(f"\n{CLIFormatter.CYAN}{title}{CLIFormatter.RESET}")
        print("-" * 70)

    @staticmethod
    def metric(name: str, value: str, unit: str = ""):
        """Print metric in aligned format."""
        unit_str = f" {unit}" if unit else ""
        print(f"  {name:30s} {value}{unit_str}")

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 50):
        """
        Generate progress bar.

        Formula:
        - filled = floor(current / total * width)
        - percentage = current / total * 100
        """
        filled = int(current / total * width)
        bar = "█" * filled + "░" * (width - filled)
        percentage = (current / total) * 100
        return f"[{bar}] {percentage:.1f}%"


# ============================================================
# MAIN CLI FUNCTION
# ============================================================

def main():
    """Main entry point for training CLI."""

    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default config
  slim-train --config ./configs/base_config.yaml \\
    --tokens ./data/tokens.txt \\
    --output ./models/slim_v1

  # Training with custom sequence length
  slim-train --config ./configs/base_config.yaml \\
    --tokens ./data/tokens.txt \\
    --output ./models/slim_v1 \\
    --seq-len 512 \\
    --vocab-size 16000

  # Training without TensorBoard
  slim-train --config ./configs/base_config.yaml \\
    --tokens ./data/tokens.txt \\
    --output ./models/slim_v1 \\
    --no-tensorboard

Configuration:
  Create a YAML config file with model and training parameters.
  See configs/base_config.yaml for an example.

Requirements:
  - Tokenized data from slim-tokenize-parallel
  - Config file (YAML format)
  - PyTorch >= 2.0.0
        """
    )

    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--tokens', '-t',
        type=str,
        required=True,
        help='Path to tokenized data (output from slim-tokenize-parallel)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for model checkpoints'
    )

    # Optional arguments
    parser.add_argument(
        '--seq-len',
        type=int,
        default=256,
        help='Sequence length (default: 256)'
    )

    parser.add_argument(
        '--vocab-size',
        type=int,
        default=16000,
        help='Vocabulary size (default: 16000, must match tokenizer)'
    )

    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )

    args = parser.parse_args()

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    config_path = Path(args.config)
    tokens_path = Path(args.tokens)
    output_dir = Path(args.output)

    if not config_path.exists():
        CLIFormatter.error(f"Config file not found: {args.config}")
        return 1

    if not tokens_path.exists():
        CLIFormatter.error(f"Tokens file not found: {args.tokens}")
        return 1

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        CLIFormatter.error(f"Failed to load config: {e}")
        return 1

    # ============================================================
    # HEADER
    # ============================================================

    CLIFormatter.header("SLiM-CZ-V1 Training")
    CLIFormatter.info("Slavic Linguistic integrated Micro-model for Czechia")
    print()

    # ============================================================
    # CONFIGURATION DISPLAY
    # ============================================================

    CLIFormatter.section("Configuration")

    CLIFormatter.metric("Config file:", config_path.name)
    CLIFormatter.metric("Tokens file:", tokens_path.name)
    CLIFormatter.metric("Output directory:", str(output_dir))
    CLIFormatter.metric("Sequence length:", str(args.seq_len))
    CLIFormatter.metric("Vocabulary size:", f"{args.vocab_size:,}")

    model_cfg = config.get('model', {})
    CLIFormatter.metric("Model dimension:", str(model_cfg.get('d_model', 256)))
    CLIFormatter.metric("Attention heads:", str(model_cfg.get('num_heads', 8)))
    CLIFormatter.metric("Transformer layers:", str(model_cfg.get('num_layers', 4)))
    CLIFormatter.metric("Feed-forward dim:", str(model_cfg.get('d_ff', 1024)))

    train_cfg = config.get('train', {})
    CLIFormatter.metric("Training epochs:", str(train_cfg.get('epochs', 30)))
    CLIFormatter.metric("Batch size:", str(train_cfg.get('batch_size', 32)))
    CLIFormatter.metric("Learning rate:", f"{train_cfg.get('learning_rate', 0.0001):.6f}")
    CLIFormatter.metric("Warmup steps:", str(train_cfg.get('warmup_steps', 500)))

    tensorboard_enabled = not args.no_tensorboard and train_cfg.get('use_tensorboard', True)
    CLIFormatter.metric("TensorBoard:", "Enabled" if tensorboard_enabled else "Disabled")

    # ============================================================
    # INITIALIZE TRAINER
    # ============================================================

    try:
        trainer = Trainer(
            config,
            output_dir,
            enable_tensorboard=not args.no_tensorboard
        )

        CLIFormatter.success("Trainer initialized")
        CLIFormatter.metric("Device:", str(trainer.device))
        print()

    except Exception as e:
        CLIFormatter.error(f"Failed to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================
    # START TRAINING
    # ============================================================

    CLIFormatter.section("Training Progress")

    if tensorboard_enabled:
        CLIFormatter.info(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'}")

    print()

    try:
        # Training loop
        epochs = train_cfg.get('epochs', 30)

        for progress in trainer.train(
                tokens_path=tokens_path,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size
        ):
            epoch = progress['epoch']
            train_loss = progress['train_loss']
            val_loss = progress['val_loss']
            val_ppl = progress['val_perplexity']
            is_best = progress['is_best']
            epoch_time = progress['epoch_time']

            # Progress bar
            progress_bar = CLIFormatter.progress_bar(epoch, epochs)
            print(f"\n{progress_bar}")

            # Epoch summary
            print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print()

            # Training metrics
            print("Training:")
            CLIFormatter.metric("  Loss:", f"{train_loss:.6f}")

            # Perplexity calculation (mathematical formula)
            train_ppl = 2.71828 ** train_loss  # e^loss
            CLIFormatter.metric("  Perplexity:", f"{train_ppl:.4f}")
            CLIFormatter.info("  Formula: PPL = exp(loss)")
            print()

            # Validation metrics
            print("Validation:")
            CLIFormatter.metric("  Loss:", f"{val_loss:.6f}")
            CLIFormatter.metric("  Perplexity:", f"{val_ppl:.4f}")

            # Perplexity interpretation
            if val_ppl < 10:
                CLIFormatter.success("Excellent perplexity (PPL < 10)")
            elif val_ppl < 50:
                CLIFormatter.info("Good perplexity (10 <= PPL < 50)")
            elif val_ppl < 100:
                CLIFormatter.warning("Moderate perplexity (50 <= PPL < 100)")
            else:
                CLIFormatter.warning("High perplexity (PPL >= 100)")
                CLIFormatter.info("Consider: longer training, larger model, or data quality")

            print()

            # Best model indicator
            if is_best:
                CLIFormatter.success("New best model!")
                CLIFormatter.metric("  Best val loss:", f"{trainer.best_val_loss:.6f}")
                CLIFormatter.metric("  Best val PPL:", f"{trainer.best_val_perplexity:.4f}")
                print()

            # Checkpoint saved
            if epoch % 5 == 0 or is_best:
                CLIFormatter.info(f"Checkpoint saved: checkpoint_epoch_{epoch}.pt")
                if is_best:
                    CLIFormatter.info("Best model saved: best_model.pt")
                print()

        # ============================================================
        # TRAINING COMPLETED
        # ============================================================

        CLIFormatter.header("Training Completed")

        CLIFormatter.section("Final Results")

        # Best metrics
        CLIFormatter.metric("Best validation loss:", f"{trainer.best_val_loss:.6f}")
        CLIFormatter.metric("Best perplexity:", f"{trainer.best_val_perplexity:.4f}")
        print()

        # Perplexity formula explanation
        CLIFormatter.section("Perplexity Formula")
        print("  Mathematical definition:")
        print("    PPL = exp(loss)")
        print("    PPL = exp(-1/N * Σ log P(y|x))")
        print()
        print("  Where:")
        print("    - N is the number of tokens")
        print("    - P(y|x) is the model's predicted probability")
        print("    - lower PPL = better model")
        print()

        # Interpretation guidelines
        CLIFormatter.section("Perplexity Interpretation")
        print("  Based on empirical research:")
        print("    PPL < 10:   Excellent (production-ready)")
        print("    PPL < 50:   Good (acceptable for most tasks)")
        print("    PPL < 100:  Moderate (may need improvement)")
        print("    PPL >= 100: High (requires attention)")
        print()

        # Output location
        CLIFormatter.section("Output Files")
        CLIFormatter.metric("Model directory:", str(output_dir))
        CLIFormatter.metric("Best model:", "best_model.pt")
        CLIFormatter.metric("Checkpoints:", "checkpoint_epoch_*.pt")

        if tensorboard_enabled:
            CLIFormatter.metric("TensorBoard logs:", "tensorboard/")
        print()

        # Next steps
        CLIFormatter.section("Next Steps")
        print("  1. View training logs:")
        if tensorboard_enabled:
            print(f"     tensorboard --logdir {output_dir / 'tensorboard'}")
        print()
        print("  2. Evaluate model:")
        print(f"     python scripts/evaluate_model.py --model {output_dir / 'best_model.pt'}")
        print()
        print("  3. Export for deployment:")
        print(f"     python scripts/export_model.py --model {output_dir / 'best_model.pt'}")

        print()
        CLIFormatter.success("Training pipeline completed successfully")
        print("=" * 70)
        print()

        return 0

    except Exception as e:
        CLIFormatter.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())