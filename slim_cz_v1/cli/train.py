#!/usr/bin/env python3
"""
CLI entry point for SLiM-CZ-V1 training.

Usage:
    slim-train --config config.yaml --tokens tokens.txt --output models/
    slim-train --config config.yaml --tokens tokens.txt --output models/ --verbose
    slim-train --config config.yaml --tokens tokens.txt --output models/ --quiet
"""

import argparse
import sys
from pathlib import Path

import yaml

# Import console from utils FIRST to configure it before anything else
from ..utils import console

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SLiM-CZ-V1 language model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Basic training
    slim-train --config slim_cz_v1_default.yaml --tokens tokens.txt --output ./output

    # Verbose output (recommended for Kaggle/debugging)
    slim-train --config slim_cz_v1_default.yaml --tokens tokens.txt --output ./output --verbose

    # Quiet mode (only errors)
    slim-train --config slim_cz_v1_default.yaml --tokens tokens.txt --output ./output --quiet

Environment variables:
    SLIM_VERBOSE=1     Enable verbose output (same as --verbose)
    SLIM_QUIET=1       Enable quiet mode (same as --quiet)
    SLIM_NO_COLOR=1    Disable colored output (same as --no-color)
'''
    )

    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--tokens', '-t',
        type=Path,
        required=True,
        help='Path to tokenized data file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for checkpoints and logs'
    )

    # Output control
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with batch-level progress'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all non-error output'
    )

    # Additional options
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    parser.add_argument(
        '--no-tqdm',
        action='store_true',
        help='Disable tqdm progress bars'
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure console based on CLI flags
    console.configure(
        verbose=args.verbose,
        quiet=args.quiet,
        color=not args.no_color
    )

    # Validate inputs
    if not args.config.exists():
        console.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not args.tokens.exists():
        console.error(f"Tokens file not found: {args.tokens}")
        sys.exit(1)

    # Load configuration
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        console.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Import trainer after console is configured
    from slim_cz_v1.training import Trainer

    # Extract model parameters
    model_cfg = config.get('model', {})
    seq_len = model_cfg.get('max_seq_len', 256)
    vocab_size = model_cfg.get('vocab_size', 16000)

    # Initialize trainer
    try:
        trainer = Trainer(
            config=config,
            output_dir=args.output,
            enable_tensorboard=not args.no_tensorboard,
            use_tqdm=not args.no_tqdm
        )
    except Exception as e:
        console.error(f"Failed to initialize trainer: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Run training
    try:
        final_result = None
        for progress in trainer.train(args.tokens, seq_len, vocab_size):
            if progress.get('final'):
                final_result = progress
                break

        if final_result:
            console.success(f"Training completed! Best PPL: {final_result['best_val_perplexity']:.2f}")
            sys.exit(0)
        else:
            console.warning("Training finished without final result")
            sys.exit(0)

    except KeyboardInterrupt:
        console.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        console.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()