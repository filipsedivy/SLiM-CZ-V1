#!/usr/bin/env python3
"""
CLI tool for training BPE tokenizer.

Trains SentencePiece BPE tokenizer on preprocessed text corpus.
"""

import argparse
import sys
from pathlib import Path

from ..tokenization import BPETokenizer, VocabularyManager
from ..preprocessing.base import (
    print_error,
    print_success,
    print_section,
    print_info,
    print_warning,
    print_header
)


def main():
    """Main entry point for tokenizer training CLI."""

    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 BPE Tokenizer Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  slim-train-tokenizer --input ./data/corpus.txt --output ./models/tokenizer

  # Custom vocabulary size
  slim-train-tokenizer --input ./data/corpus.txt --output ./models/tokenizer \\
    --vocab-size 24000

  # Train on directory of files (efficient for TB-scale datasets)
  slim-train-tokenizer --input ./data/corpus_dir --output ./models/tokenizer \\
    --vocab-size 20000 --character-coverage 0.9999

Research Notes:
  - Vocab size 16-24k optimal for Czech with limited data
  - Character coverage 0.9999 handles Czech diacritics
  - BPE better for morphologically rich languages like Czech
  - Directory mode streams files without loading into memory
        """
    )

    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input: text file OR directory with .txt files (for TB-scale datasets)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for tokenizer model'
    )

    # Tokenizer parameters
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=16000,
        help='Vocabulary size (default: 16000, recommended: 16000-24000)'
    )

    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9999,
        help='Character coverage for handling rare characters (default: 0.9999)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='bpe',
        choices=['bpe', 'unigram', 'char', 'word'],
        help='Tokenizer model type (default: bpe)'
    )

    parser.add_argument(
        '--model-prefix',
        type=str,
        default='tokenizer',
        help='Prefix for model files (default: tokenizer)'
    )

    # Analysis options
    parser.add_argument(
        '--show-samples',
        action='store_true',
        help='Show sample tokenizations'
    )

    parser.add_argument(
        '--show-vocab',
        action='store_true',
        help='Show vocabulary statistics'
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)

    if not input_path.exists():
        print_error(f"Input not found: {args.input}")
        return 1

    print_header("SLiM-CZ-V1 BPE Tokenizer Training")

    # Determine input mode and validate
    if input_path.is_file():
        # Single file mode
        print_section("Input Validation (Single File Mode)")

        # Basic file validation
        file_size = input_path.stat().st_size
        print_success(f"Input file: {input_path.name}")
        print_info(f"File size: {file_size / (1024 ** 2):.2f} MB")

        # Validate minimum size (10KB as a basic check)
        if file_size < 10000:
            print_error("File too small (< 10KB)")
            print_info("Recommended minimum: 100KB for quality tokenizer")
            return 1

    elif input_path.is_dir():
        # Directory mode (scalable for TB datasets)
        print_section("Input Validation (Directory Mode)")

        txt_files = list(input_path.rglob('*.txt'))
        if not txt_files:
            print_error(f"No .txt files found in {args.input}")
            return 1

        print_success(f"Found {len(txt_files)} text files")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in txt_files)
        print_info(f"Total size: {total_size / (1024 ** 3):.2f} GB")

        if total_size > 100 * 1024 ** 3:  # > 100GB
            print_warning("Large dataset detected (>100GB)")
            print_info("Training may take significant time")
            print_info("SentencePiece will stream files (no memory issues)")

    else:
        print_error(f"Input must be file or directory: {args.input}")
        return 1

    # Create configuration
    config = {
        'vocab_size': args.vocab_size,
        'character_coverage': args.character_coverage,
        'model_type': args.model_type,
    }

    # Display configuration
    print_section("Tokenizer Configuration")
    print_info(f"Vocabulary size:    {args.vocab_size:,} tokens")
    print_info(f"Character coverage: {args.character_coverage}")
    print_info(f"Model type:         {args.model_type.upper()}")

    # Vocabulary size recommendations
    if args.vocab_size < 8000:
        print_warning("Vocabulary size < 8,000 may be too small")
        print_info("Recommended: 16,000-24,000 tokens for Czech")
    elif args.vocab_size > 32000:
        print_warning("Vocabulary size > 32,000 may be too large for limited data")
        print_info("Recommended: 16,000-24,000 tokens for Czech")

    # Train tokenizer
    try:
        output_dir = Path(args.output)

        tokenizer = BPETokenizer(config)
        tokenizer.train(
            input_path=input_path,
            output_dir=output_dir,
            model_prefix=args.model_prefix
        )

        # Show vocabulary statistics
        if args.show_vocab:
            vocab_path = output_dir / f'{args.model_prefix}.vocab'
            if vocab_path.exists():
                vocab_manager = VocabularyManager(vocab_path)
                vocab_manager.print_statistics()
                vocab_manager.print_sample_tokens(50)

        print_header("Tokenizer Training Completed")

        print_section("Output Files")
        print_info(f"Model:      {output_dir}/{args.model_prefix}.model")
        print_info(f"Vocabulary: {output_dir}/{args.model_prefix}.vocab")

        print_section("Next Steps")
        print("   1. Review tokenizer statistics")
        print("   2. Test tokenization on sample texts")
        print("   3. Prepare training sequences: slim-prepare-sequences")
        print("=" * 70)

        return 0

    except Exception as e:
        print_error(f"Tokenizer training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())