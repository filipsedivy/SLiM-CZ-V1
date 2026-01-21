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

  # With custom settings
  slim-train-tokenizer --input ./data/corpus.txt --output ./models/tokenizer \\
    --vocab-size 20000 --character-coverage 0.9999

Research Notes:
  - Vocab size 16-24k optimal for Czech with limited data
  - Character coverage 0.9999 handles Czech diacritics
  - BPE better for morphologically rich languages like Czech
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

    # Determine input mode
    if input_path.is_file():
        # Single file mode
        corpus_file = input_path
        
        if not corpus_file.is_file():
            print_error(f"Input is not a file: {corpus_file}")
            return 1
            
        print_header("SLiM-CZ-V1 BPE Tokenizer Training")
        print_section("Loading Corpus (Single File Mode)")
        
    elif input_path.is_dir():
        # Directory mode (scalable for TB datasets)
        print_header("SLiM-CZ-V1 BPE Tokenizer Training")
        print_section("Input Validation (Directory Mode)")
        
        txt_files = list(input_path.rglob('*.txt'))
        if not txt_files:
            print_error(f"No .txt files found in {args.input}")
            return 1
        
        print_success(f"Found {len(txt_files)} text files")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in txt_files)
        print_info(f"Total size: {total_size / (1024**3):.2f} GB")
        
        if total_size > 100 * 1024**3:  # > 100GB
            print_warning("Large dataset detected (>100GB)")
            print_info("Training may take significant time")
        
        # For directory mode, we pass the directory directly to tokenizer
        corpus_file = input_path
        
    else:
        print_error(f"Input must be file or directory: {args.input}")
        return 1

    # Load corpus for validation (only in single file mode)
    if input_path.is_file():
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus_text = f.read()
            
            print_success(f"Loaded corpus: {corpus_file}")
            print_info(f"Corpus size: {len(corpus_text):,} characters")
            print_info(f"Corpus size: {len(corpus_text.split()):,} words")
            
            # Validate corpus size
            if len(corpus_text) < 10000:
                print_error("Corpus too small (< 10,000 characters)")
                print_info("Recommended minimum: 100,000 characters for quality tokenizer")
                return 1
                
        except Exception as e:
            print_error(f"Failed to load corpus: {e}")
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
        print_error("WARNING: Vocabulary size < 8,000 may be too small")
        print_info("Recommended: 16,000-24,000 tokens for Czech")
    elif args.vocab_size > 32000:
        print_error("WARNING: Vocabulary size > 32,000 may be too large for limited data")
        print_info("Recommended: 16,000-24,000 tokens for Czech")

    # Train tokenizer
    try:
        output_dir = Path(args.output)
        
        tokenizer = BPETokenizer(config)
        tokenizer.train(
            input_path=corpus_file,  # Now Path to file or directory
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
