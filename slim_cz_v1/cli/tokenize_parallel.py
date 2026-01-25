#!/usr/bin/env python3
"""
CLI tool for parallel batch tokenization.

High-throughput CPU-parallelized tokenization using trained SentencePiece model.
"""

import argparse
import sys
import time
from pathlib import Path

from ..tokenization.parallel_tokenizer import ParallelTokenizer
from ..preprocessing.base import (
    print_error,
    print_success,
    print_section,
    print_info,
    print_warning,
    print_header
)


def main():
    """Main entry point for parallel tokenization CLI."""

    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 Parallel Batch Tokenization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parallel tokenization (auto-detects CPU cores)
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt

  # Specify number of workers
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --workers 16

  # Adjust chunk size for large files
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --chunk-size 50000

Performance Notes:
  - Automatic CPU core detection (uses N-1 cores)
  - Expected speedup: 10-15x on modern CPUs
  - Efficiency: 90-95% parallelization
  - GPU not used (no benefit for I/O-bound operations)

Chunk Size Guidelines:
  - Small files (<100 MB):       10,000 lines
  - Medium files (100-1000 MB):  50,000 lines
  - Large files (>1 GB):        100,000 lines
        """
    )

    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input text file (raw text, one sentence per line)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='SentencePiece model file (.model from training)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file for token IDs (space-separated integers)'
    )

    # Optional parameters
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        help='Number of lines per processing chunk (default: 10000)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output (recommended for notebooks/Kaggle)'
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not input_path.exists():
        print_error(f"Input file not found: {args.input}")
        return 1

    if not model_path.exists():
        print_error(f"Model file not found: {args.model}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print_header("SLiM-CZ-V1 Parallel Batch Tokenization")

        # Display configuration
        print_section("Configuration")

        file_size = input_path.stat().st_size
        file_size_mb = file_size / (1024 ** 2)
        file_size_gb = file_size / (1024 ** 3)

        print_info(f"Input file:  {input_path.name}")
        if file_size_gb >= 1.0:
            print_info(f"File size:   {file_size_gb:.2f} GB")
        else:
            print_info(f"File size:   {file_size_mb:.2f} MB")

        print_info(f"Model:       {model_path.name}")
        print_info(f"Output:      {output_path.name}")

        if args.workers:
            print_info(f"Workers:     {args.workers} (manual)")
        else:
            print_info(f"Workers:     auto-detect")

        print_info(f"Chunk size:  {args.chunk_size:,} lines")

        # Estimate processing time
        import multiprocessing as mp
        num_workers = args.workers or max(1, mp.cpu_count() - 1)

        # Single-core baseline: ~20 MB/s
        single_core_time = file_size_mb / 20.0

        # Multi-core with 92% efficiency
        expected_throughput = 20.0 * num_workers * 0.92
        expected_time = file_size_mb / expected_throughput

        print()
        print_section("Performance Estimate")
        print_info(f"CPU cores:        {mp.cpu_count()}")
        print_info(f"Workers:          {num_workers}")
        print_info(f"Expected speedup: {num_workers * 0.92:.1f}x")
        print_info(f"Estimated time:   {expected_time:.1f} seconds")

        if single_core_time > 60:
            time_saved = single_core_time - expected_time
            print_info(f"Time saved:       {time_saved:.0f} seconds vs single-core")

        # Warning for small files
        if file_size_mb < 10:
            print()
            print_warning("Small file detected (<10 MB)")
            print_info("Parallel processing overhead may exceed benefits")
            print_info("Consider using single-core for files <10 MB")

        print()
        print_section("Tokenization Progress")
    else:
        # In quiet mode, just show basic info
        print(f"[INFO] Tokenizing {input_path.name} ({input_path.stat().st_size / (1024 ** 3):.2f} GB)")
        print(f"[INFO] Using {args.workers or 'auto'} workers, chunk size {args.chunk_size:,}")

    # Perform parallel tokenization
    try:
        start_time = time.time()

        tokenizer = ParallelTokenizer(
            model_path=model_path,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            quiet=args.quiet
        )

        stats = tokenizer.tokenize_file(
            input_path=input_path,
            output_path=output_path,
            show_progress=not args.quiet
        )

        total_time = time.time() - start_time

        if not args.quiet:
            print()
            print_header("Tokenization Completed")

            print_section("Performance Summary")
            print_info(f"Total time:       {total_time:.2f} seconds")
            print_info(f"Throughput:       {stats['throughput_mb_per_second']:.2f} MB/s")
            print_info(f"Actual speedup:   {stats['actual_speedup']:.2f}x")
            print_info(f"Efficiency:       {stats['efficiency']:.1%}")
            print_info(f"Tokens/second:    {stats['tokens_per_second']:,.0f}")

            # Efficiency interpretation
            if stats['efficiency'] >= 0.90:
                print_success("Excellent parallelization efficiency")
            elif stats['efficiency'] >= 0.75:
                print_info("Good parallelization efficiency")
            else:
                print_warning("Lower efficiency - consider adjusting chunk size")

            print()
            print_section("Output")
            print_success(f"Token IDs saved: {output_path}")
            print_info(f"Total tokens:    {stats['total_tokens']:,}")
            print_info(f"Chunks processed: {stats['num_chunks']:,}")

            print()
            print_section("Next Steps")
            print("   1. Review tokenization statistics above")
            print("   2. Prepare training sequences: python prepare_sequences.py")
            print("   3. Integrate tokenizer statistics for comprehensive analysis")

            # Show prepare_sequences command
            print()
            print("   Recommended command:")
            print(f"      python prepare_sequences.py \\")
            print(f"        --input {output_path} \\")
            print(f"        --output ./data/prepared \\")
            print(f"        --seq-len 512 \\")
            print(f"        --tokenizer-stats {model_path.parent}/tokenizer.statistics.json")

            print("=" * 70)
        else:
            # In quiet mode, just show essential results
            print(f"[SUCCESS] Tokenization completed in {total_time:.1f}s")
            print(f"[INFO] Throughput: {stats['throughput_mb_per_second']:.1f} MB/s")
            print(f"[INFO] Total tokens: {stats['total_tokens']:,}")
            print(f"[SUCCESS] Output: {output_path}")

        return 0

    except Exception as e:
        print_error(f"Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())