#!/usr/bin/env python3
"""
CLI tool for parallel batch tokenization - STREAMING VERSION.

High-throughput CPU-parallelized tokenization using trained SentencePiece model.
Memory-efficient streaming I/O for large files (4+ GB).
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
  python tokenize_parallel.py --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt

  # With debug logging to file
  python tokenize_parallel.py --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --debug --log-file ./tokenization.log

  # Sequential mode (for debugging or small files)
  python tokenize_parallel.py --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --sequential

  # Large files (>4 GB) - increase chunk size
  python tokenize_parallel.py --input ./data/large_corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --chunk-size 50000 --workers 8

Memory Usage Guidelines:
  - Streaming mode: ~50 MB per worker + chunk data
  - For 4 GB file with 8 workers: ~1 GB peak memory
  - Previous non-streaming: 4+ GB memory required

Chunk Size Guidelines:
  - Small files (<100 MB):       10,000 lines
  - Medium files (100 MB-1 GB):  25,000 lines
  - Large files (1-10 GB):       50,000 lines
  - Very large files (>10 GB):  100,000 lines

Troubleshooting:
  - If parallel fails, use --sequential
  - If memory issues, reduce --workers or --chunk-size
  - Use --debug --log-file for detailed diagnostics
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

    # New debugging options
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Path to debug log file (enables file logging)'
    )

    parser.add_argument(
        '--sequential', '-s',
        action='store_true',
        help='Force sequential processing (no multiprocessing, useful for debugging)'
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)
    log_file = Path(args.log_file) if args.log_file else None

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
        print_info("Streaming Mode - Memory Efficient for Large Files")
        print()

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

        if args.sequential:
            print_warning("Sequential mode enabled (single process)")

        if args.debug:
            print_info("Debug logging: ENABLED")
            if log_file:
                print_info(f"Log file: {log_file}")

        # Estimate processing time
        import multiprocessing as mp
        num_workers = 1 if args.sequential else (args.workers or max(1, mp.cpu_count() - 1))

        # Baseline: ~15-25 MB/s per core for tokenization
        single_core_throughput = 20.0  # MB/s estimate
        expected_throughput = single_core_throughput * num_workers * 0.85  # 85% efficiency estimate
        expected_time = file_size_mb / expected_throughput

        print()
        print_section("Performance Estimate")
        print_info(f"CPU cores:        {mp.cpu_count()}")
        print_info(f"Workers:          {num_workers}")
        print_info(f"Est. throughput:  {expected_throughput:.1f} MB/s")

        if expected_time < 60:
            print_info(f"Estimated time:   {expected_time:.1f} seconds")
        elif expected_time < 3600:
            print_info(f"Estimated time:   {expected_time / 60:.1f} minutes")
        else:
            print_info(f"Estimated time:   {expected_time / 3600:.1f} hours")

        # Memory estimate
        memory_per_worker = 50 + (args.chunk_size * 0.001)  # ~50 MB base + chunk data
        total_memory = memory_per_worker * num_workers
        print_info(f"Est. peak memory: {total_memory:.0f} MB")

        # Recommendations for large files
        if file_size_gb >= 4.0:
            print()
            print_section("Large File Recommendations")
            if args.chunk_size < 50000:
                print_warning(f"Consider --chunk-size 50000 for {file_size_gb:.1f} GB file")
            if not args.sequential and num_workers > 8:
                print_info("With many workers, monitor memory usage")

        print()
        print("=" * 70)
        print("  Starting tokenization...")
        print("=" * 70)
        print()
    else:
        # In quiet mode, just show basic info
        file_size_gb = input_path.stat().st_size / (1024 ** 3)
        mode = "SEQUENTIAL" if args.sequential else "PARALLEL"
        print(f"[INFO] Tokenizing {input_path.name} ({file_size_gb:.2f} GB) - {mode} mode")
        print(f"[INFO] Workers: {args.workers or 'auto'}, chunk size: {args.chunk_size:,}")

    # Perform parallel tokenization
    try:
        start_time = time.time()

        tokenizer = ParallelTokenizer(
            model_path=model_path,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            quiet=args.quiet,
            debug=args.debug,
            log_file=log_file,
            sequential=args.sequential
        )

        stats = tokenizer.tokenize_file(
            input_path=input_path,
            output_path=output_path,
            show_progress=not args.quiet
        )

        total_time = time.time() - start_time

        if not args.quiet:
            print()
            print_header("Tokenization Completed Successfully")

            # Summary
            print_section("Summary")
            print_success(f"Output saved: {output_path}")
            print_info(f"Total tokens:    {stats['total_tokens']:,}")
            print_info(f"Total lines:     {stats['total_lines']:,}")
            print_info(f"Total time:      {total_time:.2f} seconds")
            print_info(f"Throughput:      {stats['throughput_mb_per_second']:.2f} MB/s")

            if stats['num_errors'] > 0:
                print_warning(f"Chunks with errors: {stats['num_errors']}")
                if log_file:
                    print_info(f"See {log_file} for error details")

            print()
            print("=" * 70)
        else:
            # Quiet mode results
            print(f"[SUCCESS] Tokenization completed in {total_time:.1f}s")
            print(f"[INFO] Throughput: {stats['throughput_mb_per_second']:.1f} MB/s")
            print(f"[INFO] Total tokens: {stats['total_tokens']:,}")
            if stats['num_errors'] > 0:
                print(f"[WARNING] Errors: {stats['num_errors']} chunks failed")
            print(f"[SUCCESS] Output: {output_path}")

        return 0

    except KeyboardInterrupt:
        print()
        print_warning("Tokenization interrupted by user")
        return 130

    except Exception as e:
        print_error(f"Tokenization failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        elif not args.quiet:
            print_info("Use --debug for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())