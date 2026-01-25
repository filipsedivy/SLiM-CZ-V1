#!/usr/bin/env python3
"""
CLI tool for parallel batch tokenization - Industrial Grade.

High-throughput CPU-parallelized tokenization using trained SentencePiece model.
Designed for TB-scale corpus processing with checkpointing and resume capability.

Usage:
    slim-tokenize-parallel --input corpus.txt --model tokenizer.model --output tokens.txt

For TB-scale files:
    slim-tokenize-parallel \\
        --input corpus.txt \\
        --model tokenizer.model \\
        --output tokens.txt \\
        --workers 32 \\
        --chunk-size 100000 \\
        --checkpoint-dir ./checkpoints \\
        --log-file tokenization.log
"""

import argparse
import sys
import multiprocessing as mp
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


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} EB"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def main():
    """Main entry point for parallel tokenization CLI."""

    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 Parallel Batch Tokenization (Industrial Grade)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parallel tokenization (auto-detects CPU cores)
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt

  # With checkpointing for TB-scale files (can resume after crash)
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --checkpoint-dir ./checkpoints

  # Full configuration for maximum throughput
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --workers 32 \\
    --chunk-size 100000 \\
    --write-buffer 20000 \\
    --checkpoint-dir ./checkpoints \\
    --log-file ./tokenization.log \\
    --debug

  # Compress intermediate files (saves disk space for TB files)
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer/tokenizer.model \\
    --output ./data/tokens.txt \\
    --compress-temp

Architecture:
  - Streaming I/O: Constant memory regardless of file size
  - Parallel workers: Each process handles independent chunks
  - Checkpointing: Resume after crash without re-processing

Memory Usage (approximate):
  - Per worker: ~100-150 MB
  - 16 workers: ~2 GB total
  - 32 workers: ~4 GB total

Recommended Settings by File Size:
  - <10 GB:   --chunk-size 50000  --workers 8
  - 10-100 GB: --chunk-size 100000 --workers 16
  - 100+ GB:  --chunk-size 100000 --workers 32 --checkpoint-dir ./cp
  - 1+ TB:    --chunk-size 200000 --workers 32 --checkpoint-dir ./cp --compress-temp
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

    # Performance tuning
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50000,
        help='Number of lines per processing chunk (default: 50000)'
    )

    parser.add_argument(
        '--write-buffer',
        type=int,
        default=10000,
        help='Lines to buffer before disk write (default: 10000)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for checkpoint files (enables resume after crash)'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not attempt to resume from checkpoint (start fresh)'
    )

    # Output options
    parser.add_argument(
        '--compress-temp',
        action='store_true',
        help='Gzip compress intermediate files (saves disk, costs CPU)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    # Debugging
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Path to detailed log file'
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)
    log_file = Path(args.log_file) if args.log_file else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    if not input_path.exists():
        print_error(f"Input file not found: {args.input}")
        return 1

    if not model_path.exists():
        print_error(f"Model file not found: {args.model}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Display configuration
    if not args.quiet:
        print_header("SLiM-CZ-V1 Parallel Batch Tokenization")
        print_info("Industrial Grade - Streaming I/O for TB-scale files")
        print()

        print_section("Input")

        file_size = input_path.stat().st_size
        print_info(f"File:      {input_path}")
        print_info(f"Size:      {format_bytes(file_size)}")
        print_info(f"Model:     {model_path.name}")
        print_info(f"Output:    {output_path}")

        print()
        print_section("Configuration")

        num_workers = args.workers or max(1, mp.cpu_count() - 1)
        print_info(f"CPU cores:       {mp.cpu_count()}")
        print_info(f"Workers:         {num_workers}")
        print_info(f"Chunk size:      {args.chunk_size:,} lines")
        print_info(f"Write buffer:    {args.write_buffer:,} lines")
        print_info(f"Compress temp:   {args.compress_temp}")

        if checkpoint_dir:
            print_info(f"Checkpoint dir:  {checkpoint_dir}")
            print_info(f"Resume enabled:  {not args.no_resume}")

        # Memory estimate
        memory_per_worker = 100 + (args.chunk_size * 0.002)  # MB estimate
        total_memory = memory_per_worker * num_workers
        print()
        print_info(f"Est. memory:     {total_memory:.0f} MB peak")

        # Time estimate
        throughput_estimate = 20.0 * num_workers * 0.85  # MB/s
        time_estimate = (file_size / (1024 * 1024)) / throughput_estimate
        print_info(f"Est. time:       {format_duration(time_estimate)}")

        # Recommendations
        file_size_gb = file_size / (1024 ** 3)
        if file_size_gb > 100 and not checkpoint_dir:
            print()
            print_warning(f"Large file ({file_size_gb:.0f} GB) without checkpointing!")
            print_info("Recommend: --checkpoint-dir ./checkpoints")

        if file_size_gb > 500 and args.chunk_size < 100000:
            print()
            print_warning("Consider larger chunk size for very large files")
            print_info("Recommend: --chunk-size 100000 or --chunk-size 200000")

        print()
    else:
        file_size = input_path.stat().st_size
        print(f"[INFO] Tokenizing {input_path.name} ({format_bytes(file_size)})")

    # Run tokenization
    try:
        tokenizer = ParallelTokenizer(
            model_path=model_path,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            write_buffer_size=args.write_buffer,
            compress_temp=args.compress_temp,
            quiet=args.quiet,
            debug=args.debug,
            log_file=log_file,
            checkpoint_dir=checkpoint_dir
        )

        stats = tokenizer.tokenize_file(
            input_path=input_path,
            output_path=output_path,
            resume=not args.no_resume,
            show_progress=not args.quiet
        )

        # Final summary
        if not args.quiet:
            print()
            print_header("Tokenization Completed Successfully")
            print()
            print_success(f"Output: {output_path}")
            print_info(f"Total tokens: {stats['total_tokens']:,}")
            print_info(f"Total lines:  {stats['total_lines']:,}")
            print_info(f"Throughput:   {stats['throughput_mb_per_second']:.2f} MB/s")
            print_info(f"Wall time:    {format_duration(stats['wall_time_seconds'])}")

            if stats['failed_chunks'] > 0:
                print()
                print_warning(f"Failed chunks: {stats['failed_chunks']}")
                if log_file:
                    print_info(f"Check {log_file} for details")

            print()
        else:
            print(f"[SUCCESS] Completed in {format_duration(stats['wall_time_seconds'])}")
            print(f"[INFO] {stats['total_tokens']:,} tokens at {stats['throughput_mb_per_second']:.1f} MB/s")

        return 0

    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user")
        if checkpoint_dir:
            print_info("Progress saved - run again to resume")
        return 130

    except Exception as e:
        print_error(f"Tokenization failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print_info("Use --debug for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())