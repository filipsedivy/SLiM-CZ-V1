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

from ..tokenization.parallel_tokenizer import ParallelTokenizer, ShardProcessor
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
  # Single file tokenization
  slim-tokenize-parallel --input ./data/corpus.txt \\
    --model ./models/tokenizer.model \\
    --output ./data/tokens.txt

  # Process shards (recommended for large datasets)
  slim-tokenize-parallel --shards "./shards/shard-*.txt" \\
    --model ./models/tokenizer.model \\
    --output-dir ./tokens/

  # Process shards and merge into single file
  slim-tokenize-parallel --shards "./shards/shard-*.txt" \\
    --model ./models/tokenizer.model \\
    --output ./tokens.txt \\
    --merge

  # Full configuration for TB-scale shard processing
  slim-tokenize-parallel --shards "./shards/shard-*.txt" \\
    --model ./models/tokenizer.model \\
    --output-dir ./tokens/ \\
    --workers 32 \\
    --checkpoint-dir ./checkpoints \\
    --log-file tokenization.log

Shard Processing Benefits:
  - No file scanning needed (each shard = one work unit)
  - Natural checkpoint granularity (per-shard)
  - Better I/O parallelism (multiple files)
  - Can process across multiple machines

Recommended Settings by Total Size:
  Single File Mode:
  - <10 GB:    --chunk-size 50000  --workers 8
  - 10-100 GB: --chunk-size 100000 --workers 16
  - 100+ GB:   --chunk-size 100000 --workers 32 --checkpoint-dir ./cp

  Shard Mode (preferred for large data):
  - Any size:  --shards "pattern" --workers N --checkpoint-dir ./cp
        """
    )

    # Input mode (mutually exclusive: single file or shards)
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input text file (single file mode)'
    )

    input_group.add_argument(
        '--shards', '-s',
        type=str,
        help='Glob pattern for shard files (e.g., "shards/shard-*.txt")'
    )

    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='SentencePiece model file (.model from training)'
    )

    # Output (different meaning for single vs shard mode)
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file (single file mode) or merged output (shard mode with --merge)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for per-shard token files (shard mode only)'
    )

    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge shard outputs into single file (shard mode only)'
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

    # Determine mode
    shard_mode = args.shards is not None

    # Validate paths
    model_path = Path(args.model)
    log_file = Path(args.log_file) if args.log_file else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    if not model_path.exists():
        print_error(f"Model file not found: {args.model}")
        return 1

    # Mode-specific validation
    if shard_mode:
        # Shard mode validation
        if args.merge:
            if not args.output:
                print_error("--merge requires --output for merged output file")
                return 1
            output_path = Path(args.output)
            output_dir = None
        else:
            if not args.output_dir:
                print_error("Shard mode requires --output-dir or --merge with --output")
                return 1
            output_dir = Path(args.output_dir)
            output_path = None

        # Discover shards for info display
        import glob
        pattern = args.shards
        if Path(pattern).is_dir():
            shard_files = list(Path(pattern).glob("*.txt"))
        else:
            shard_files = [Path(f) for f in glob.glob(pattern)]

        if not shard_files:
            print_error(f"No shard files found matching: {args.shards}")
            return 1

        total_size = sum(f.stat().st_size for f in shard_files)

    else:
        # Single file mode validation
        if not args.output:
            print_error("Single file mode requires --output")
            return 1

        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            print_error(f"Input file not found: {args.input}")
            return 1

        total_size = input_path.stat().st_size

    # Create output directories
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if shard_mode and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Display configuration
    if not args.quiet:
        print_header("SLiM-CZ-V1 Parallel Batch Tokenization")

        if shard_mode:
            print_info("Mode: SHARD PROCESSING (recommended for large datasets)")
        else:
            print_info("Mode: Single File Processing")
        print()

        print_section("Input")

        if shard_mode:
            print_info(f"Pattern:   {args.shards}")
            print_info(f"Shards:    {len(shard_files)} files")
            print_info(f"Total:     {format_bytes(total_size)}")
        else:
            print_info(f"File:      {input_path}")
            print_info(f"Size:      {format_bytes(total_size)}")

        print_info(f"Model:     {model_path.name}")

        if shard_mode:
            if args.merge:
                print_info(f"Output:    {output_path} (merged)")
            else:
                print_info(f"Output:    {output_dir}/ (per-shard)")
        else:
            print_info(f"Output:    {output_path}")

        print()
        print_section("Configuration")

        num_workers = args.workers or max(1, mp.cpu_count() - 1)
        print_info(f"CPU cores:       {mp.cpu_count()}")
        print_info(f"Workers:         {num_workers}")

        if not shard_mode:
            print_info(f"Chunk size:      {args.chunk_size:,} lines")
            print_info(f"Write buffer:    {args.write_buffer:,} lines")
            print_info(f"Compress temp:   {args.compress_temp}")

        if checkpoint_dir:
            print_info(f"Checkpoint dir:  {checkpoint_dir}")
            print_info(f"Resume enabled:  {not args.no_resume}")

        # Estimates
        print()
        memory_per_worker = 100  # MB estimate
        total_memory = memory_per_worker * num_workers
        print_info(f"Est. memory:     {total_memory:.0f} MB peak")

        throughput_estimate = 20.0 * num_workers * 0.85  # MB/s
        time_estimate = (total_size / (1024 * 1024)) / throughput_estimate
        print_info(f"Est. time:       {format_duration(time_estimate)}")

        # Recommendations
        total_size_gb = total_size / (1024 ** 3)
        if total_size_gb > 100 and not checkpoint_dir:
            print()
            print_warning(f"Large dataset ({total_size_gb:.0f} GB) without checkpointing!")
            print_info("Recommend: --checkpoint-dir ./checkpoints")

        print()
    else:
        if shard_mode:
            print(f"[INFO] Processing {len(shard_files)} shards ({format_bytes(total_size)})")
        else:
            print(f"[INFO] Tokenizing {input_path.name} ({format_bytes(total_size)})")

    # Run tokenization
    try:
        if shard_mode:
            # SHARD MODE
            processor = ShardProcessor(
                model_path=model_path,
                num_workers=args.workers,
                quiet=args.quiet,
                debug=args.debug,
                log_file=log_file,
                checkpoint_dir=checkpoint_dir
            )

            stats = processor.process_shards(
                input_pattern=args.shards,
                output_dir=output_dir,
                output_path=output_path if args.merge else None,
                merge_output=args.merge,
                show_progress=not args.quiet,
                resume=not args.no_resume
            )

            # Normalize stats keys for common output
            stats['wall_time_seconds'] = stats.get('wall_time_seconds', 0)
            stats['throughput_mb_per_second'] = stats.get('throughput_mb_per_second', 0)
            failed_key = 'failed_shards'

        else:
            # SINGLE FILE MODE
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
            failed_key = 'failed_chunks'

        # Final summary
        if not args.quiet:
            print()
            print_header("Tokenization Completed Successfully")
            print()

            if shard_mode:
                if args.merge:
                    print_success(f"Output: {output_path}")
                else:
                    print_success(f"Output: {output_dir}/")
                print_info(f"Shards:       {stats.get('total_shards', 0)}")
            else:
                print_success(f"Output: {output_path}")

            print_info(f"Total tokens: {stats['total_tokens']:,}")
            print_info(f"Total lines:  {stats['total_lines']:,}")
            print_info(f"Throughput:   {stats['throughput_mb_per_second']:.2f} MB/s")
            print_info(f"Wall time:    {format_duration(stats['wall_time_seconds'])}")

            failed = stats.get(failed_key, 0)
            if failed > 0:
                print()
                print_warning(f"Failed: {failed}")
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