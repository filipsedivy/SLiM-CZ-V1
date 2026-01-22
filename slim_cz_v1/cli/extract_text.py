#!/usr/bin/env python3
"""
CLI tool for text extraction and preprocessing.

Extracts and cleans text from various file formats (TXT, PDF, EPUB).
"""

import argparse
import sys
import os
from pathlib import Path

from ..preprocessing import TextExtractionPipeline, print_error, print_success, print_section, print_info


def main():
    """Main entry point for text extraction CLI."""

    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 Text Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (individual files only)
  slim-extract-text --input ./data/raw --output ./data/processed

  # With corpus file creation
  slim-extract-text --input ./data/raw --output ./data/processed \\
    --output-corpus ./data/corpus.txt

  # With parallel processing (faster)
  slim-extract-text --input ./data/raw --output ./data/processed \\
    --max-workers 8

  # With anonymization
  slim-extract-text --input ./data/raw --output ./data/processed \\
    --anonymize-emails --anonymize-phones --anonymize-urls \\
    --output-corpus ./data/corpus.txt

  # Full configuration
  slim-extract-text --input ./data/raw --output ./data/processed \\
    --anonymize-emails --anonymize-phones --anonymize-urls \\
    --min-line-length 20 --pdf-min-chars-per-page 300 \\
    --max-workers 8 \\
    --output-corpus ./data/corpus.txt

Output Structure:
  --output: Directory for individual processed files
            Files maintain original directory structure
            Primary output - always created

  --output-corpus: Single concatenated corpus file
                   Optional - only created if specified
                   Recommended for tokenizer training

Performance:
  --max-workers: Controls parallel processing
                 Default: 4 workers
                 Recommended: 4-8 for most systems
                 Expected speedup: 3-7x faster than sequential
        """
    )

    # ============================================================
    # REQUIRED ARGUMENTS
    # ============================================================

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory with raw files (TXT, PDF, EPUB)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for individual processed files (maintains directory structure)'
    )

    # ============================================================
    # OPTIONAL ARGUMENTS - OUTPUT
    # ============================================================

    parser.add_argument(
        '--output-corpus', '-c',
        type=str,
        default=None,
        help='Optional: Path to save concatenated corpus file (e.g., corpus.txt)'
    )

    # ============================================================
    # OPTIONAL ARGUMENTS - PERFORMANCE
    # ============================================================

    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers for file processing (default: 4). '
             'Recommended: 4-8 for most systems. Higher values = faster processing. '
             'Set to 1 for sequential processing (debugging).'
    )

    # ============================================================
    # OPTIONAL ARGUMENTS - TEXT PROCESSING
    # ============================================================

    parser.add_argument(
        '--min-line-length',
        type=int,
        default=10,
        help='Minimum line length in characters (default: 10)'
    )

    # ============================================================
    # OPTIONAL ARGUMENTS - PDF PROCESSING
    # ============================================================

    parser.add_argument(
        '--pdf-min-chars-per-page',
        type=float,
        default=200.0,
        help='PDF minimum average chars/page for text-based detection (default: 200.0)'
    )

    # ============================================================
    # OPTIONAL ARGUMENTS - ANONYMIZATION
    # ============================================================

    parser.add_argument(
        '--anonymize-emails',
        action='store_true',
        help='Replace email addresses with <EMAIL> token'
    )

    parser.add_argument(
        '--anonymize-phones',
        action='store_true',
        help='Replace phone numbers with <PHONE> token'
    )

    parser.add_argument(
        '--anonymize-urls',
        action='store_true',
        help='Replace URLs with <URL> token'
    )

    args = parser.parse_args()

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input directory not found: {args.input}")
        return 1

    if not input_path.is_dir():
        print_error(f"Input path is not a directory: {args.input}")
        return 1

    # ============================================================
    # WORKER COUNT VALIDATION & OPTIMIZATION
    # ============================================================

    cpu_cores = os.cpu_count() or 4
    max_workers = args.max_workers

    # Validate worker count
    if max_workers < 1:
        print_error(f"Invalid worker count: {max_workers}. Must be >= 1.")
        return 1

    # Warn if worker count is suboptimal
    if max_workers > 2 * cpu_cores:
        print_section("Performance Warning")
        print_info(f"Specified workers ({max_workers}) exceeds 2x CPU cores ({cpu_cores})")
        print_info(f"Recommended maximum: {2 * cpu_cores} workers")
        print_info("High worker count may cause overhead and reduce performance")
        print()

    # ============================================================
    # CONFIGURATION
    # ============================================================

    config = {
        'min_line_length': args.min_line_length,
        'pdf_min_chars_per_page': args.pdf_min_chars_per_page,
        'anonymize_emails': args.anonymize_emails,
        'anonymize_phones': args.anonymize_phones,
        'anonymize_urls': args.anonymize_urls,
        'max_workers': max_workers,
    }

    # ============================================================
    # PERFORMANCE NOTICE
    # ============================================================

    if max_workers > 1:
        print_section("Parallel Processing Enabled")
        print_info(f"Workers: {max_workers}")
        print_info(f"Available CPU cores: {cpu_cores}")

        # Estimate speedup based on worker count
        # FORMULA: estimated_speedup ≈ min(workers, cores) * efficiency
        # WHERE: efficiency ≈ 0.85 for 4 workers, 0.75 for 8 workers
        if max_workers <= 4:
            efficiency = 0.85
        elif max_workers <= 8:
            efficiency = 0.75
        else:
            efficiency = 0.65

        effective_workers = min(max_workers, cpu_cores)
        estimated_speedup = effective_workers * efficiency

        print_info(f"Expected speedup: ~{estimated_speedup:.1f}x faster than sequential")
        print()
    else:
        print_section("Sequential Processing Mode")
        print_info("Running with 1 worker (sequential mode)")
        print_info("Use --max-workers 4 or higher for parallel processing")
        print()

    # ============================================================
    # ANONYMIZATION NOTICE
    # ============================================================

    if args.anonymize_emails or args.anonymize_phones or args.anonymize_urls:
        print_section("Anonymization Notice")
        print("   Sensitive information will be replaced with tokens:")
        if args.anonymize_emails:
            print("      • Email addresses → <EMAIL>")
        if args.anonymize_phones:
            print("      • Phone numbers → <PHONE>")
        if args.anonymize_urls:
            print("      • URLs → <URL>")
        print()

    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================

    try:
        pipeline = TextExtractionPipeline(config)
        pipeline.run(args.input, args.output, args.output_corpus)

        print_success("Text extraction completed successfully")
        return 0

    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())