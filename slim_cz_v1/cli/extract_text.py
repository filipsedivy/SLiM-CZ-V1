#!/usr/bin/env python3
"""
CLI tool for text extraction and preprocessing.

Extracts and cleans text from various file formats (TXT, PDF, EPUB).
"""

import argparse
import sys
import os
from pathlib import Path

from ..preprocessing import TextExtractionPipeline


# ============================================================
# CLI OUTPUT FORMATTING
# ============================================================

class CLIFormatter:
    """CLI output formatter with ANSI color support."""

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


# ============================================================
# MAIN CLI FUNCTION
# ============================================================

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

    # Required arguments
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
        help='Output directory for individual processed files'
    )

    # Optional arguments - output
    parser.add_argument(
        '--output-corpus', '-c',
        type=str,
        default=None,
        help='Optional: Path to save concatenated corpus file'
    )

    # Optional arguments - performance
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, recommended: 4-8)'
    )

    # Optional arguments - text processing
    parser.add_argument(
        '--min-line-length',
        type=int,
        default=10,
        help='Minimum line length in characters (default: 10)'
    )

    # Optional arguments - PDF processing
    parser.add_argument(
        '--pdf-min-chars-per-page',
        type=float,
        default=200.0,
        help='PDF minimum average chars/page (default: 200.0)'
    )

    # Optional arguments - anonymization
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
        CLIFormatter.error(f"Input directory not found: {args.input}")
        return 1

    if not input_path.is_dir():
        CLIFormatter.error(f"Input path is not a directory: {args.input}")
        return 1

    # ============================================================
    # WORKER COUNT VALIDATION
    # ============================================================

    cpu_cores = os.cpu_count() or 4
    max_workers = args.max_workers

    if max_workers < 1:
        CLIFormatter.error(f"Invalid worker count: {max_workers}. Must be >= 1.")
        return 1

    if max_workers > 2 * cpu_cores:
        CLIFormatter.warning(f"Specified workers ({max_workers}) exceeds 2x CPU cores ({cpu_cores})")
        CLIFormatter.info(f"Recommended maximum: {2 * cpu_cores} workers")
        CLIFormatter.info("High worker count may cause overhead and reduce performance")
        print()

    # ============================================================
    # HEADER
    # ============================================================

    CLIFormatter.header("SLiM-CZ-V1 Text Extraction Pipeline")

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

    CLIFormatter.section("Configuration")

    CLIFormatter.metric("Input directory:", str(input_path))
    CLIFormatter.metric("Output directory:", str(args.output))

    if args.output_corpus:
        CLIFormatter.metric("Corpus file:", args.output_corpus)
    else:
        CLIFormatter.metric("Corpus file:", "Not created")

    CLIFormatter.metric("Min line length:", f"{args.min_line_length} chars")
    CLIFormatter.metric("PDF min chars/page:", f"{args.pdf_min_chars_per_page:.1f}")

    # ============================================================
    # PERFORMANCE ESTIMATE
    # ============================================================

    if max_workers > 1:
        CLIFormatter.section("Parallel Processing")

        CLIFormatter.metric("Workers:", str(max_workers))
        CLIFormatter.metric("Available CPU cores:", str(cpu_cores))

        # Speedup calculation
        # Formula: speedup ≈ min(workers, cores) * efficiency
        if max_workers <= 4:
            efficiency = 0.85
        elif max_workers <= 8:
            efficiency = 0.75
        else:
            efficiency = 0.65

        effective_workers = min(max_workers, cpu_cores)
        estimated_speedup = effective_workers * efficiency

        CLIFormatter.metric("Expected speedup:", f"~{estimated_speedup:.1f}x vs sequential")
        CLIFormatter.info(f"Formula: speedup = min(workers, cores) * efficiency")
        CLIFormatter.info(f"Calculation: {effective_workers} * {efficiency:.2f} = {estimated_speedup:.1f}x")
    else:
        CLIFormatter.section("Sequential Processing")
        CLIFormatter.info("Running with 1 worker (sequential mode)")
        CLIFormatter.info("Use --max-workers 4 or higher for parallel processing")

    # ============================================================
    # ANONYMIZATION NOTICE
    # ============================================================

    if args.anonymize_emails or args.anonymize_phones or args.anonymize_urls:
        CLIFormatter.section("Anonymization")
        CLIFormatter.info("Sensitive information will be replaced with tokens:")
        if args.anonymize_emails:
            print("    • Email addresses → <EMAIL>")
        if args.anonymize_phones:
            print("    • Phone numbers → <PHONE>")
        if args.anonymize_urls:
            print("    • URLs → <URL>")

    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================

    try:
        CLIFormatter.section("Extraction Progress")

        pipeline = TextExtractionPipeline(config)
        pipeline.run(args.input, args.output, args.output_corpus)

        CLIFormatter.header("Extraction Completed")

        CLIFormatter.section("Output Files")
        CLIFormatter.metric("Processed files:", str(args.output))
        if args.output_corpus:
            CLIFormatter.metric("Corpus file:", args.output_corpus)

        print()
        CLIFormatter.section("Next Steps")
        print("  1. Train tokenizer:")
        print(f"     slim-train-tokenizer --input {args.output_corpus or args.output} \\")
        print("       --output ./models/tokenizer --vocab-size 16000")
        print()
        print("  2. Tokenize corpus:")
        print("     slim-tokenize-parallel --input <corpus> \\")
        print("       --model ./models/tokenizer/tokenizer.model \\")
        print("       --output ./data/tokens.txt")

        print()
        CLIFormatter.success("Text extraction completed successfully")
        print("=" * 70)
        print()

        return 0

    except Exception as e:
        CLIFormatter.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())