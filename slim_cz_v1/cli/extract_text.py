#!/usr/bin/env python3
"""
CLI tool for text extraction and preprocessing.

Extracts and cleans text from various file formats (TXT, PDF).
"""

import argparse
import sys
from pathlib import Path

from ..preprocessing import TextExtractionPipeline, print_error, print_success, print_section


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

  # With anonymization
  slim-extract-text --input ./data/raw --output ./data/processed \\
    --anonymize-emails --anonymize-phones --anonymize-urls \\
    --output-corpus ./data/corpus.txt

  # Full configuration
  slim-extract-text --input ./data/raw --output ./data/processed \\
    --anonymize-emails --anonymize-phones --anonymize-urls \\
    --min-line-length 20 --pdf-min-chars-per-page 300 \\
    --output-corpus ./data/corpus.txt

Output Structure:
  --output: Directory for individual processed files
            Files maintain original directory structure
            Primary output - always created

  --output-corpus: Single concatenated corpus file
                   Optional - only created if specified
                   Recommended for tokenizer training
        """
    )

    # ============================================================
    # REQUIRED ARGUMENTS
    # ============================================================

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory with raw files (TXT, PDF)'
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
    # CONFIGURATION
    # ============================================================

    config = {
        'min_line_length': args.min_line_length,
        'pdf_min_chars_per_page': args.pdf_min_chars_per_page,
        'anonymize_emails': args.anonymize_emails,
        'anonymize_phones': args.anonymize_phones,
        'anonymize_urls': args.anonymize_urls,
    }

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