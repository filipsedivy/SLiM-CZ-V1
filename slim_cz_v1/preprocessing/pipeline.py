"""
Main preprocessing pipeline orchestration.

Combines extractors and processors into a complete text preparation pipeline.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .base import (
    PipelineRegistry,
    ProcessingResult,
    ProgressBar,
    print_header,
    print_section,
    print_success,
    print_info,
    print_warning,
    print_error,
    Colors
)
from .extractors import TxtExtractor, PdfExtractor
from .processors import (
    EncodingProcessor,
    CleaningProcessor,
    AnonymizationProcessor
)


class TextExtractionPipeline:
    """
    Complete text extraction and preprocessing pipeline.
    
    Pipeline stages:
    1. File Discovery - Scan input directory
    2. Extraction - Extract text from files (TXT, PDF)
    3. Encoding - Ensure UTF-8 encoding
    4. Cleaning - Normalize whitespace, filter short lines
    5. Anonymization - Replace sensitive data with tokens (optional)
    6. Output - Save processed texts
    
    Configuration:
    - Input/output directories
    - Processor settings (min_line_length, anonymization, etc.)
    - Output format (individual files, corpus, or both)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = PipelineRegistry()
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Initialize and register extractors and processors."""

        # Register extractors
        self.registry.register_extractor(TxtExtractor(self.config))

        # Only register PDF extractor if PyMuPDF is available
        try:
            self.registry.register_extractor(PdfExtractor(self.config))
        except ImportError:
            print_warning("PDF extraction disabled (PyMuPDF not installed)")

        # Register processors in order
        # ORDER MATTERS: encoding → cleaning → anonymization

        # 1. Encoding validation/conversion (ALWAYS first)
        self.registry.register_processor(EncodingProcessor(self.config))

        # 2. Text cleaning
        self.registry.register_processor(CleaningProcessor(self.config))

        # 3. Anonymization (optional, ALWAYS last)
        if self.config.get('anonymize_emails') or \
           self.config.get('anonymize_phones') or \
           self.config.get('anonymize_urls'):
            self.registry.register_processor(AnonymizationProcessor(self.config))

    def collect_files(self, input_dir: Path) -> List[Path]:
        """
        Collect all supported files from input directory.

        Args:
            input_dir: Directory to scan

        Returns:
            List of file paths to process
        """
        supported_extensions = {'.txt', '.pdf'}
        files = []

        for ext in supported_extensions:
            files.extend(input_dir.rglob(f'*{ext}'))

        return sorted(files)

    def process_files(
        self,
        files: List[Path],
        input_dir: Path
    ) -> List[ProcessingResult]:
        """
        Process all files through the pipeline.

        Args:
            files: List of files to process
            input_dir: Base input directory

        Returns:
            List of processing results
        """
        results = []
        pbar = ProgressBar(len(files), "Processing files")

        for file_path in files:
            # Process file through pipeline
            processed_text = self.registry.process_file(file_path)

            # Create result
            success = processed_text is not None
            result = ProcessingResult(
                file_path=file_path,
                text=processed_text,
                success=success,
                error=None if success else "Processing failed"
            )
            results.append(result)

            pbar.update(1)

        pbar.close()

        return results

    def save_individual_files(
        self,
        results: List[ProcessingResult],
        input_dir: Path,
        output_dir: Path
    ):
        """
        Save processed texts as individual files (PRIMARY OUTPUT).

        Each processed file is saved with its original directory structure
        preserved relative to the input directory.

        Args:
            results: List of processing results
            input_dir: Input directory (for relative path calculation)
            output_dir: Output directory for individual files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # CALCULATION: Count successful saves
        saved_count = 0
        total_chars = 0

        for result in results:
            if not result.success or not result.text:
                continue

            # Create relative path structure
            relative_path = result.file_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix('.txt')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save processed text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.text)

            saved_count += 1
            total_chars += len(result.text)

        # CALCULATION: Success rate
        # FORMULA: success_rate = saved_count / total_results
        total_results = len(results)
        success_rate = saved_count / total_results if total_results > 0 else 0.0

        print_section("Individual Files Output")
        print_success(f"Saved {saved_count} individual files to {output_dir}")
        print_info(f"Success rate: {success_rate * 100:.1f}% ({saved_count}/{total_results})")
        print_info(f"Total characters: {total_chars:,}")

    def save_corpus_file(
        self,
        results: List[ProcessingResult],
        corpus_path: Path
    ):
        """
        Save all processed texts as single concatenated corpus file (OPTIONAL OUTPUT).

        Args:
            results: List of processing results
            corpus_path: Path to corpus file (including filename)
        """
        # Collect successful texts
        texts = [r.text for r in results if r.success and r.text]

        if not texts:
            print_warning("No texts to save to corpus file")
            return

        # CALCULATION: Corpus statistics
        # FORMULA: total_chars = sum(len(text) for text in texts)
        # FORMULA: avg_chars_per_file = total_chars / file_count

        file_count = len(texts)
        total_chars = sum(len(t) for t in texts)
        avg_chars_per_file = total_chars / file_count if file_count > 0 else 0

        # Create parent directory if needed
        corpus_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as single concatenated file
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(texts))

        print_section("Corpus File Output")
        print_success(f"Saved corpus: {corpus_path}")
        print_info(f"Files concatenated: {file_count}")
        print_info(f"Total characters: {total_chars:,}")
        print_info(f"Average chars per file: {avg_chars_per_file:,.0f}")

    def save_metadata(
        self,
        results: List[ProcessingResult],
        metadata_path: Path
    ):
        """
        Save processing metadata to JSON file.

        Args:
            results: List of processing results
            metadata_path: Path to metadata file
        """
        # CALCULATION: Processing statistics
        total_files = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_files - successful
        success_rate = successful / total_files if total_files > 0 else 0.0

        texts = [r.text for r in results if r.success and r.text]
        total_chars = sum(len(t) for t in texts)

        metadata = {
            'processing_statistics': {
                'total_files': total_files,
                'successful': successful,
                'failed': failed,
                'success_rate': round(success_rate, 4),
                'total_characters': total_chars
            },
            'files_processed': [
                {
                    'path': str(r.file_path),
                    'success': r.success,
                    'characters': len(r.text) if r.text else 0,
                    'error': r.error
                }
                for r in results
            ],
            'pipeline_config': self.config
        }

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print_success(f"Saved metadata: {metadata_path}")

    def run(self, input_dir: str, output_dir: str, corpus_file: Optional[str] = None):
        """
        Run complete extraction pipeline.

        Args:
            input_dir: Input directory with raw files
            output_dir: Output directory for individual processed files
            corpus_file: Optional path to save concatenated corpus
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        print_header("SLiM-CZ-V1 Text Extraction Pipeline")
        print(f"{Colors.BOLD}Slavic Linguistic integrated Micro-model for Czechia{Colors.ENDC}".center(70))

        print_section("Configuration")
        print_info(f"Input:  {input_dir}")
        print_info(f"Output: {output_dir} (individual files)")
        if corpus_file:
            print_info(f"Corpus: {corpus_file}")

        # Show pipeline configuration
        print_section("Pipeline Configuration")
        print(f"   {Colors.GREEN}✓{Colors.ENDC} UTF-8 Encoding Validation")
        print(f"   {Colors.GREEN}✓{Colors.ENDC} Text Cleaning (min line length: {self.config.get('min_line_length', 10)} chars)")

        if self.config.get('anonymize_emails'):
            print(f"   {Colors.GREEN}✓{Colors.ENDC} Email Anonymization → <EMAIL>")
        if self.config.get('anonymize_phones'):
            print(f"   {Colors.GREEN}✓{Colors.ENDC} Phone Anonymization → <PHONE>")
        if self.config.get('anonymize_urls'):
            print(f"   {Colors.GREEN}✓{Colors.ENDC} URL Anonymization → <URL>")

        # Collect files
        print_section("File Discovery")
        files = self.collect_files(input_path)

        if not files:
            print_error(f"No supported files found in {input_dir}")
            return

        # CALCULATION: File type distribution
        # FORMULA: txt_ratio = txt_files / total_files
        # FORMULA: pdf_ratio = pdf_files / total_files

        total_files = len(files)
        txt_files = sum(1 for f in files if f.suffix == '.txt')
        pdf_files = sum(1 for f in files if f.suffix == '.pdf')
        txt_ratio = txt_files / total_files if total_files > 0 else 0.0
        pdf_ratio = pdf_files / total_files if total_files > 0 else 0.0

        print_success(f"Found {total_files} files")
        print_info(f"TXT files: {txt_files} ({txt_ratio * 100:.1f}%)")
        print_info(f"PDF files: {pdf_files} ({pdf_ratio * 100:.1f}%)")

        # Process files
        print_section("Text Extraction & Processing")
        results = self.process_files(files, input_path)

        # CALCULATION: Processing results
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        success_rate = successful / len(results) if len(results) > 0 else 0.0

        print_section("Processing Results")
        print_success(f"Successfully processed: {successful}/{len(results)} files ({success_rate * 100:.1f}%)")
        if failed > 0:
            print_warning(f"Failed: {failed} files ({(1 - success_rate) * 100:.1f}%)")

        # Save outputs
        print_section("Saving Output")

        # PRIMARY OUTPUT: Individual files
        self.save_individual_files(results, input_path, output_path)

        # OPTIONAL OUTPUT: Corpus file
        if corpus_file:
            corpus_path = Path(corpus_file)
            self.save_corpus_file(results, corpus_path)

        # Save metadata
        metadata_path = output_path / 'extraction_metadata.json'
        self.save_metadata(results, metadata_path)

        print_header("Text Extraction Completed")

        print_section("Next Steps")
        if corpus_file:
            print(f"   1. Review corpus file: {corpus_file}")
            print(f"   2. Train tokenizer: slim-train-tokenizer --input {corpus_file}")
        else:
            print(f"   1. Review individual files in: {output_dir}")
            print(f"   2. Create corpus: slim-extract-text --input {input_dir} --output {output_dir} --output-corpus corpus.txt")
        print("=" * 70)