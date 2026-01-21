"""
PDF file extractor.

Handles PDF text extraction with detection of text-based vs image-based PDFs.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings

from ..base import BaseExtractor, print_warning, print_info

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    warnings.warn("PyMuPDF not installed. Install with: pip install pymupdf")


class PdfExtractor(BaseExtractor):
    """
    Extractor for PDF files (.pdf).
    
    Features:
    - Text-based vs image-based detection
    - Text extraction from text-based PDFs
    - Rejection of scanned/image-based PDFs (no OCR)
    
    Detection Method:
    ----------------
    Average Characters Per Page Analysis
    
    FORMULA:
        avg_chars_per_page = total_characters / page_count
        
    INTERPRETATION:
        - High average (> threshold) → text-based PDF
        - Low average (≤ threshold) → scanned/image-based PDF
        
    THRESHOLD:
        Default: 200 chars/page (empirical, conservative)
        
    RATIONALE:
        - Text-based PDFs: typically 500-5000+ chars/page
        - Scanned PDFs: typically 0-100 chars/page (OCR artifacts only)
        - 200 chars/page catches most scans while avoiding false positives
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install pymupdf"
            )
        
        # Empirical threshold for text-based PDF detection
        self.min_chars_per_page = config.get('pdf_min_chars_per_page', 200)

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == '.pdf'

    def is_text_based_pdf(self, doc: fitz.Document) -> Tuple[bool, float, int]:
        """
        Determine if PDF is text-based or image-based (scan).

        METHOD: Average Characters Per Page Analysis

        FORMULA:
            avg_chars_per_page = total_text_length / page_count

        VARIABLES:
            total_text_length: Total character count across all pages
            page_count: Number of pages in document

        THRESHOLD:
            min_chars_per_page: Empirical threshold (default: 200)

        Returns:
            Tuple[bool, float, int]: (is_text_based, avg_chars_per_page, total_chars)

        INTERPRETATION:
            - avg >= threshold → text-based PDF (extractable)
            - avg < threshold  → image-based PDF (scanned, requires OCR)
        """
        page_count = len(doc)
        if page_count == 0:
            return False, 0.0, 0

        # Extract text from all pages
        total_chars = 0
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            total_chars += len(text)

        # Calculate average
        avg_chars_per_page = total_chars / page_count

        # Determine if text-based
        is_text_based = avg_chars_per_page >= self.min_chars_per_page

        return is_text_based, avg_chars_per_page, total_chars

    def extract(self, file_path: Path) -> Optional[str]:
        """
        Extract text from PDF file.

        Process:
        1. Open PDF document
        2. Detect if text-based or image-based
        3. Extract text if text-based
        4. Reject if image-based (no OCR)

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text or None if:
            - PDF is image-based (scanned)
            - Extraction failed
            - Document is empty
        """
        doc = None
        try:
            # Open PDF
            doc = fitz.open(file_path)

            # Store page count before any operations
            page_count = len(doc)

            if page_count == 0:
                print_warning(f"Skipping {file_path.name}: Empty PDF (0 pages)")
                return None

            # Check if text-based
            is_text_based, avg_chars, total_chars = self.is_text_based_pdf(doc)

            if not is_text_based:
                print_warning(
                    f"Skipping {file_path.name}: Image-based PDF detected "
                    f"(avg {avg_chars:.1f} chars/page < {self.min_chars_per_page} threshold)"
                )
                return None

            # Extract text from all pages
            texts = []
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    texts.append(text)

            # Combine all pages
            full_text = "\n\n".join(texts)

            if not full_text.strip():
                print_warning(f"No text extracted from {file_path.name}")
                return None

            # CALCULATION: Extraction statistics
            # FORMULA: chars_per_page = total_chars / page_count
            # FORMULA: pages_with_text = len(texts)
            # FORMULA: extraction_ratio = pages_with_text / page_count

            pages_with_text = len(texts)
            extraction_ratio = pages_with_text / page_count if page_count > 0 else 0.0

            print_info(
                f"Extracted from {file_path.name}: "
                f"{len(full_text):,} chars, "
                f"{page_count} pages, "
                f"avg {avg_chars:.1f} chars/page, "
                f"extracted {pages_with_text}/{page_count} pages ({extraction_ratio * 100:.1f}%)"
            )

            return full_text

        except Exception as e:
            print_warning(f"Failed to extract text from {file_path.name}: {e}")
            return None

        finally:
            # CRITICAL: Always close document to free resources
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass  # Ignore errors during cleanup