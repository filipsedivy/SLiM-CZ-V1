"""
EPUB file extractor.

Handles EPUB text extraction from eBook files with HTML/XHTML content parsing.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import re
import warnings

from ..base import BaseExtractor, print_warning, print_info

try:
    import ebooklib
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    warnings.warn("ebooklib not installed. Install with: pip install ebooklib")


class EpubExtractor(BaseExtractor):
    """
    Extractor for EPUB files (.epub).

    Features:
    - EPUB 2.0 and EPUB 3.0 support
    - HTML/XHTML content parsing
    - Automatic text extraction from all chapters
    - Metadata extraction
    - Image-only EPUB rejection

    Detection Method:
    ----------------
    Average Characters Per Document Analysis

    FORMULA:
        avg_chars_per_item = total_characters / document_count

    INTERPRETATION:
        - High average (> threshold) → text-based EPUB
        - Low average (≤ threshold) → image-only EPUB

    THRESHOLD:
        Default: 100 chars/item (empirical, conservative)

    RATIONALE:
        - Text-based EPUBs: typically 1000-10000+ chars/item
        - Image-only EPUBs: typically 0-50 chars/item
        - 100 chars/item catches most image-only books while avoiding false positives
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not EBOOKLIB_AVAILABLE:
            raise ImportError(
                "ebooklib is required for EPUB processing. "
                "Install with: pip install ebooklib"
            )

        # Empirical threshold for text-based EPUB detection
        self.min_chars_per_item = config.get('epub_min_chars_per_item', 100)

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is an EPUB."""
        return file_path.suffix.lower() == '.epub'

    def _extract_text_from_html(self, html_content: bytes) -> str:
        """
        Extract clean text from HTML/XHTML content.

        Process:
        1. Decode HTML bytes to string
        2. Remove script and style tags with content
        3. Remove all HTML tags
        4. Normalize whitespace
        5. Preserve paragraph structure

        Args:
            html_content: Raw HTML/XHTML bytes

        Returns:
            Extracted plain text
        """
        try:
            # Decode HTML content
            try:
                html_str = html_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try alternative encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        html_str = html_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # Last resort: replace errors
                    html_str = html_content.decode('utf-8', errors='replace')

            # Remove script tags and content
            html_str = re.sub(r'<script[^>]*>.*?</script>', '', html_str, flags=re.DOTALL | re.IGNORECASE)

            # Remove style tags and content
            html_str = re.sub(r'<style[^>]*>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)

            # Replace block-level tags with newlines (preserve paragraph structure)
            block_tags = r'</?(?:p|div|br|h[1-6]|li|tr|td|th)[^>]*>'
            html_str = re.sub(block_tags, '\n', html_str, flags=re.IGNORECASE)

            # Remove all remaining HTML tags
            html_str = re.sub(r'<[^>]+>', '', html_str)

            # Decode HTML entities
            html_str = html_str.replace('&nbsp;', ' ')
            html_str = html_str.replace('&amp;', '&')
            html_str = html_str.replace('&lt;', '<')
            html_str = html_str.replace('&gt;', '>')
            html_str = html_str.replace('&quot;', '"')
            html_str = html_str.replace('&#39;', "'")

            # Normalize whitespace
            # Replace multiple spaces with single space
            html_str = re.sub(r' +', ' ', html_str)

            # Replace multiple newlines with double newline (paragraph break)
            html_str = re.sub(r'\n\n+', '\n\n', html_str)

            # Remove leading/trailing whitespace from each line
            lines = [line.strip() for line in html_str.split('\n')]
            html_str = '\n'.join(lines)

            # Final cleanup
            html_str = html_str.strip()

            return html_str

        except Exception as e:
            print_warning(f"Failed to parse HTML content: {e}")
            return ""

    def extract(self, file_path: Path) -> Optional[str]:
        """
        Extract text from EPUB file.

        Process:
        1. Open EPUB archive
        2. Iterate through all document items
        3. Extract text from HTML/XHTML content
        4. Combine all chapters
        5. Validate text content (reject image-only EPUBs)

        Args:
            file_path: Path to EPUB file

        Returns:
            Extracted text or None if:
            - EPUB is image-only (scanned)
            - Extraction failed
            - Document is empty
        """
        try:
            # Open EPUB
            book = epub.read_epub(str(file_path))

            # Extract text from all document items
            texts = []
            item_count = 0
            total_chars = 0

            for item in book.get_items():
                # Process only document items (HTML/XHTML)
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    item_count += 1

                    # Extract text from HTML content
                    html_content = item.get_content()
                    text = self._extract_text_from_html(html_content)

                    if text.strip():
                        texts.append(text)
                        total_chars += len(text)

            # Check if EPUB contains sufficient text
            if item_count == 0:
                print_warning(f"Skipping {file_path.name}: No document items found")
                return None

            # CALCULATION: Average characters per item
            # FORMULA: avg_chars_per_item = total_chars / item_count

            avg_chars_per_item = total_chars / item_count if item_count > 0 else 0.0

            # Validate text content
            if avg_chars_per_item < self.min_chars_per_item:
                print_warning(
                    f"Skipping {file_path.name}: Image-only EPUB detected "
                    f"(avg {avg_chars_per_item:.1f} chars/item < {self.min_chars_per_item} threshold)"
                )
                return None

            if not texts:
                print_warning(f"No text extracted from {file_path.name}")
                return None

            # Combine all chapters
            full_text = "\n\n".join(texts)

            if not full_text.strip():
                print_warning(f"Empty text after extraction from {file_path.name}")
                return None

            # CALCULATION: Extraction statistics
            # FORMULA: items_with_text = len(texts)
            # FORMULA: extraction_ratio = items_with_text / item_count

            items_with_text = len(texts)
            extraction_ratio = items_with_text / item_count if item_count > 0 else 0.0

            print_info(
                f"Extracted from {file_path.name}: "
                f"{len(full_text):,} chars, "
                f"{item_count} items, "
                f"avg {avg_chars_per_item:.1f} chars/item, "
                f"extracted {items_with_text}/{item_count} items ({extraction_ratio * 100:.1f}%)"
            )

            return full_text

        except Exception as e:
            print_warning(f"Failed to extract text from {file_path.name}: {e}")
            return None