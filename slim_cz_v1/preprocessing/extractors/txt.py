"""
TXT file extractor.

Handles plain text file extraction with encoding detection.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import chardet

from ..base import BaseExtractor, print_warning


class TxtExtractor(BaseExtractor):
    """
    Extractor for plain text files (.txt).
    
    Features:
    - Automatic encoding detection
    - UTF-8 conversion
    - Error handling for malformed files
    """

    def can_extract(self, file_path: Path) -> bool:
        """Check if file is a text file."""
        return file_path.suffix.lower() == '.txt'

    def extract(self, file_path: Path) -> Optional[str]:
        """
        Extract text from TXT file with encoding detection.
        
        Process:
        1. Try UTF-8 first (most common)
        2. If fails, detect encoding using chardet
        3. Convert to UTF-8
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Extracted text in UTF-8 or None if extraction failed
        """
        try:
            # Try UTF-8 first (fast path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                pass

            # If UTF-8 fails, detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            # Detect encoding
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0.0)

            if confidence < 0.7:
                print_warning(
                    f"Low encoding confidence ({confidence:.2f}) for {file_path.name}, "
                    f"detected as {encoding}"
                )

            # Decode with detected encoding
            try:
                text = raw_data.decode(encoding)
                return text
            except Exception as decode_error:
                print_warning(
                    f"Failed to decode {file_path.name} with {encoding}: {decode_error}"
                )
                
                # Last resort: try with errors='replace'
                try:
                    text = raw_data.decode(encoding, errors='replace')
                    print_warning(f"Decoded {file_path.name} with character replacements")
                    return text
                except:
                    return None

        except Exception as e:
            print_warning(f"Failed to read {file_path.name}: {e}")
            return None