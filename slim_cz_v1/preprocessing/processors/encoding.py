"""
Encoding validation and conversion processor.

Ensures all text is in UTF-8 encoding.
"""

from typing import Dict, Any
import chardet

from ..base import BaseProcessor, print_warning


class EncodingProcessor(BaseProcessor):
    """
    Processor for UTF-8 encoding validation and conversion.
    
    Features:
    - UTF-8 validation
    - Automatic encoding detection for non-UTF-8 text
    - Conversion to UTF-8
    - Error handling with character replacement
    
    This processor ensures all text going through the pipeline
    is in UTF-8 encoding, which is required for:
    - Consistent tokenization
    - Proper handling of Czech diacritics
    - Data serialization (JSON, etc.)
    """

    def get_name(self) -> str:
        return "EncodingProcessor"

    def process(self, text: str) -> str:
        """
        Validate and ensure UTF-8 encoding.
        
        Process:
        1. Try to encode as UTF-8
        2. If successful, return original text
        3. If fails, detect encoding and convert
        
        Args:
            text: Input text (potentially non-UTF-8)
            
        Returns:
            Text guaranteed to be in UTF-8 encoding
        """
        try:
            # Test if text is valid UTF-8
            text.encode('utf-8')
            return text
            
        except UnicodeEncodeError:
            # Text is not valid UTF-8, need to convert
            print_warning("Non-UTF-8 text detected, attempting conversion")
            
            try:
                # Convert string to bytes for encoding detection
                # Try common encodings first
                for encoding in ['latin-1', 'cp1252', 'iso-8859-2']:
                    try:
                        # Encode with suspected encoding, then decode as UTF-8
                        byte_data = text.encode(encoding)
                        converted_text = byte_data.decode('utf-8')
                        return converted_text
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        continue
                
                # If common encodings fail, use chardet
                byte_data = text.encode('latin-1', errors='replace')
                detected = chardet.detect(byte_data)
                encoding = detected.get('encoding', 'utf-8')
                confidence = detected.get('confidence', 0.0)
                
                print_warning(
                    f"Detected encoding: {encoding} "
                    f"(confidence: {confidence:.2f})"
                )
                
                # Convert to UTF-8
                converted_text = byte_data.decode(encoding, errors='replace')
                return converted_text
                
            except Exception as e:
                print_warning(f"Encoding conversion failed: {e}")
                # Last resort: replace problematic characters
                return text.encode('utf-8', errors='replace').decode('utf-8')
