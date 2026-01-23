"""
Text cleaning processor.

Handles whitespace normalization, line filtering, and basic cleaning.
"""

import re
from typing import Dict, Any

from ..base import BaseProcessor


class CleaningProcessor(BaseProcessor):
    """
    Processor for text cleaning operations.
    
    Features:
    - Whitespace normalization
    - Excessive newline removal
    - Short line filtering
    - Preserves Czech diacritics
    - Preserves sentence structure
    
    Configuration:
    - min_line_length: Minimum characters per line (default: 10)
    """

    def get_name(self) -> str:
        return "CleaningProcessor"

    def process(self, text: str) -> str:
        """
        Clean text while preserving linguistic structure.
        
        Operations:
        1. Normalize horizontal whitespace (spaces, tabs)
        2. Remove excessive newlines (keep max 2 consecutive)
        3. Filter out very short lines
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        min_line_length = self.config.get('min_line_length', 10)
        
        # 1. Normalize horizontal whitespace
        # Replace tabs and multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 2. Normalize vertical whitespace
        # Keep at most 2 consecutive newlines (paragraph breaks)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # 3. Filter short lines
        # Remove lines that are too short (likely noise/headers/footers)
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Keep empty lines (paragraph breaks) or lines meeting minimum length
            if not stripped or len(stripped) >= min_line_length:
                filtered_lines.append(line)
        
        text = '\n'.join(filtered_lines)
        
        # 4. Final cleanup
        # Strip leading/trailing whitespace from entire text
        text = text.strip()
        
        return text


class AdvancedCleaningProcessor(BaseProcessor):
    """
    Advanced text cleaning with more aggressive filtering.
    
    Additional features beyond CleaningProcessor:
    - Remove repeated characters (e.g., "!!!!!!" → "!")
    - Remove lines with excessive punctuation
    - Remove lines with unusual character ratios
    
    Use this for noisy data sources (web scraping, OCR, etc.)
    """

    def get_name(self) -> str:
        return "AdvancedCleaningProcessor"

    def process(self, text: str) -> str:
        """
        Perform advanced text cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # 1. Remove excessive repeated characters
        # e.g., "Hello!!!!!" → "Hello!"
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)
        
        # 2. Filter lines by character composition
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines (keep them)
            if not stripped:
                filtered_lines.append(line)
                continue
            
            # Calculate character type ratios
            total_chars = len(stripped)
            alpha_chars = sum(c.isalpha() for c in stripped)
            punct_chars = sum(c in '.,!?;:' for c in stripped)
            
            # Skip lines with too much punctuation (> 40%)
            if total_chars > 0 and (punct_chars / total_chars) > 0.4:
                continue
            
            # Skip lines with too few alphabetic characters (< 40%)
            if total_chars > 0 and (alpha_chars / total_chars) < 0.4:
                continue
            
            filtered_lines.append(line)
        
        text = '\n'.join(filtered_lines)
        
        # 3. Final cleanup
        text = text.strip()
        
        return text