"""
File extractors for different formats.
"""

from .txt import TxtExtractor
from .pdf import PdfExtractor
from .epub import EpubExtractor

__all__ = [
    'TxtExtractor',
    'PdfExtractor',
    'EpubExtractor',
]
