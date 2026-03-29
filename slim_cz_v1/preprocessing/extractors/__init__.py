"""
File extractors for different formats.
"""

from .epub import EpubExtractor
from .pdf import PdfExtractor
from .txt import TxtExtractor

__all__ = [
    "EpubExtractor",
    "PdfExtractor",
    "TxtExtractor",
]
