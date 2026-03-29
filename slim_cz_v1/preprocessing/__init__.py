"""
Preprocessing module for SLiM-CZ-V1.

Provides modular pipeline for text extraction, cleaning, and anonymization.
"""

from .base import (
    BaseExtractor,
    BaseProcessor,
    PipelineRegistry,
    ProcessingResult,
    print_error,
    print_header,
    print_info,
    print_section,
    print_success,
    print_warning,
)
from .extractors import EpubExtractor, PdfExtractor, TxtExtractor
from .pipeline import TextExtractionPipeline
from .processors import (
    AdvancedAnonymizationProcessor,
    AdvancedCleaningProcessor,
    AnonymizationProcessor,
    CleaningProcessor,
    EncodingProcessor,
)

__all__ = [
    "AdvancedAnonymizationProcessor",
    "AdvancedCleaningProcessor",
    "AnonymizationProcessor",
    # Base classes
    "BaseExtractor",
    "BaseProcessor",
    "CleaningProcessor",
    # Processors
    "EncodingProcessor",
    "EpubExtractor",
    "PdfExtractor",
    "PipelineRegistry",
    "ProcessingResult",
    # Pipeline
    "TextExtractionPipeline",
    # Extractors
    "TxtExtractor",
    "print_error",
    # Utilities
    "print_header",
    "print_info",
    "print_section",
    "print_success",
    "print_warning",
]
