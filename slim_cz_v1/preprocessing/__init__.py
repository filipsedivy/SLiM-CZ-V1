"""
Preprocessing module for SLiM-CZ-V1.

Provides modular pipeline for text extraction, cleaning, and anonymization.
"""

from .base import (
    BaseExtractor,
    BaseProcessor,
    PipelineRegistry,
    ProcessingResult,
    print_header,
    print_section,
    print_success,
    print_info,
    print_warning,
    print_error,
)
from .extractors import TxtExtractor, PdfExtractor, EpubExtractor
from .processors import (
    EncodingProcessor,
    CleaningProcessor,
    AdvancedCleaningProcessor,
    AnonymizationProcessor,
    AdvancedAnonymizationProcessor,
)
from .pipeline import TextExtractionPipeline

__all__ = [
    # Base classes
    'BaseExtractor',
    'BaseProcessor',
    'PipelineRegistry',
    'ProcessingResult',

    # Extractors
    'TxtExtractor',
    'PdfExtractor',
    'EpubExtractor',
    
    # Processors
    'EncodingProcessor',
    'CleaningProcessor',
    'AdvancedCleaningProcessor',
    'AnonymizationProcessor',
    'AdvancedAnonymizationProcessor',
    
    # Pipeline
    'TextExtractionPipeline',
    
    # Utilities
    'print_header',
    'print_section',
    'print_success',
    'print_info',
    'print_warning',
    'print_error',
]
