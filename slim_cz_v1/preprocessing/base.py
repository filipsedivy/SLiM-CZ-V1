"""
Base classes and utilities for data preprocessing pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys
import time


# ============================================================
# ANSI COLOR UTILITIES
# ============================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ============================================================
# TERMINAL OUTPUT UTILITIES
# ============================================================

class ProgressBar:
    """Simple progress bar for console output."""

    def __init__(self, total: int, desc: str = "", width: int = 50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress bar."""
        self.current += n
        self._draw()

    def _draw(self):
        """Draw progress bar."""
        if self.total == 0:
            return

        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"

        sys.stdout.write(f'\r{self.desc} |{bar}| {self.current}/{self.total} ({percent * 100:.1f}%) {eta_str}')
        sys.stdout.flush()

        if self.current >= self.total:
            print()  # New line when complete

    def close(self):
        """Complete the progress bar."""
        self.current = self.total
        self._draw()


def print_header(text: str):
    """Print formatted header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"{text:^{width}}")
    print(f"{'=' * width}")


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 70)


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


# ============================================================
# BASE PIPELINE CLASSES
# ============================================================

class BaseExtractor(ABC):
    """
    Base class for file extractors.
    
    Extractors handle reading raw content from different file formats.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def can_extract(self, file_path: Path) -> bool:
        """Check if this extractor can process the given file."""
        pass

    @abstractmethod
    def extract(self, file_path: Path) -> Optional[str]:
        """
        Extract raw text from the file.
        
        Returns:
            Raw text content or None if extraction failed.
        """
        pass


class BaseProcessor(ABC):
    """
    Base class for text processors.
    
    Processors transform text (cleaning, anonymization, encoding, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process the input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this processor for logging."""
        pass


class PipelineRegistry:
    """
    Registry for managing extractors and processors.
    
    Allows dynamic registration and execution of processing steps.
    """

    def __init__(self):
        self.extractors: List[BaseExtractor] = []
        self.processors: List[BaseProcessor] = []

    def register_extractor(self, extractor: BaseExtractor):
        """Register a file extractor."""
        self.extractors.append(extractor)

    def register_processor(self, processor: BaseProcessor):
        """Register a text processor."""
        self.processors.append(processor)

    def get_extractor(self, file_path: Path) -> Optional[BaseExtractor]:
        """Find appropriate extractor for the file."""
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None

    def process_text(self, text: str) -> str:
        """
        Run text through all registered processors in order.
        
        Args:
            text: Input text
            
        Returns:
            Processed text after all processors
        """
        result = text
        for processor in self.processors:
            result = processor.process(result)
        return result

    def process_file(self, file_path: Path) -> Optional[str]:
        """
        Complete file processing: extraction → processing.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Processed text or None if processing failed
        """
        try:
            # Step 1: Extract raw text
            extractor = self.get_extractor(file_path)
            if extractor is None:
                print_warning(f"No extractor found for {file_path.suffix}")
                return None

            raw_text = extractor.extract(file_path)
            if raw_text is None or not raw_text.strip():
                print_warning(f"No text extracted from {file_path.name}")
                return None

            # Step 2: Process text through all processors
            processed_text = self.process_text(raw_text)

            if not processed_text.strip():
                print_warning(f"Text empty after processing: {file_path.name}")
                return None

            return processed_text

        except Exception as e:
            print_warning(f"Error processing {file_path.name}: {e}")
            return None


# ============================================================
# DATA STRUCTURES
# ============================================================

class ProcessingResult:
    """
    Result of file processing operation.
    """

    def __init__(self, file_path: Path, text: Optional[str], success: bool, error: Optional[str] = None):
        self.file_path = file_path
        self.text = text
        self.success = success
        self.error = error

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"ProcessingResult({self.file_path.name}, {status})"