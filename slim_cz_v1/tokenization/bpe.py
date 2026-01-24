"""
BPE Tokenizer using SentencePiece.

Implements Byte-Pair Encoding for Czech text with optimizations for morphology.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

from ..preprocessing.base import (
    print_section,
    print_success,
    print_info,
    print_warning,
    print_error,
    ProgressBar
)

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    warnings.warn("SentencePiece not installed. Install with: pip install sentencepiece")


class BPETokenizer:
    """
    BPE Tokenizer for Czech text using SentencePiece.

    Features:
    - Byte-Pair Encoding (BPE) algorithm
    - Optimized for Czech morphology
    - High character coverage for diacritics
    - Vocabulary size optimization for limited data
    - Supports both single file and directory training (efficient for TB-scale datasets)

    Recommended Configuration:
    - Vocab size: 16,000-24,000 tokens (optimal for Czech with limited data)
    - Character coverage: 0.9999 (handles Czech diacritics)
    - Model type: BPE (better for morphologically rich languages)

    Research Background:
    - Smaller vocab (16-24k) works better with limited training data
    - BPE handles Czech inflection/declension better than WordPiece
    - High character coverage prevents unknown characters in Czech
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BPE tokenizer.

        Args:
            config: Configuration dictionary with:
                - vocab_size: Target vocabulary size (default: 16000)
                - character_coverage: Character coverage (default: 0.9999)
                - model_type: Model type (default: 'bpe')
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "SentencePiece is required for BPE tokenization. "
                "Install with: pip install sentencepiece"
            )

        self.config = config
        self.vocab_size = config.get('vocab_size', 16000)
        self.character_coverage = config.get('character_coverage', 0.9999)
        self.model_type = config.get('model_type', 'bpe')

        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self.model_prefix: Optional[str] = None

    def train(
        self,
        input_path: Path,
        output_dir: Path,
        model_prefix: str = 'tokenizer'
    ):
        """
        Train BPE tokenizer on input files.

        Supports both single file and directory modes:
        - Single file: Trains on one text file
        - Directory: Trains on all .txt files in directory (efficient for TB-scale datasets)

        SentencePiece streams files during training, so even large datasets
        don't need to fit in memory.

        Args:
            input_path: Path to text file OR directory containing .txt files
            output_dir: Directory to save model
            model_prefix: Prefix for model files (default: 'tokenizer')
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model_prefix = str(output_dir / model_prefix)

        print_section("BPE Tokenizer Training")
        print_info(f"Vocabulary size: {self.vocab_size:,}")
        print_info(f"Character coverage: {self.character_coverage}")
        print_info(f"Model type: {self.model_type}")

        # Prepare input for SentencePiece
        # SentencePiece natively supports:
        # - Single file path: 'file.txt'
        # - Multiple files: 'file1.txt,file2.txt,file3.txt' (comma-separated)
        if input_path.is_file():
            # Single file mode
            training_input = str(input_path)
            print_info(f"Training on single file: {input_path.name}")

        elif input_path.is_dir():
            # Directory mode - find all .txt files
            txt_files = sorted(input_path.rglob('*.txt'))
            if not txt_files:
                raise ValueError(f"No .txt files found in {input_path}")

            # SentencePiece accepts comma-separated file paths
            training_input = ','.join(str(f) for f in txt_files)
            print_info(f"Training on {len(txt_files)} text files from directory")

            # Show total size for large datasets
            total_size = sum(f.stat().st_size for f in txt_files)
            if total_size > 1024**3:  # > 1GB
                print_info(f"Total corpus size: {total_size / (1024**3):.2f} GB")

        else:
            raise ValueError(f"Input must be file or directory: {input_path}")

        # Train SentencePiece model
        print_info("Training tokenizer (this may take a few minutes)...")

        spm.SentencePieceTrainer.train(
            input=training_input,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            # Czech-specific optimizations
            user_defined_symbols=['<EMAIL>', '<PHONE>', '<URL>', '<IPADDR>', '<SSN>', '<DATE>', '<CREDITCARD>'],
            unk_id=0,  # Unknown token
            bos_id=1,  # Beginning of sequence
            eos_id=2,  # End of sequence
            pad_id=3,  # Padding
            # Training parameters
            num_threads=4,
            train_extremely_large_corpus=False,
            # Disable progress messages (we have our own)
            minloglevel=1,
        )

        print_success("Tokenizer trained successfully")

        # Load trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{self.model_prefix}.model")

        print_success(f"Model saved: {self.model_prefix}.model")
        print_success(f"Vocabulary saved: {self.model_prefix}.vocab")

        # Display statistics
        self._print_statistics()

    def _print_statistics(self):
        """Print tokenizer statistics."""
        if self.sp_model is None:
            return

        print_section("Tokenizer Statistics")
        print_info(f"Vocabulary size: {self.sp_model.vocab_size():,} tokens")
        print_info(f"Special tokens: <unk>, <s>, </s>, <pad>")
        print_info(f"User-defined tokens: <EMAIL>, <PHONE>, <URL>, etc.")

        # Show sample tokenization
        sample_texts = [
            "Dobrý den, jak se máte?",
            "Kontaktujte mě na email@example.com",
            "České národní kolo"
        ]

        print_section("Sample Tokenization")
        for text in sample_texts:
            tokens = self.sp_model.encode_as_pieces(text)
            print(f"   Text:   {text}")
            print(f"   Tokens: {' | '.join(tokens)}")
            print(f"   Count:  {len(tokens)} tokens")
            print()

    def load(self, model_path: Path):
        """
        Load pre-trained tokenizer.

        Args:
            model_path: Path to .model file
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(model_path))

        print_success(f"Loaded tokenizer from {model_path}")
        print_info(f"Vocabulary size: {self.sp_model.vocab_size():,} tokens")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        return self.sp_model.encode_as_ids(text)

    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Encode text to token pieces (subwords).

        Args:
            text: Input text

        Returns:
            List of token pieces
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        return self.sp_model.encode_as_pieces(text)

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        return self.sp_model.decode_ids(ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        return self.sp_model.vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary mapping.

        Returns:
            Dictionary mapping tokens to IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained or loaded")

        vocab = {}
        for idx in range(self.sp_model.vocab_size()):
            piece = self.sp_model.id_to_piece(idx)
            vocab[piece] = idx

        return vocab

    def save_vocab(self, output_path: Path):
        """
        Save vocabulary to JSON file.

        Args:
            output_path: Path to save vocabulary
        """
        import json

        vocab = self.get_vocab()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

        print_success(f"Vocabulary saved: {output_path}")