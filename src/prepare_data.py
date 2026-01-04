"""
SLiM-CZ-V1 Data Preparation Script
Processes text files and prepares data for training with proper special token handling.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from collections import Counter
import re

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed - some features will be unavailable")


# ============================================================================
# Configuration
# ============================================================================

SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.rst', '.py', '.js', '.html', '.css',
    '.json', '.xml', '.csv', '.log', '.c', '.cpp', '.java'
}

# Special tokens
SPECIAL_TOKENS = {
    'PAD': '<pad>',
    'UNK': '<unk>',
    'BOS': '<bos>',
    'EOS': '<eos>',
}


# ============================================================================
# Text Cleaner
# ============================================================================

class TextCleaner:
    """Cleans and normalizes text for LLM training."""

    def __init__(self,
                 lowercase=False,
                 remove_urls=True,
                 remove_emails=True,
                 remove_extra_whitespace=True,
                 min_line_length=10):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_line_length = min_line_length

    def clean(self, text: str) -> str:
        """Clean text."""
        if not text:
            return ""

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)

        # Remove short lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) >= self.min_line_length]
        text = '\n'.join(lines)

        return text.strip()


# ============================================================================
# File Collector
# ============================================================================

class FileCollector:
    """Collects text files from specified directories."""

    def __init__(self, supported_extensions=SUPPORTED_EXTENSIONS):
        self.supported_extensions = supported_extensions
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'total_chars': 0,
            'files_by_extension': Counter()
        }

    def collect_files(self, root_dir: str, recursive=True) -> List[str]:
        """Collect all supported files from directory."""
        root_path = Path(root_dir)

        if not root_path.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        files = []

        print(f"\n📂 Scanning: {root_dir}")
        print(f"   Recursive: {'Yes' if recursive else 'No'}")

        if recursive:
            # Recursive scan
            for file_path in root_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in self.supported_extensions:
                    files.append(str(file_path))
                    self.stats['files_by_extension'][file_path.suffix] += 1
        else:
            # Current directory only
            for file_path in root_path.glob('*'):
                if file_path.is_file() and file_path.suffix in self.supported_extensions:
                    files.append(str(file_path))
                    self.stats['files_by_extension'][file_path.suffix] += 1

        self.stats['total_files'] = len(files)

        print(f"   ✅ Files found: {len(files)}")

        return sorted(files)

    def read_file(self, file_path: str, encoding='utf-8') -> str:
        """Read file content."""
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            self.stats['processed_files'] += 1
            self.stats['total_chars'] += len(content)
            return content
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            self.stats['skipped_files'] += 1
            return ""

    def print_stats(self):
        """Print statistics."""
        print("\n" + "=" * 70)
        print("📊 FILE STATISTICS")
        print("=" * 70)
        print(f"Total found:        {self.stats['total_files']}")
        print(f"Successfully read:  {self.stats['processed_files']}")
        print(f"Skipped:            {self.stats['skipped_files']}")
        print(f"Total characters:   {self.stats['total_chars']:,}")
        print(f"\nFiles by type:")
        for ext, count in sorted(self.stats['files_by_extension'].items()):
            print(f"  {ext}: {count}")
        print("=" * 70)


# ============================================================================
# Simple Tokenizer (Character-level)
# ============================================================================

class SimpleTokenizer:
    """
    Simple tokenizer for SLiM-CZ-V1 with proper special token handling.
    For production use, we recommend using the 'tokenizers' library.
    """

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = []

        # Add special tokens (these must be added first to get IDs 0-3)
        for token in SPECIAL_TOKENS.values():
            self._add_token(token)

        # Store special token IDs for quick access
        self.pad_id = self.token_to_id[SPECIAL_TOKENS['PAD']]
        self.unk_id = self.token_to_id[SPECIAL_TOKENS['UNK']]
        self.bos_id = self.token_to_id[SPECIAL_TOKENS['BOS']]
        self.eos_id = self.token_to_id[SPECIAL_TOKENS['EOS']]

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.vocab.append(token)
            return token_id
        return self.token_to_id[token]

    def train(self, texts: List[str], min_frequency=2):
        """
        Train tokenizer on texts.
        Uses character-level tokenization for simplicity.
        """
        print("\n🔤 Training tokenizer...")

        # Collect all characters
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)

        # Select most frequent characters
        most_common = char_freq.most_common(self.vocab_size - len(SPECIAL_TOKENS))

        for char, freq in most_common:
            if freq >= min_frequency:
                self._add_token(char)

        print(f"   ✅ Vocabulary contains {len(self.vocab)} tokens")
        print(f"   Special tokens: PAD={self.pad_id}, UNK={self.unk_id}, BOS={self.bos_id}, EOS={self.eos_id}")
        print(f"   Most frequent tokens: {list(self.token_to_id.keys())[:20]}")

    def encode(self, text: str, add_special_tokens=False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: If True, adds BOS at start and EOS at end

        Returns:
            List of token IDs
        """
        token_ids = [self.token_to_id.get(char, self.unk_id) for char in text]

        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]

        return token_ids

    def encode_document(self, text: str) -> List[int]:
        """
        Encode a complete document with BOS and EOS tokens.
        This is the recommended method for encoding training documents.

        Args:
            text: Document text

        Returns:
            List of token IDs with BOS at start and EOS at end
        """
        return self.encode(text, add_special_tokens=True)

    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, removes special tokens from output

        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special tokens
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        return ''.join([self.id_to_token.get(tid, SPECIAL_TOKENS['UNK']) for tid in token_ids])

    def pad_sequence(self, token_ids: List[int], max_length: int, pad_value: Optional[int] = None) -> List[int]:
        """
        Pad sequence to specified length.

        Args:
            token_ids: List of token IDs
            max_length: Target length
            pad_value: Padding value (uses PAD token if None)

        Returns:
            Padded sequence
        """
        if pad_value is None:
            pad_value = self.pad_id

        if len(token_ids) >= max_length:
            return token_ids[:max_length]

        return token_ids + [pad_value] * (max_length - len(token_ids))

    def save(self, path: str):
        """Save tokenizer."""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'vocab': self.vocab,
            'special_token_ids': {
                'pad_id': self.pad_id,
                'unk_id': self.unk_id,
                'bos_id': self.bos_id,
                'eos_id': self.eos_id,
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Tokenizer saved: {path}")

    @classmethod
    def load(cls, path: str):
        """Load tokenizer."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.vocab = data['vocab']

        # Restore special token IDs
        if 'special_token_ids' in data:
            tokenizer.pad_id = data['special_token_ids']['pad_id']
            tokenizer.unk_id = data['special_token_ids']['unk_id']
            tokenizer.bos_id = data['special_token_ids']['bos_id']
            tokenizer.eos_id = data['special_token_ids']['eos_id']

        return tokenizer


# ============================================================================
# Dataset Creator
# ============================================================================

class DatasetCreator:
    """Creates training sequences from tokenized text with proper special token handling."""

    def __init__(self, tokenizer: SimpleTokenizer, seq_len=512, stride=256,
                 use_document_boundaries=True, pad_sequences=False):
        """
        Initialize dataset creator.

        Args:
            tokenizer: Trained tokenizer
            seq_len: Sequence length
            stride: Stride for sliding window
            use_document_boundaries: If True, adds BOS/EOS tokens to each document
            pad_sequences: If True, pads sequences to seq_len
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.use_document_boundaries = use_document_boundaries
        self.pad_sequences = pad_sequences

    def create_sequences(self, text: str) -> List[List[int]]:
        """
        Create overlapping sequences from text.

        Args:
            text: Input text (one document)

        Returns:
            List of token ID sequences
        """
        # Encode text with document boundaries (BOS/EOS)
        if self.use_document_boundaries:
            token_ids = self.tokenizer.encode_document(text)
        else:
            token_ids = self.tokenizer.encode(text)

        # Create sequences with stride
        sequences = []

        # If text is too short, just return it as one sequence
        if len(token_ids) <= self.seq_len + 1:
            if self.pad_sequences:
                seq = self.tokenizer.pad_sequence(token_ids, self.seq_len + 1)
                sequences.append(seq)
            elif len(token_ids) > 1:  # Need at least 2 tokens (input + label)
                sequences.append(token_ids)
            return sequences

        # Create overlapping windows
        for i in range(0, len(token_ids) - self.seq_len, self.stride):
            seq = token_ids[i:i + self.seq_len + 1]  # +1 for labels

            if self.pad_sequences:
                seq = self.tokenizer.pad_sequence(seq, self.seq_len + 1)

            if len(seq) == self.seq_len + 1:
                sequences.append(seq)

        return sequences

    def prepare_dataset(
        self,
        texts: List[str],
        output_dir: str,
        split_ratio: Tuple[float, float, float] = (0.9, 0.05, 0.05)
    ) -> Dict:
        """
        Prepare complete dataset with train/val/test splits.

        Args:
            texts: List of text documents (each text = one document/file)
            output_dir: Output directory
            split_ratio: (train, val, test) split ratios

        Returns:
            Dictionary with statistics
        """
        print(f"\n   Creating sequences (seq_len={self.seq_len}, stride={self.stride})...")
        print(f"   Document boundaries: {'Enabled (BOS/EOS)' if self.use_document_boundaries else 'Disabled'}")
        print(f"   Padding: {'Enabled' if self.pad_sequences else 'Disabled'}")

        # Create all sequences
        all_sequences = []
        document_stats = {
            'total_documents': len(texts),
            'sequences_per_doc': [],
            'tokens_per_doc': []
        }

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"      Processed: {i + 1}/{len(texts)}")

            sequences = self.create_sequences(text)
            all_sequences.extend(sequences)

            # Track statistics
            document_stats['sequences_per_doc'].append(len(sequences))
            if self.use_document_boundaries:
                token_count = len(self.tokenizer.encode_document(text))
            else:
                token_count = len(self.tokenizer.encode(text))
            document_stats['tokens_per_doc'].append(token_count)

        print(f"   ✅ Created {len(all_sequences)} sequences from {len(texts)} documents")
        print(f"      Avg sequences/doc: {sum(document_stats['sequences_per_doc'])/len(texts):.1f}")
        print(f"      Avg tokens/doc: {sum(document_stats['tokens_per_doc'])/len(texts):.1f}")

        # Shuffle
        random.shuffle(all_sequences)

        # Split
        train_ratio, val_ratio, test_ratio = split_ratio
        n_total = len(all_sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_seq = all_sequences[:n_train]
        val_seq = all_sequences[n_train:n_train + n_val]
        test_seq = all_sequences[n_train + n_val:]

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'train.json', 'w') as f:
            json.dump(train_seq, f)

        with open(output_path / 'val.json', 'w') as f:
            json.dump(val_seq, f)

        with open(output_path / 'test.json', 'w') as f:
            json.dump(test_seq, f)

        # Statistics
        stats = {
            'total_sequences': len(all_sequences),
            'train_sequences': len(train_seq),
            'val_sequences': len(val_seq),
            'test_sequences': len(test_seq),
            'seq_len': self.seq_len,
            'vocab_size': self.tokenizer.vocab_size,
            'split_ratio': split_ratio,
            'use_document_boundaries': self.use_document_boundaries,
            'pad_sequences': self.pad_sequences,
            'document_stats': document_stats
        }

        with open(output_path / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"   ✅ Dataset saved:")
        print(f"      Train: {len(train_seq)} sequences")
        print(f"      Val:   {len(val_seq)} sequences")
        print(f"      Test:  {len(test_seq)} sequences")

        return stats


# ============================================================================
# Main Pipeline
# ============================================================================

class DataPreparationPipeline:
    """Complete data preparation pipeline."""

    def __init__(self, config):
        self.config = config
        self.collector = FileCollector()
        self.cleaner = TextCleaner(
            lowercase=config.get('lowercase', False),
            remove_urls=config.get('remove_urls', True),
            remove_emails=config.get('remove_emails', True),
            min_line_length=config.get('min_line_length', 10)
        )
        self.tokenizer = None
        self.dataset_creator = None

    def run(self, input_dir: str, output_dir: str):
        """Run complete pipeline."""
        print("=" * 70)
        print("🚀 SLiM-CZ-V1 DATA PREPARATION PIPELINE")
        print("=" * 70)
        print(f"📂 Input:  {input_dir}")
        print(f"📁 Output: {output_dir}")
        print("=" * 70)

        # 1. Collect files
        print("\n[1/5] 📂 Collecting files...")
        files = self.collector.collect_files(
            input_dir,
            recursive=self.config.get('recursive', True)
        )

        if not files:
            print("❌ No files found!")
            return

        # 2. Read and clean texts
        print("\n[2/5] 🧹 Cleaning texts...")
        texts = []
        for i, file_path in enumerate(files):
            if (i + 1) % 50 == 0:
                print(f"   Processed: {i + 1}/{len(files)}")

            content = self.collector.read_file(file_path)
            if content:
                cleaned = self.cleaner.clean(content)
                if cleaned:
                    texts.append(cleaned)

        print(f"   ✅ Processed {len(texts)} texts")

        # Statistics
        self.collector.print_stats()

        # 3. Train tokenizer
        print("\n[3/5] 🔤 Training tokenizer...")
        self.tokenizer = SimpleTokenizer(vocab_size=self.config.get('vocab_size', 10000))
        self.tokenizer.train(texts, min_frequency=self.config.get('min_frequency', 2))

        # Save tokenizer
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_path / 'tokenizer.json'))

        # 4. Create dataset
        print("\n[4/5] 📊 Creating dataset...")
        self.dataset_creator = DatasetCreator(
            self.tokenizer,
            seq_len=self.config.get('seq_len', 512),
            stride=self.config.get('stride', 256),
            use_document_boundaries=self.config.get('use_document_boundaries', True),
            pad_sequences=self.config.get('pad_sequences', False)
        )

        stats = self.dataset_creator.prepare_dataset(
            texts,
            output_dir,
            split_ratio=self.config.get('split_ratio', (0.9, 0.05, 0.05))
        )

        # 5. Save configuration
        print("\n[5/5] 💾 Saving configuration...")
        config_path = output_path / 'data_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"   ✅ Configuration saved: {config_path}")

        # Final report
        print("\n" + "=" * 70)
        print("🎉 DATA PREPARATION COMPLETE!")
        print("=" * 70)
        print(f"📊 Statistics:")
        print(f"   Files:                {self.collector.stats['processed_files']}")
        print(f"   Total chars:          {self.collector.stats['total_chars']:,}")
        print(f"   Vocabulary:           {len(self.tokenizer.vocab)} tokens")
        print(f"   Special tokens:       PAD, UNK, BOS, EOS")
        print(f"   Document boundaries:  {'Yes (BOS/EOS added)' if self.config.get('use_document_boundaries', True) else 'No'}")
        print(f"   Sequence padding:     {'Yes' if self.config.get('pad_sequences', False) else 'No'}")
        print(f"   Train sequences:      {stats['train_sequences']}")
        print(f"   Val sequences:        {stats['val_sequences']}")
        print(f"   Test sequences:       {stats['test_sequences']}")
        print(f"\n📁 Outputs in: {output_dir}")
        print(f"   ├── tokenizer.json")
        print(f"   ├── train.json")
        print(f"   ├── val.json")
        print(f"   ├── test.json")
        print(f"   ├── stats.json")
        print(f"   └── data_config.json")
        print("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 Data Preparation Script (Improved)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic usage with document boundaries (recommended)
  python prepare_data_improved.py --input ./raw_data --output ./processed_data

  # Without document boundaries (legacy mode)
  python prepare_data_improved.py --input ./raw_data --output ./processed_data \\
    --no-document-boundaries

  # With sequence padding enabled
  python prepare_data_improved.py --input ./raw_data --output ./processed_data \\
    --pad-sequences

  # With custom configuration
  python prepare_data_improved.py --input ./raw_data --output ./processed_data \\
    --vocab-size 30000 --seq-len 1024 --lowercase

  # Current directory only (no recursion)
  python prepare_data_improved.py --input ./raw_data --output ./processed_data \\
    --no-recursive
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory with text files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--vocab-size',
        type=int,
        default=10000,
        help='Vocabulary size (default: 10000)'
    )

    parser.add_argument(
        '--seq-len',
        type=int,
        default=512,
        help='Sequence length (default: 512)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=256,
        help='Stride for sequence creation (default: 256)'
    )

    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum token frequency (default: 2)'
    )

    parser.add_argument(
        '--min-line-length',
        type=int,
        default=10,
        help='Minimum line length (default: 10)'
    )

    parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=3,
        default=[0.9, 0.05, 0.05],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Data split ratio (default: 0.9 0.05 0.05)'
    )

    parser.add_argument(
        '--lowercase',
        action='store_true',
        help='Convert text to lowercase'
    )

    parser.add_argument(
        '--no-remove-urls',
        action='store_true',
        help='Do not remove URLs'
    )

    parser.add_argument(
        '--no-remove-emails',
        action='store_true',
        help='Do not remove emails'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not scan directories recursively'
    )

    parser.add_argument(
        '--no-document-boundaries',
        action='store_true',
        help='Do not add BOS/EOS tokens to documents (not recommended)'
    )

    parser.add_argument(
        '--pad-sequences',
        action='store_true',
        help='Pad sequences to fixed length (useful for batch processing)'
    )

    args = parser.parse_args()

    # Create configuration
    config = {
        'vocab_size': args.vocab_size,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'min_frequency': args.min_frequency,
        'min_line_length': args.min_line_length,
        'split_ratio': tuple(args.split_ratio),
        'lowercase': args.lowercase,
        'remove_urls': not args.no_remove_urls,
        'remove_emails': not args.no_remove_emails,
        'recursive': not args.no_recursive,
        'use_document_boundaries': not args.no_document_boundaries,
        'pad_sequences': args.pad_sequences,
    }

    # Run pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    main()
