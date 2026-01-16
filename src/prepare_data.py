"""
Data Preparation Pipeline for SLiM-CZ-V1
Slavic Linguistic integrated Micro-model for Czechia

Uses SentencePiece BPE tokenizer optimized for Czech language.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    warnings.warn("SentencePiece not installed. Install with: pip install sentencepiece")


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


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
        
        sys.stdout.write(f'\r{self.desc} |{bar}| {self.current}/{self.total} ({percent*100:.1f}%) {eta_str}')
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


class CzechTextPreprocessor:
    """Preprocessor for Czech text data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        """Clean text while preserving Czech characters."""
        
        # Remove URLs
        if self.config.get('remove_urls', True):
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if self.config.get('remove_emails', True):
            text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Remove very short lines
        min_length = self.config.get('min_line_length', 10)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) >= min_length]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def process_file(self, file_path: Path) -> str:
        """Process a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            text = self.clean_text(text)
            return text
        
        except Exception as e:
            print_warning(f"Error processing {file_path}: {e}")
            return ""


class SentencePieceBPETokenizer:
    """SentencePiece BPE tokenizer optimized for Czech language."""
    
    def __init__(self, vocab_size: int = 16000):
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece is required. Install with: pip install sentencepiece")
        
        self.vocab_size = vocab_size
        self.model = None
        self.model_prefix = None
    
    def train(
        self,
        texts: List[str],
        model_prefix: str,
        character_coverage: float = 0.9999,
        model_type: str = 'bpe'
    ):
        """Train SentencePiece BPE tokenizer on Czech text."""
        self.model_prefix = model_prefix
        
        # Save texts to temporary file
        temp_file = f"{model_prefix}_temp.txt"
        print_section("📝 Preparing training data")
        
        pbar = ProgressBar(len(texts), "Writing texts")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                if text.strip():
                    f.write(text + '\n')
                pbar.update(1)
        pbar.close()
        
        print_section("🔧 Training SentencePiece BPE tokenizer")
        print_info(f"Vocab size: {self.vocab_size}")
        print_info(f"Character coverage: {character_coverage}")
        print_info(f"Model type: {model_type}")
        print_info("Training in progress...")
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            
            # Special tokens
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            
            # Czech-specific settings
            normalization_rule_name='identity',
            remove_extra_whitespaces=True,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            
            # Training parameters
            num_threads=4,
            max_sentence_length=16384,
        )
        
        # Load trained model
        self.model = spm.SentencePieceProcessor()
        self.model.load(f'{model_prefix}.model')
        
        # Clean up temporary file
        Path(temp_file).unlink(missing_ok=True)
        
        print_success(f"Tokenizer trained! Actual vocab size: {self.model.get_piece_size()}")
        print_success(f"Model saved: {model_prefix}.model")
        
        # Print sample tokens
        print_info("Sample Czech tokens:")
        sample_tokens = []
        for i in range(min(20, self.model.get_piece_size())):
            token = self.model.id_to_piece(i)
            sample_tokens.append(f"{i}:{token}")
        print("   " + ", ".join(sample_tokens[:10]))
    
    def load(self, model_path: str):
        """Load trained tokenizer."""
        self.model = spm.SentencePieceProcessor()
        self.model.load(model_path)
        self.vocab_size = self.model.get_piece_size()
        print_success(f"Tokenizer loaded: {model_path}")
        print_info(f"Vocab size: {self.vocab_size}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.model.encode(text, out_type=int)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.model.decode(ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.get_piece_size() if self.model else self.vocab_size
    
    def save_vocab(self, output_path: str):
        """Save vocabulary to JSON."""
        vocab = {}
        for i in range(self.model.get_piece_size()):
            vocab[i] = self.model.id_to_piece(i)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        print_success(f"Vocabulary saved: {output_path}")


class DataPreparationPipeline:
    """Complete data preparation pipeline for SLiM-CZ-V1."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = CzechTextPreprocessor(config)
        self.tokenizer = None
        
    def collect_files(self, input_dir: Path) -> List[Path]:
        """Collect all text files."""
        supported_extensions = {
            '.txt', '.md', '.rst', '.py', '.js', '.html', 
            '.css', '.json', '.xml', '.csv', '.log',
            '.c', '.cpp', '.java', '.cs', '.php'
        }
        
        files = []
        for ext in supported_extensions:
            files.extend(input_dir.rglob(f'*{ext}'))
        
        return files
    
    def process_files(self, files: List[Path]) -> List[str]:
        """Process all files and return cleaned texts."""
        print_section("🔄 Processing files")
        
        texts = []
        pbar = ProgressBar(len(files), "Processing")
        
        for file_path in files:
            text = self.preprocessor.process_file(file_path)
            if text:
                texts.append(text)
            pbar.update(1)
        
        pbar.close()
        
        # Calculate statistics
        total_chars = sum(len(text) for text in texts)
        print_success(f"Processed {len(texts)} files")
        print_info(f"Total characters: {total_chars:,}")
        if texts:
            print_info(f"Average chars/file: {total_chars // len(texts):,}")
        
        return texts
    
    def train_tokenizer(self, texts: List[str], output_dir: Path):
        """Train SentencePiece BPE tokenizer."""
        vocab_size = self.config.get('vocab_size', 16000)
        model_prefix = str(output_dir / 'tokenizer')
        
        self.tokenizer = SentencePieceBPETokenizer(vocab_size)
        self.tokenizer.train(
            texts=texts,
            model_prefix=model_prefix,
            character_coverage=0.9999,
            model_type='bpe'
        )
        
        # Save vocabulary
        self.tokenizer.save_vocab(str(output_dir / 'vocab.json'))
        
        return self.tokenizer
    
    def create_sequences(self, texts: List[str]) -> List[List[int]]:
        """Create training sequences with proper stride."""
        seq_len = self.config.get('seq_len', 512)
        stride = self.config.get('stride', 256)
        
        print_section("📊 Creating sequences")
        print_info(f"Sequence length: {seq_len}")
        print_info(f"Stride: {stride}")
        
        all_sequences = []
        total_tokens = 0
        
        pbar = ProgressBar(len(texts), "Tokenizing")
        
        for text in texts:
            # Tokenize
            token_ids = self.tokenizer.encode(text)
            total_tokens += len(token_ids)
            
            # Create overlapping sequences
            for i in range(0, len(token_ids) - seq_len + 1, stride):
                sequence = token_ids[i:i + seq_len]
                if len(sequence) == seq_len:
                    all_sequences.append(sequence)
            
            pbar.update(1)
        
        pbar.close()
        
        print_success(f"Created {len(all_sequences)} sequences")
        print_info(f"Total tokens: {total_tokens:,}")
        print_info(f"Tokens per sequence: {seq_len}")
        print_warning(f"Recommended epochs: 20-50 (based on research)")
        
        return all_sequences
    
    def split_data(self, sequences: List[List[int]]) -> Dict[str, List[List[int]]]:
        """Split data into train/val/test sets."""
        import random
        
        print_section("✂️  Splitting data")
        
        random.shuffle(sequences)
        
        train_ratio = self.config.get('train_split', 0.90)
        val_ratio = self.config.get('val_split', 0.05)
        
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': sequences[:train_end],
            'val': sequences[train_end:val_end],
            'test': sequences[val_end:]
        }
        
        print_info(f"Train: {len(splits['train'])} sequences ({train_ratio*100:.1f}%)")
        print_info(f"Val:   {len(splits['val'])} sequences ({val_ratio*100:.1f}%)")
        print_info(f"Test:  {len(splits['test'])} sequences ({(1-train_ratio-val_ratio)*100:.1f}%)")
        
        return splits
    
    def save_data(self, splits: Dict[str, List[List[int]]], output_dir: Path):
        """Save processed data."""
        print_section("💾 Saving data")
        
        for split_name, sequences in splits.items():
            output_path = output_dir / f'{split_name}.json'
            with open(output_path, 'w') as f:
                json.dump(sequences, f)
            print_success(f"{split_name}.json: {len(sequences)} sequences")
        
        # Save statistics
        stats = {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'seq_len': self.config.get('seq_len', 512),
            'stride': self.config.get('stride', 256),
            'train_sequences': len(splits['train']),
            'val_sequences': len(splits['val']),
            'test_sequences': len(splits['test']),
            'total_sequences': sum(len(s) for s in splits.values()),
            'tokenizer_type': 'sentencepiece_bpe',
            'character_coverage': 0.9999,
            'recommended_epochs': '20-50',
            'config': self.config
        }
        
        with open(output_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print_success("stats.json")
    
    def run(self, input_dir: str, output_dir: str):
        """Run complete pipeline."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print_header("🇨🇿 SLiM-CZ-V1 Data Preparation")
        print(f"{Colors.BOLD}Slavic Linguistic integrated Micro-model for Czechia{Colors.ENDC}".center(70))
        
        print_section("📋 Configuration")
        print_info(f"Input:  {input_dir}")
        print_info(f"Output: {output_dir}")
        print_info(f"Vocab:  {self.config.get('vocab_size', 16000):,} tokens")
        
        print_section("✨ Research-Based Optimizations")
        print(f"   {Colors.GREEN}✓{Colors.ENDC} SentencePiece BPE tokenizer")
        print(f"   {Colors.GREEN}✓{Colors.ENDC} 16-24k vocab size for limited data")
        print(f"   {Colors.GREEN}✓{Colors.ENDC} Optimized for Czech morphology")
        print(f"   {Colors.GREEN}✓{Colors.ENDC} Character coverage 0.9999 for diacritics")
        
        # 1. Collect files
        print_section("📁 Collecting files")
        files = self.collect_files(input_path)
        if not files:
            print_error(f"No files found in {input_dir}")
            return
        print_success(f"Found {len(files)} files")
        
        # 2. Process files
        texts = self.process_files(files)
        if not texts:
            print_error("No valid text extracted from files")
            return
        
        # 3. Train tokenizer
        self.train_tokenizer(texts, output_path)
        
        # 4. Create sequences
        sequences = self.create_sequences(texts)
        if not sequences:
            print_error("No sequences created")
            return
        
        # 5. Split data
        splits = self.split_data(sequences)
        
        # 6. Save everything
        self.save_data(splits, output_path)
        
        print_header("✅ Data preparation completed!")
        
        print_section("📊 Summary")
        print(f"   Tokenizer:  SentencePiece BPE")
        print(f"   Vocabulary: {self.tokenizer.get_vocab_size():,} tokens")
        print(f"   Sequences:  {len(sequences):,} total")
        print(f"   Output dir: {output_dir}")
        
        print_section("💡 Next steps")
        print(f"   1. Review stats.json for dataset statistics")
        print(f"   2. Train model: python train.py --data-dir {output_dir}")
        print(f"   3. Use 20-50 epochs as recommended by research")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='🇨🇿 SLiM-CZ-V1 Data Preparation (SentencePiece BPE)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, required=True, help='Input directory with text files')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--vocab-size', type=int, default=16000, help='Vocabulary size (default: 16000)')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length (default: 512)')
    parser.add_argument('--stride', type=int, default=256, help='Stride for sequences (default: 256)')
    parser.add_argument('--train-split', type=float, default=0.90)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--test-split', type=float, default=0.05)
    parser.add_argument('--remove-urls', action='store_true', default=True)
    parser.add_argument('--remove-emails', action='store_true', default=True)
    parser.add_argument('--min-line-length', type=int, default=10)
    
    args = parser.parse_args()
    
    # Check SentencePiece availability
    if not SENTENCEPIECE_AVAILABLE:
        print_error("SentencePiece is not installed!")
        print_info("Install with: pip install sentencepiece")
        return
    
    # Create configuration
    config = {
        'vocab_size': args.vocab_size,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'remove_urls': args.remove_urls,
        'remove_emails': args.remove_emails,
        'min_line_length': args.min_line_length,
    }
    
    # Run pipeline
    pipeline = DataPreparationPipeline(config)
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
