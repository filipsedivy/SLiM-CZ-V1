"""
Vocabulary management utilities.

Handles vocabulary loading, statistics, and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json

from ..preprocessing.base import (
    print_section,
    print_info,
    print_success
)


class VocabularyManager:
    """
    Manager for tokenizer vocabulary operations.
    
    Features:
    - Load vocabulary from JSON
    - Vocabulary statistics
    - Token frequency analysis
    - Special token management
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            vocab_path: Path to vocabulary JSON file (optional)
        """
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        
        if vocab_path:
            self.load(vocab_path)

    def load(self, vocab_path: Path):
        """
        Load vocabulary from JSON file.
        
        Args:
            vocab_path: Path to vocabulary JSON
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Create reverse mapping
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print_success(f"Loaded vocabulary from {vocab_path}")
        print_info(f"Vocabulary size: {len(self.vocab):,} tokens")

    def save(self, output_path: Path):
        """
        Save vocabulary to JSON file.
        
        Args:
            output_path: Path to save vocabulary
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        
        print_success(f"Vocabulary saved: {output_path}")

    def get_token(self, token_id: int) -> Optional[str]:
        """
        Get token string from ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string or None if not found
        """
        return self.reverse_vocab.get(token_id)

    def get_id(self, token: str) -> Optional[int]:
        """
        Get token ID from string.
        
        Args:
            token: Token string
            
        Returns:
            Token ID or None if not found
        """
        return self.vocab.get(token)

    def get_special_tokens(self) -> Dict[str, int]:
        """
        Get all special tokens.
        
        Returns:
            Dictionary of special tokens and their IDs
        """
        special = {}
        special_prefixes = ['<', '▁']  # Common special token prefixes
        
        for token, token_id in self.vocab.items():
            # Check if token starts with special prefix
            if any(token.startswith(prefix) for prefix in special_prefixes):
                # Additional check for user-defined tokens
                if token in ['<unk>', '<s>', '</s>', '<pad>', 
                             '<EMAIL>', '<PHONE>', '<URL>', 
                             '<IPADDR>', '<SSN>', '<DATE>', '<CREDITCARD>']:
                    special[token] = token_id
        
        return special

    def print_statistics(self):
        """Print vocabulary statistics."""
        print_section("Vocabulary Statistics")
        
        print_info(f"Total tokens: {len(self.vocab):,}")
        
        # Count special tokens
        special = self.get_special_tokens()
        print_info(f"Special tokens: {len(special)}")
        
        # Show special tokens
        if special:
            print("\n   Special Tokens:")
            for token, token_id in sorted(special.items(), key=lambda x: x[1])[:20]:
                print(f"      {token_id:5d}: {token}")
        
        # Token length distribution
        lengths = [len(token) for token in self.vocab.keys()]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            min_length = min(lengths)
            max_length = max(lengths)
            
            print(f"\n   Token Length Statistics:")
            print(f"      Average: {avg_length:.2f} characters")
            print(f"      Min:     {min_length} characters")
            print(f"      Max:     {max_length} characters")

    def analyze_coverage(self, text: str) -> Dict[str, float]:
        """
        Analyze vocabulary coverage on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with coverage statistics
        """
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {'coverage': 0.0, 'oov_count': 0}
        
        # Count words in vocabulary
        in_vocab = sum(1 for word in words if word in self.vocab)
        coverage = (in_vocab / total_words) * 100
        oov_count = total_words - in_vocab
        
        return {
            'coverage': coverage,
            'total_words': total_words,
            'in_vocab': in_vocab,
            'oov_count': oov_count,
            'oov_rate': (oov_count / total_words) * 100
        }

    def get_most_common_tokens(self, n: int = 20) -> List[tuple]:
        """
        Get most common tokens by ID (lower IDs = more common in BPE).
        
        Args:
            n: Number of tokens to return
            
        Returns:
            List of (token, id) tuples
        """
        # Sort by ID (lower = more common)
        sorted_tokens = sorted(self.vocab.items(), key=lambda x: x[1])
        return sorted_tokens[:n]

    def print_sample_tokens(self, n: int = 50):
        """
        Print sample of most common tokens.
        
        Args:
            n: Number of tokens to show
        """
        print_section(f"Sample Tokens (Most Common)")
        
        tokens = self.get_most_common_tokens(n)
        
        print("\n   Token ID | Token")
        print("   " + "-" * 60)
        
        for token, token_id in tokens:
            # Format token for display (handle special characters)
            display_token = token.replace('\n', '\\n').replace('\t', '\\t')
            if len(display_token) > 40:
                display_token = display_token[:37] + "..."
            
            print(f"   {token_id:8d} | {display_token}")
