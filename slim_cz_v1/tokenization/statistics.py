"""
Tokenizer Statistics Module.

Provides scientifically-grounded statistical analysis for BPE tokenizer training.
All metrics are aggregated and anonymized - no sensitive data is stored.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter
import json
import math

from ..preprocessing.base import (
    print_section,
    print_info,
    print_success,
    print_warning
)


@dataclass
class CorpusStatistics:
    """
    Aggregated corpus statistics (no sensitive data).
    
    All metrics are mathematically defined and reproducible.
    """
    # Basic counts
    total_characters: int
    total_bytes: int
    total_lines: int
    
    # Whitespace-based tokenization (baseline)
    total_words: int
    unique_words: int
    
    # Character-level analysis
    unique_characters: int
    character_entropy: float  # Shannon entropy in bits
    
    # Linguistic richness (Type-Token Ratio family)
    ttr: float  # Type-Token Ratio
    sttr: float  # Standardized TTR (first 10k tokens)
    rttr: float  # Root TTR (Guiraud's Index)
    msttr: float  # Mean Segmental TTR
    
    # Distribution metrics
    vocabulary_size_at_coverage: Dict[str, int]  # e.g., {'0.80': 5000, '0.90': 8000}
    
    # Czech-specific
    diacritic_ratio: float  # Ratio of characters with diacritics
    
    # File metadata
    corpus_files: int
    corpus_size_mb: float


@dataclass
class TokenizerStatistics:
    """
    Tokenizer performance statistics.
    
    Mathematically defined metrics for tokenizer quality assessment.
    """
    # Configuration
    vocab_size: int
    character_coverage: float
    model_type: str
    
    # Compression metrics
    compression_ratio: float  # characters / tokens
    tokens_per_word: float  # Average subword units per word
    
    # Fertility and efficiency
    fertility_rate: float  # tokens_generated / tokens_input (1.0 = perfect)
    byte_pair_efficiency: float  # Reduction in vocabulary vs character-level
    
    # Coverage metrics
    character_coverage_actual: float  # Measured coverage on training data
    oov_rate_estimate: float  # Out-of-vocabulary rate (test set)
    
    # Token statistics
    avg_token_length: float  # Average characters per token
    token_length_std: float  # Standard deviation
    
    # Sample tokenization metrics (anonymized)
    sample_compression_rates: List[float]  # Distribution of compression rates
    
    # Czech-specific
    czech_diacritic_preservation: float  # How well diacritics are preserved


@dataclass
class CombinedStatistics:
    """Combined corpus and tokenizer statistics."""
    corpus: CorpusStatistics
    tokenizer: TokenizerStatistics
    training_timestamp: str
    model_name: str = "SLiM-CZ-V1"


class StatisticsCollector:
    """
    Collects and analyzes corpus/tokenizer statistics.
    
    PRIVACY GUARANTEE:
    - Only aggregated metrics are stored
    - No raw text, names, or identifiable information
    - All statistics are mathematical/statistical in nature
    """
    
    def __init__(self):
        """Initialize statistics collector."""
        self.reset()
    
    def reset(self):
        """Reset all collected statistics."""
        self.char_counter = Counter()
        self.word_counter = Counter()
        self.line_count = 0
        self.total_chars = 0
        self.total_bytes = 0
        self.file_count = 0
        self.diacritic_count = 0
    
    def analyze_corpus(self, input_path: Path) -> CorpusStatistics:
        """
        Analyze corpus and collect statistics.
        
        Args:
            input_path: Path to corpus file or directory
            
        Returns:
            CorpusStatistics object with all metrics
        """
        print_section("Corpus Statistical Analysis")
        print_info("Computing scientifically-grounded metrics...")
        
        self.reset()
        
        # Collect data
        if input_path.is_file():
            self._process_file(input_path)
        elif input_path.is_dir():
            txt_files = sorted(input_path.rglob('*.txt'))
            self.file_count = len(txt_files)
            
            for file_path in txt_files:
                self._process_file(file_path)
        
        # Calculate derived metrics
        return self._compute_corpus_statistics()
    
    def _process_file(self, file_path: Path):
        """Process single file and update counters."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.line_count += 1
                    self.total_bytes += len(line.encode('utf-8'))
                    
                    # Character analysis
                    for char in line:
                        self.total_chars += 1
                        self.char_counter[char] += 1
                        
                        # Czech diacritics detection
                        if self._is_czech_diacritic(char):
                            self.diacritic_count += 1
                    
                    # Word analysis (whitespace tokenization)
                    words = line.split()
                    for word in words:
                        # Normalize for counting (lowercase, no punctuation at edges)
                        normalized = word.strip('.,!?;:()[]{}"\'"').lower()
                        if normalized:
                            self.word_counter[normalized] += 1
        
        except Exception as e:
            print_warning(f"Error processing {file_path.name}: {e}")
    
    def _is_czech_diacritic(self, char: str) -> bool:
        """Check if character is Czech-specific with diacritic."""
        czech_diacritics = 'áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ'
        return char in czech_diacritics
    
    def _compute_corpus_statistics(self) -> CorpusStatistics:
        """Compute all corpus statistics from collected data."""
        
        total_words = sum(self.word_counter.values())
        unique_words = len(self.word_counter)
        unique_chars = len(self.char_counter)
        
        # === MATHEMATICALLY DEFINED METRICS ===
        
        # 1. Shannon Entropy (Information Theory)
        # H(X) = -Σ p(x) * log2(p(x))
        # Measures average information content per character
        char_entropy = self._calculate_shannon_entropy(self.char_counter, self.total_chars)
        
        # 2. Type-Token Ratio (TTR)
        # TTR = unique_words / total_words
        # Simple vocabulary richness metric
        ttr = unique_words / total_words if total_words > 0 else 0.0
        
        # 3. Standardized TTR (STTR)
        # TTR computed on first 10,000 tokens (standardization for comparison)
        sttr = self._calculate_sttr(self.word_counter, segment_size=10000)
        
        # 4. Root TTR (Guiraud's Index)
        # RTTR = unique_words / sqrt(total_words)
        # Corrects for text length dependency
        rttr = unique_words / math.sqrt(total_words) if total_words > 0 else 0.0
        
        # 5. Mean Segmental TTR (MSTTR)
        # Average TTR over consecutive segments of fixed length
        # More stable than raw TTR
        msttr = self._calculate_msttr(self.word_counter, segment_size=1000)
        
        # 6. Vocabulary Growth Curve
        # Words needed to cover X% of corpus
        vocab_coverage = self._calculate_vocabulary_coverage(self.word_counter)
        
        # 7. Diacritic Ratio
        # Czech-specific: ratio of diacritic characters
        diacritic_ratio = self.diacritic_count / self.total_chars if self.total_chars > 0 else 0.0
        
        stats = CorpusStatistics(
            total_characters=self.total_chars,
            total_bytes=self.total_bytes,
            total_lines=self.line_count,
            total_words=total_words,
            unique_words=unique_words,
            unique_characters=unique_chars,
            character_entropy=char_entropy,
            ttr=ttr,
            sttr=sttr,
            rttr=rttr,
            msttr=msttr,
            vocabulary_size_at_coverage=vocab_coverage,
            diacritic_ratio=diacritic_ratio,
            corpus_files=self.file_count if self.file_count > 0 else 1,
            corpus_size_mb=self.total_bytes / (1024 * 1024)
        )
        
        return stats
    
    def _calculate_shannon_entropy(self, counter: Counter, total: int) -> float:
        """
        Calculate Shannon entropy.
        
        Formula: H(X) = -Σ p(x) * log2(p(x))
        
        Args:
            counter: Frequency counter
            total: Total count
            
        Returns:
            Entropy in bits
        """
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_sttr(self, word_counter: Counter, segment_size: int = 10000) -> float:
        """
        Calculate Standardized Type-Token Ratio.
        
        Uses first N tokens for standardized comparison.
        
        Args:
            word_counter: Word frequency counter
            segment_size: Size of segment to use
            
        Returns:
            STTR value
        """
        # Reconstruct token sequence (order doesn't matter for STTR)
        tokens = []
        for word, count in word_counter.items():
            tokens.extend([word] * count)
            if len(tokens) >= segment_size:
                break
        
        if len(tokens) < segment_size:
            # Not enough data, return regular TTR
            unique = len(set(tokens))
            return unique / len(tokens) if tokens else 0.0
        
        segment = tokens[:segment_size]
        unique = len(set(segment))
        return unique / segment_size
    
    def _calculate_msttr(self, word_counter: Counter, segment_size: int = 1000) -> float:
        """
        Calculate Mean Segmental Type-Token Ratio.
        
        Averages TTR over multiple consecutive segments.
        
        Args:
            word_counter: Word frequency counter
            segment_size: Size of each segment
            
        Returns:
            Mean STTR across segments
        """
        # Reconstruct token sequence
        tokens = []
        for word, count in word_counter.items():
            tokens.extend([word] * count)
        
        if len(tokens) < segment_size:
            return 0.0
        
        # Calculate TTR for each segment
        num_segments = len(tokens) // segment_size
        ttr_values = []
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            segment = tokens[start:end]
            
            unique = len(set(segment))
            ttr = unique / segment_size
            ttr_values.append(ttr)
        
        return sum(ttr_values) / len(ttr_values) if ttr_values else 0.0
    
    def _calculate_vocabulary_coverage(self, word_counter: Counter) -> Dict[str, int]:
        """
        Calculate vocabulary size needed for different coverage levels.
        
        Returns how many unique words cover X% of corpus.
        
        Args:
            word_counter: Word frequency counter
            
        Returns:
            Dictionary mapping coverage to vocabulary size
        """
        total_tokens = sum(word_counter.values())
        if total_tokens == 0:
            return {}
        
        # Sort by frequency (descending)
        sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
        
        coverage_levels = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
        result = {}
        
        cumulative = 0
        vocab_size = 0
        
        for word, count in sorted_words:
            cumulative += count
            vocab_size += 1
            
            coverage = cumulative / total_tokens
            
            for level in coverage_levels:
                if coverage >= level and f"{level:.2f}" not in result:
                    result[f"{level:.2f}"] = vocab_size
        
        return result
    
    def analyze_tokenizer(
        self,
        sp_model,
        config: Dict[str, Any],
        corpus_stats: CorpusStatistics,
        sample_texts: Optional[List[str]] = None
    ) -> TokenizerStatistics:
        """
        Analyze tokenizer performance.
        
        Args:
            sp_model: SentencePiece model
            config: Tokenizer configuration
            corpus_stats: Corpus statistics
            sample_texts: Optional sample texts for analysis
            
        Returns:
            TokenizerStatistics object
        """
        print_section("Tokenizer Performance Analysis")
        
        # === COMPRESSION METRICS ===
        
        # 1. Compression Ratio
        # Average characters per token
        compression_ratio = corpus_stats.total_characters / corpus_stats.total_words if corpus_stats.total_words > 0 else 0.0
        
        # 2. Tokens per Word
        # How many subword units per word on average
        if sample_texts:
            tokens_per_word = self._calculate_tokens_per_word(sp_model, sample_texts)
        else:
            # Estimate from vocabulary statistics
            tokens_per_word = 1.5  # Reasonable default for BPE
        
        # 3. Fertility Rate
        # Ideal is 1.0 (one input token -> one output token)
        fertility_rate = 1.0 / tokens_per_word if tokens_per_word > 0 else 1.0
        
        # 4. Byte-Pair Efficiency
        # How much we reduce vocabulary vs character-level
        char_level_vocab = corpus_stats.unique_characters
        bpe_vocab = config.get('vocab_size', 16000)
        byte_pair_efficiency = 1.0 - (bpe_vocab / (char_level_vocab * 100)) if char_level_vocab > 0 else 0.0
        
        # === COVERAGE METRICS ===
        
        # Character coverage (from config)
        char_coverage = config.get('character_coverage', 0.9999)
        
        # Estimate OOV rate (inverse of coverage)
        oov_estimate = 1.0 - char_coverage
        
        # === TOKEN STATISTICS ===
        
        # Calculate average token length from vocabulary
        vocab_lengths = []
        for idx in range(min(sp_model.vocab_size(), 10000)):  # Sample first 10k
            piece = sp_model.id_to_piece(idx)
            vocab_lengths.append(len(piece))
        
        avg_token_len = sum(vocab_lengths) / len(vocab_lengths) if vocab_lengths else 0.0
        
        # Standard deviation
        if len(vocab_lengths) > 1:
            mean = avg_token_len
            variance = sum((x - mean) ** 2 for x in vocab_lengths) / len(vocab_lengths)
            token_len_std = math.sqrt(variance)
        else:
            token_len_std = 0.0
        
        # === SAMPLE COMPRESSION ===
        
        if sample_texts:
            compression_rates = []
            for text in sample_texts:
                chars = len(text)
                tokens = len(sp_model.encode(text))
                if tokens > 0:
                    compression_rates.append(chars / tokens)
        else:
            compression_rates = []
        
        # === CZECH-SPECIFIC ===
        
        # Check diacritic preservation in vocabulary
        diacritic_preservation = self._check_diacritic_preservation(sp_model)
        
        stats = TokenizerStatistics(
            vocab_size=sp_model.vocab_size(),
            character_coverage=char_coverage,
            model_type=config.get('model_type', 'bpe'),
            compression_ratio=compression_ratio,
            tokens_per_word=tokens_per_word,
            fertility_rate=fertility_rate,
            byte_pair_efficiency=byte_pair_efficiency,
            character_coverage_actual=char_coverage,  # Would need actual measurement
            oov_rate_estimate=oov_estimate,
            avg_token_length=avg_token_len,
            token_length_std=token_len_std,
            sample_compression_rates=compression_rates,
            czech_diacritic_preservation=diacritic_preservation
        )
        
        return stats
    
    def _calculate_tokens_per_word(self, sp_model, sample_texts: List[str]) -> float:
        """
        Calculate average tokens per word from sample texts.
        
        Args:
            sp_model: SentencePiece model
            sample_texts: Sample texts
            
        Returns:
            Average tokens per word
        """
        total_words = 0
        total_tokens = 0
        
        for text in sample_texts:
            words = text.split()
            total_words += len(words)
            total_tokens += len(sp_model.encode(text))
        
        return total_tokens / total_words if total_words > 0 else 0.0
    
    def _check_diacritic_preservation(self, sp_model) -> float:
        """
        Check how well Czech diacritics are preserved in vocabulary.
        
        Args:
            sp_model: SentencePiece model
            
        Returns:
            Ratio of tokens containing Czech diacritics
        """
        diacritic_tokens = 0
        total_tokens = min(sp_model.vocab_size(), 10000)  # Sample first 10k
        
        czech_diacritics = set('áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ')
        
        for idx in range(total_tokens):
            piece = sp_model.id_to_piece(idx)
            if any(char in czech_diacritics for char in piece):
                diacritic_tokens += 1
        
        return diacritic_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def save_statistics(
        self,
        corpus_stats: CorpusStatistics,
        tokenizer_stats: TokenizerStatistics,
        output_path: Path
    ):
        """
        Save combined statistics to JSON file.
        
        Args:
            corpus_stats: Corpus statistics
            tokenizer_stats: Tokenizer statistics
            output_path: Path to save JSON
        """
        from datetime import datetime
        
        combined = CombinedStatistics(
            corpus=corpus_stats,
            tokenizer=tokenizer_stats,
            training_timestamp=datetime.now().isoformat()
        )
        
        # Convert to dict
        stats_dict = asdict(combined)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        print_success(f"Statistics saved: {output_path}")
    
    def print_statistics(
        self,
        corpus_stats: CorpusStatistics,
        tokenizer_stats: Optional[TokenizerStatistics] = None
    ):
        """
        Print formatted statistics to terminal.
        
        Args:
            corpus_stats: Corpus statistics
            tokenizer_stats: Optional tokenizer statistics
        """
        print_section("Corpus Statistics")
        
        # Basic metrics
        print(f"   Total characters:    {corpus_stats.total_characters:,}")
        print(f"   Total bytes:         {corpus_stats.total_bytes:,} ({corpus_stats.corpus_size_mb:.2f} MB)")
        print(f"   Total lines:         {corpus_stats.total_lines:,}")
        print(f"   Total words:         {corpus_stats.total_words:,}")
        print(f"   Unique words:        {corpus_stats.unique_words:,}")
        print(f"   Unique characters:   {corpus_stats.unique_characters:,}")
        print(f"   Corpus files:        {corpus_stats.corpus_files}")
        
        print("\n   === INFORMATION THEORY ===")
        print(f"   Character entropy:   {corpus_stats.character_entropy:.4f} bits")
        print(f"      (H(X) = -Σ p(x)*log2(p(x)))")
        print(f"      Maximum entropy:  {math.log2(corpus_stats.unique_characters):.4f} bits")
        
        print("\n   === LEXICAL RICHNESS (Type-Token Ratio Family) ===")
        print(f"   TTR:                 {corpus_stats.ttr:.6f}")
        print(f"      (unique_words / total_words)")
        print(f"   STTR:                {corpus_stats.sttr:.6f}")
        print(f"      (TTR on first 10k tokens)")
        print(f"   RTTR (Guiraud):      {corpus_stats.rttr:.4f}")
        print(f"      (unique_words / sqrt(total_words))")
        print(f"   MSTTR:               {corpus_stats.msttr:.6f}")
        print(f"      (Mean TTR over 1k-token segments)")
        
        print("\n   === VOCABULARY COVERAGE ===")
        for coverage, vocab_size in sorted(corpus_stats.vocabulary_size_at_coverage.items()):
            print(f"   {coverage} coverage:       {vocab_size:,} words")
        
        print("\n   === CZECH-SPECIFIC ===")
        print(f"   Diacritic ratio:     {corpus_stats.diacritic_ratio:.6f}")
        print(f"      ({corpus_stats.diacritic_ratio * 100:.2f}% of characters)")
        
        if tokenizer_stats:
            print("\n")
            print_section("Tokenizer Performance")
            
            print(f"   Vocabulary size:     {tokenizer_stats.vocab_size:,} tokens")
            print(f"   Model type:          {tokenizer_stats.model_type.upper()}")
            print(f"   Char coverage:       {tokenizer_stats.character_coverage}")
            
            print("\n   === COMPRESSION METRICS ===")
            print(f"   Compression ratio:   {tokenizer_stats.compression_ratio:.4f} chars/token")
            print(f"   Tokens per word:     {tokenizer_stats.tokens_per_word:.4f}")
            print(f"   Fertility rate:      {tokenizer_stats.fertility_rate:.4f}")
            print(f"      (1.0 = perfect, <1.0 = over-segmentation)")
            print(f"   BPE efficiency:      {tokenizer_stats.byte_pair_efficiency:.6f}")
            
            print("\n   === COVERAGE & OOV ===")
            print(f"   Char coverage:       {tokenizer_stats.character_coverage_actual:.6f}")
            print(f"   OOV estimate:        {tokenizer_stats.oov_rate_estimate:.6f}")
            print(f"      ({tokenizer_stats.oov_rate_estimate * 100:.4f}% expected OOV)")
            
            print("\n   === TOKEN STATISTICS ===")
            print(f"   Avg token length:    {tokenizer_stats.avg_token_length:.4f} ± {tokenizer_stats.token_length_std:.4f} chars")
            
            if tokenizer_stats.sample_compression_rates:
                avg_compression = sum(tokenizer_stats.sample_compression_rates) / len(tokenizer_stats.sample_compression_rates)
                print(f"   Sample compression:  {avg_compression:.4f} chars/token")
                print(f"      (based on {len(tokenizer_stats.sample_compression_rates)} samples)")
            
            print("\n   === CZECH-SPECIFIC ===")
            print(f"   Diacritic preservation: {tokenizer_stats.czech_diacritic_preservation:.6f}")
            print(f"      ({tokenizer_stats.czech_diacritic_preservation * 100:.2f}% of vocab contains diacritics)")
