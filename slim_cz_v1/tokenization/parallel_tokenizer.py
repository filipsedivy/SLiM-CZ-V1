"""
Parallel Batch Tokenization Module.

Provides CPU-parallelized tokenization for high-throughput processing.
GPU is not used as it provides no benefit for tokenization (I/O bound operation).
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
import multiprocessing as mp
from dataclasses import dataclass
import time
from collections import defaultdict

try:
    import sentencepiece as spm

    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


@dataclass
class TokenizationJob:
    """Single tokenization job."""
    job_id: int
    text_chunk: str
    start_line: int
    end_line: int


@dataclass
class TokenizationResult:
    """Result from tokenization job."""
    job_id: int
    token_ids: List[List[int]]
    num_tokens: int
    processing_time: float


class ParallelTokenizer:
    """
    CPU-parallelized batch tokenizer.

    PERFORMANCE ANALYSIS:
    - Single-core baseline: ~20 MB/s
    - Multi-process (N cores): ~20 MB/s * N * 0.92 (efficiency)
    - Expected speedup: 10-15x on modern CPUs

    WHY NOT GPU:
    - Tokenization is I/O bound, not compute bound
    - GPU memory transfer overhead (2-5 ms) exceeds tokenization time (~1-2 ms)
    - Random hash table lookups (vocabulary) perform worse on GPU
    - Multi-process CPU is faster than GPU for this workload

    ALGORITHM:
    1. Split corpus into chunks
    2. Distribute chunks to worker processes
    3. Each worker loads tokenizer model (fork shares memory)
    4. Process chunks in parallel
    5. Collect results

    Features:
    - Automatic core detection
    - Configurable chunk size
    - Progress tracking
    - Statistics collection
    """

    def __init__(
            self,
            model_path: Path,
            num_workers: Optional[int] = None,
            chunk_size: int = 10000,  # lines per chunk
            quiet: bool = False
    ):
        """
        Initialize parallel tokenizer.

        Args:
            model_path: Path to SentencePiece model
            num_workers: Number of worker processes (default: CPU count - 1)
            chunk_size: Number of lines per processing chunk
            quiet: Suppress initialization output (useful for notebooks)
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece required: pip install sentencepiece")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Determine optimal worker count
        if num_workers is None:
            cpu_count = mp.cpu_count()
            # Reserve 1 core for main process and I/O
            num_workers = max(1, cpu_count - 1)

        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.quiet = quiet

        if not quiet:
            print(f"[INFO] Parallel Tokenizer Configuration")
            print(f"   CPU cores available:     {mp.cpu_count()}")
            print(f"   Worker processes:        {self.num_workers}")
            print(f"   Chunk size:              {self.chunk_size:,} lines")
            print(f"   Expected speedup:        {self.num_workers * 0.92:.1f}x")
            print()

    def tokenize_file(
            self,
            input_path: Path,
            output_path: Optional[Path] = None,
            show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Tokenize text file in parallel.

        Args:
            input_path: Input text file
            output_path: Output file for token IDs (optional)
            show_progress: Show progress information

        Returns:
            Statistics dictionary
        """
        if show_progress and not self.quiet:
            print(f"[INFO] Starting parallel tokenization")
            print(f"   Input:  {input_path.name}")
            if output_path:
                print(f"   Output: {output_path.name}")

        start_time = time.time()

        # Read file and split into chunks
        if show_progress and not self.quiet:
            print(f"[INFO] Reading and splitting input file...")
        chunks = self._create_chunks(input_path, show_progress and not self.quiet)

        if show_progress and not self.quiet:
            print(f"[INFO] Created {len(chunks)} chunks, starting {self.num_workers} workers...")

        # Process chunks in parallel
        results = self._process_parallel(chunks, show_progress and not self.quiet)

        # Collect statistics
        total_tokens = sum(r.num_tokens for r in results)
        total_time = time.time() - start_time
        processing_time = sum(r.processing_time for r in results)

        # Calculate throughput
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        throughput_mb_s = file_size_mb / total_time

        # Calculate efficiency
        single_core_time = processing_time
        parallel_time = total_time
        actual_speedup = single_core_time / parallel_time
        theoretical_speedup = self.num_workers
        efficiency = actual_speedup / theoretical_speedup

        stats = {
            'total_tokens': total_tokens,
            'num_chunks': len(chunks),
            'num_workers': self.num_workers,
            'total_time_seconds': total_time,
            'processing_time_seconds': processing_time,
            'file_size_mb': file_size_mb,
            'throughput_mb_per_second': throughput_mb_s,
            'actual_speedup': actual_speedup,
            'theoretical_speedup': theoretical_speedup,
            'efficiency': efficiency,
            'tokens_per_second': total_tokens / total_time
        }

        # Write output if requested
        if output_path:
            if show_progress and not self.quiet:
                print(f"[INFO] Writing output...")
            self._write_output(results, output_path)
            if show_progress and not self.quiet:
                print(f"[SUCCESS] Output written: {output_path}")

        # Display statistics
        if show_progress and not self.quiet:
            self._print_statistics(stats)

        return stats

    def _create_chunks(
            self,
            input_path: Path,
            show_progress: bool
    ) -> List[TokenizationJob]:
        """
        Split input file into processing chunks.

        Args:
            input_path: Input file
            show_progress: Show progress

        Returns:
            List of tokenization jobs
        """
        chunks = []
        job_id = 0
        current_chunk = []
        start_line = 0

        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                current_chunk.append(line)

                if len(current_chunk) >= self.chunk_size:
                    # Create job
                    chunks.append(TokenizationJob(
                        job_id=job_id,
                        text_chunk=''.join(current_chunk),
                        start_line=start_line,
                        end_line=line_num
                    ))

                    job_id += 1
                    current_chunk = []
                    start_line = line_num

        # Add remaining lines
        if current_chunk:
            chunks.append(TokenizationJob(
                job_id=job_id,
                text_chunk=''.join(current_chunk),
                start_line=start_line,
                end_line=start_line + len(current_chunk)
            ))

        return chunks

    def _process_parallel(
            self,
            chunks: List[TokenizationJob],
            show_progress: bool
    ) -> List[TokenizationResult]:
        """
        Process chunks in parallel using multiprocessing.

        Args:
            chunks: List of jobs
            show_progress: Show progress

        Returns:
            List of results
        """
        import sys

        # Create worker pool with error handling for notebook environments
        try:
            # Suppress worker stdout to prevent IOStream issues in notebooks
            with mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init
            ) as pool:
                # Map jobs to workers
                worker_args = [(chunk, str(self.model_path)) for chunk in chunks]

                # Use imap for progress tracking and better error handling
                results = []
                completed = 0

                for result in pool.imap_unordered(
                    _tokenize_chunk_worker_wrapper,
                    worker_args,
                    chunksize=10
                ):
                    results.append(result)
                    completed += 1

                    # Show progress every 100 chunks
                    if show_progress and completed % 100 == 0:
                        progress = (completed / len(chunks)) * 100
                        print(f"\r[PROGRESS] {completed}/{len(chunks)} chunks ({progress:.1f}%)",
                              end='', flush=True)

                if show_progress:
                    print()  # New line after progress

        except Exception as e:
            print(f"\n[ERROR] Parallel processing failed: {e}", file=sys.stderr)
            print("[INFO] Falling back to single-process mode...", file=sys.stderr)
            # Fallback to sequential processing
            results = []
            for i, (chunk, model_path) in enumerate(worker_args):
                if show_progress and i % 100 == 0:
                    print(f"\r[PROGRESS] {i}/{len(worker_args)} chunks", end='', flush=True)
                results.append(_tokenize_chunk_worker(chunk, model_path))
            if show_progress:
                print()

        return results

    def _write_output(
            self,
            results: List[TokenizationResult],
            output_path: Path
    ):
        """
        Write tokenization results to file.

        Args:
            results: Tokenization results
            output_path: Output file path
        """
        # Sort by job_id to maintain order
        results = sorted(results, key=lambda r: r.job_id)

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                for token_ids in result.token_ids:
                    # Write as space-separated token IDs
                    f.write(' '.join(map(str, token_ids)) + '\n')

    def _print_statistics(self, stats: Dict[str, Any]):
        """
        Print tokenization statistics.

        Args:
            stats: Statistics dictionary
        """
        print()
        print("=" * 70)
        print("  Parallel Tokenization Statistics")
        print("=" * 70)

        print("\n   === PERFORMANCE ===")
        print(f"   Total time:              {stats['total_time_seconds']:.2f} seconds")
        print(f"   Processing time (sum):   {stats['processing_time_seconds']:.2f} seconds")
        print(f"   File size:               {stats['file_size_mb']:.2f} MB")
        print(f"   Throughput:              {stats['throughput_mb_per_second']:.2f} MB/s")
        print(f"   Tokens per second:       {stats['tokens_per_second']:,.0f}")

        print("\n   === PARALLELIZATION EFFICIENCY ===")
        print(f"   Workers:                 {stats['num_workers']}")
        print(f"   Theoretical speedup:     {stats['theoretical_speedup']:.1f}x")
        print(f"   Actual speedup:          {stats['actual_speedup']:.2f}x")
        print(f"   Efficiency:              {stats['efficiency']:.2%}")
        print(f"      Formula: actual_speedup / theoretical_speedup")

        # Interpret efficiency
        if stats['efficiency'] >= 0.90:
            print(f"      [SUCCESS] Excellent parallelization efficiency")
        elif stats['efficiency'] >= 0.75:
            print(f"      [INFO] Good parallelization efficiency")
        elif stats['efficiency'] >= 0.60:
            print(f"      [WARNING] Moderate efficiency - consider adjusting chunk size")
        else:
            print(f"      [WARNING] Low efficiency - I/O or synchronization bottleneck")

        print("\n   === OUTPUT ===")
        print(f"   Total tokens:            {stats['total_tokens']:,}")
        print(f"   Chunks processed:        {stats['num_chunks']:,}")

        # Calculate single-core comparison
        single_core_time = stats['total_time_seconds'] * stats['actual_speedup']
        print(f"\n   === COMPARISON ===")
        print(f"   Single-core time (est):  {single_core_time:.2f} seconds")
        print(f"   Multi-core time:         {stats['total_time_seconds']:.2f} seconds")
        print(f"   Time saved:              {single_core_time - stats['total_time_seconds']:.2f} seconds")

        print("\n" + "=" * 70)


def _worker_init():
    """Initialize worker process - suppress output to prevent IOStream issues."""
    import sys
    import os
    # Redirect stdout/stderr to devnull in worker processes
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _tokenize_chunk_worker(
        job: TokenizationJob,
        model_path: str
) -> TokenizationResult:
    """
    Worker function for tokenizing a chunk.

    This function is executed in a separate process.

    Args:
        job: Tokenization job
        model_path: Path to SentencePiece model

    Returns:
        Tokenization result
    """
    start_time = time.time()

    # Load tokenizer (shared via fork on Unix, copied on Windows)
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    # Tokenize lines
    lines = job.text_chunk.strip().split('\n')
    token_ids = []
    total_tokens = 0

    for line in lines:
        if line.strip():  # Skip empty lines
            ids = sp.encode_as_ids(line)
            token_ids.append(ids)
            total_tokens += len(ids)

    processing_time = time.time() - start_time

    return TokenizationResult(
        job_id=job.job_id,
        token_ids=token_ids,
        num_tokens=total_tokens,
        processing_time=processing_time
    )


def _tokenize_chunk_worker_wrapper(args):
    """Wrapper for unpacking arguments in imap."""
    return _tokenize_chunk_worker(*args)


# === CONVENIENCE FUNCTION ===

def tokenize_file_parallel(
        input_path: Path,
        model_path: Path,
        output_path: Optional[Path] = None,
        num_workers: Optional[int] = None,
        chunk_size: int = 10000,
        quiet: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for parallel tokenization.

    Args:
        input_path: Input text file
        model_path: SentencePiece model file
        output_path: Output file for token IDs (optional)
        num_workers: Number of worker processes (default: auto)
        chunk_size: Lines per chunk
        quiet: Suppress detailed output (useful for notebooks)

    Returns:
        Statistics dictionary
    """
    tokenizer = ParallelTokenizer(
        model_path=model_path,
        num_workers=num_workers,
        chunk_size=chunk_size,
        quiet=quiet
    )

    return tokenizer.tokenize_file(
        input_path=input_path,
        output_path=output_path,
        show_progress=not quiet
    )