"""
Parallel Batch Tokenization Module - STREAMING VERSION.

Memory-efficient CPU-parallelized tokenization for large files.
Uses streaming I/O and disk-based intermediate storage.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
import multiprocessing as mp
from dataclasses import dataclass
import time
import tempfile
import shutil
import os
import sys
import logging
import traceback

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def setup_logger(debug: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logger with optional file output."""
    logger = logging.getLogger('parallel_tokenizer')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_format = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ChunkInfo:
    """Lightweight chunk reference (no data in memory)."""
    chunk_id: int
    start_byte: int
    end_byte: int
    line_count: int


@dataclass
class TokenizationResult:
    """Result from tokenization job."""
    chunk_id: int
    output_file: str  # Path to temp file with results
    num_tokens: int
    num_lines: int
    processing_time: float
    error: Optional[str] = None


# ============================================================
# STREAMING PARALLEL TOKENIZER
# ============================================================

class ParallelTokenizer:
    """
    Memory-efficient parallel tokenizer using streaming I/O.

    KEY IMPROVEMENTS OVER PREVIOUS VERSION:
    1. Never loads entire file into memory
    2. Each worker writes results to temp file immediately
    3. Final merge is streaming (read chunk, write, delete)
    4. Proper error propagation from workers
    5. Comprehensive debug logging

    MEMORY USAGE:
    - Main process: ~100 MB overhead + chunk_size lines
    - Each worker: ~50 MB (tokenizer model) + chunk_size lines
    - Peak: num_workers * (50 MB + chunk_size * avg_line_size)

    For 4 GB file with 16 workers and 10k chunk_size:
    - Previous version: 4+ GB in memory (all chunks)
    - This version: ~1 GB peak (workers + current chunks)
    """

    def __init__(
        self,
        model_path: Path,
        num_workers: Optional[int] = None,
        chunk_size: int = 10000,
        quiet: bool = False,
        debug: bool = False,
        log_file: Optional[Path] = None,
        sequential: bool = False
    ):
        """
        Initialize parallel tokenizer.

        Args:
            model_path: Path to SentencePiece model
            num_workers: Number of worker processes (default: CPU count - 1)
            chunk_size: Number of lines per processing chunk
            quiet: Suppress all output
            debug: Enable verbose debug logging
            log_file: Path to debug log file (enables file logging)
            sequential: Force sequential processing (no multiprocessing)
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece required: pip install sentencepiece")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Setup logging
        self.logger = setup_logger(debug=debug, log_file=log_file)
        self.debug = debug
        self.quiet = quiet
        self.sequential = sequential

        # Determine worker count
        if sequential:
            self.num_workers = 1
        elif num_workers is None:
            cpu_count = mp.cpu_count()
            self.num_workers = max(1, cpu_count - 1)
        else:
            self.num_workers = num_workers

        self.chunk_size = chunk_size

        # Temp directory for intermediate files
        self.temp_dir: Optional[Path] = None

        if not quiet:
            self._print_config()

    def _print_config(self):
        """Print configuration summary."""
        print("=" * 70)
        print("  Parallel Tokenizer Configuration (Streaming Mode)")
        print("=" * 70)
        print(f"   Model:              {self.model_path.name}")
        print(f"   CPU cores:          {mp.cpu_count()}")
        print(f"   Worker processes:   {self.num_workers}")
        print(f"   Chunk size:         {self.chunk_size:,} lines")
        print(f"   Processing mode:    {'SEQUENTIAL' if self.sequential else 'PARALLEL'}")
        print(f"   Debug logging:      {'ENABLED' if self.debug else 'disabled'}")
        print("=" * 70)
        print()

    def tokenize_file(
        self,
        input_path: Path,
        output_path: Path,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Tokenize text file with streaming I/O.

        Args:
            input_path: Input text file
            output_path: Output file for token IDs
            show_progress: Show progress information

        Returns:
            Statistics dictionary
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        file_size = input_path.stat().st_size
        file_size_mb = file_size / (1024 ** 2)
        file_size_gb = file_size / (1024 ** 3)

        self.logger.info(f"Input file: {input_path}")
        self.logger.info(f"File size: {file_size_gb:.2f} GB ({file_size:,} bytes)")
        self.logger.info(f"Output: {output_path}")

        start_time = time.time()

        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix='tokenizer_'))
        self.logger.debug(f"Temp directory: {self.temp_dir}")

        try:
            # PHASE 1: Count lines and create chunk boundaries
            if show_progress and not self.quiet:
                print("[PHASE 1/4] Scanning input file...")

            chunks = self._scan_file(input_path, show_progress)
            self.logger.info(f"Created {len(chunks)} chunks")

            # PHASE 2: Process chunks
            if show_progress and not self.quiet:
                print(f"[PHASE 2/4] Processing {len(chunks)} chunks with {self.num_workers} workers...")

            if self.sequential:
                results = self._process_sequential(input_path, chunks, show_progress)
            else:
                results = self._process_parallel(input_path, chunks, show_progress)

            # Check for errors
            errors = [r for r in results if r.error]
            if errors:
                self.logger.error(f"Errors in {len(errors)} chunks!")
                for err in errors[:5]:  # Show first 5
                    self.logger.error(f"  Chunk {err.chunk_id}: {err.error}")
                if len(errors) > 5:
                    self.logger.error(f"  ... and {len(errors) - 5} more errors")

            # PHASE 3: Merge results
            if show_progress and not self.quiet:
                print("[PHASE 3/4] Merging results...")

            total_tokens, total_lines = self._merge_results(results, output_path, show_progress)

            # PHASE 4: Calculate statistics
            total_time = time.time() - start_time
            processing_time = sum(r.processing_time for r in results if not r.error)

            stats = self._calculate_stats(
                total_tokens=total_tokens,
                total_lines=total_lines,
                num_chunks=len(chunks),
                total_time=total_time,
                processing_time=processing_time,
                file_size_mb=file_size_mb,
                num_errors=len(errors)
            )

            if show_progress and not self.quiet:
                print("[PHASE 4/4] Complete!")
                self._print_statistics(stats)

            return stats

        finally:
            # Cleanup temp directory
            if self.temp_dir and self.temp_dir.exists():
                self.logger.debug(f"Cleaning up temp directory: {self.temp_dir}")
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp dir: {e}")

    def _scan_file(
        self,
        input_path: Path,
        show_progress: bool
    ) -> List[ChunkInfo]:
        """
        Scan file to determine chunk boundaries without loading into memory.

        Returns list of ChunkInfo with byte positions.
        """
        chunks = []
        chunk_id = 0
        line_count = 0
        chunk_start_byte = 0
        chunk_line_count = 0
        total_lines = 0

        self.logger.debug("Scanning file for chunk boundaries...")

        with open(input_path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                line_count += 1
                chunk_line_count += 1
                total_lines += 1

                if chunk_line_count >= self.chunk_size:
                    current_byte = f.tell()
                    chunks.append(ChunkInfo(
                        chunk_id=chunk_id,
                        start_byte=chunk_start_byte,
                        end_byte=current_byte,
                        line_count=chunk_line_count
                    ))

                    if self.debug and chunk_id % 100 == 0:
                        self.logger.debug(f"Chunk {chunk_id}: bytes {chunk_start_byte}-{current_byte}, {chunk_line_count} lines")

                    chunk_id += 1
                    chunk_start_byte = current_byte
                    chunk_line_count = 0

                # Progress update
                if show_progress and not self.quiet and total_lines % 1_000_000 == 0:
                    print(f"\r   Scanned {total_lines:,} lines, {chunk_id} chunks...", end='', flush=True)

        # Add remaining lines
        if chunk_line_count > 0:
            current_byte = input_path.stat().st_size
            chunks.append(ChunkInfo(
                chunk_id=chunk_id,
                start_byte=chunk_start_byte,
                end_byte=current_byte,
                line_count=chunk_line_count
            ))

        if show_progress and not self.quiet:
            print(f"\r   Scanned {total_lines:,} lines, {len(chunks)} chunks    ")

        self.logger.info(f"Total lines: {total_lines:,}")
        self.logger.info(f"Total chunks: {len(chunks)}")

        return chunks

    def _process_sequential(
        self,
        input_path: Path,
        chunks: List[ChunkInfo],
        show_progress: bool
    ) -> List[TokenizationResult]:
        """Process chunks sequentially (single process)."""
        self.logger.info("Processing in SEQUENTIAL mode")

        results = []

        for i, chunk in enumerate(chunks):
            if show_progress and not self.quiet:
                progress = ((i + 1) / len(chunks)) * 100
                print(f"\r   Processing chunk {i+1}/{len(chunks)} ({progress:.1f}%)...", end='', flush=True)

            try:
                result = _process_single_chunk(
                    input_path=str(input_path),
                    chunk=chunk,
                    model_path=str(self.model_path),
                    temp_dir=str(self.temp_dir),
                    debug=self.debug
                )
                results.append(result)

                if self.debug:
                    self.logger.debug(f"Chunk {chunk.chunk_id}: {result.num_tokens} tokens, {result.processing_time:.2f}s")

            except Exception as e:
                self.logger.error(f"Chunk {chunk.chunk_id} failed: {e}")
                self.logger.debug(traceback.format_exc())
                results.append(TokenizationResult(
                    chunk_id=chunk.chunk_id,
                    output_file="",
                    num_tokens=0,
                    num_lines=0,
                    processing_time=0,
                    error=str(e)
                ))

        if show_progress and not self.quiet:
            print()

        return results

    def _process_parallel(
        self,
        input_path: Path,
        chunks: List[ChunkInfo],
        show_progress: bool
    ) -> List[TokenizationResult]:
        """Process chunks in parallel with proper error handling."""
        self.logger.info(f"Processing in PARALLEL mode with {self.num_workers} workers")

        # Prepare worker arguments
        worker_args = [
            (str(input_path), chunk, str(self.model_path), str(self.temp_dir), self.debug)
            for chunk in chunks
        ]

        results = []
        completed = 0

        try:
            # Use spawn instead of fork for better compatibility
            ctx = mp.get_context('spawn')

            with ctx.Pool(processes=self.num_workers) as pool:
                # Use imap_unordered for better memory efficiency
                for result in pool.imap_unordered(
                    _process_chunk_wrapper,
                    worker_args,
                    chunksize=max(1, len(chunks) // (self.num_workers * 4))
                ):
                    results.append(result)
                    completed += 1

                    if result.error:
                        self.logger.error(f"Worker error in chunk {result.chunk_id}: {result.error}")
                    elif self.debug and completed % 50 == 0:
                        self.logger.debug(f"Completed {completed}/{len(chunks)} chunks")

                    if show_progress and not self.quiet and completed % 10 == 0:
                        progress = (completed / len(chunks)) * 100
                        print(f"\r   Processing: {completed}/{len(chunks)} chunks ({progress:.1f}%)...", end='', flush=True)

            if show_progress and not self.quiet:
                print(f"\r   Processing: {completed}/{len(chunks)} chunks (100.0%)    ")

        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            self.logger.error(traceback.format_exc())

            if show_progress and not self.quiet:
                print(f"\n[WARNING] Parallel processing failed, falling back to sequential...")

            # Fallback to sequential
            return self._process_sequential(input_path, chunks, show_progress)

        return results

    def _merge_results(
        self,
        results: List[TokenizationResult],
        output_path: Path,
        show_progress: bool
    ) -> Tuple[int, int]:
        """
        Merge all result files into final output.

        Uses streaming to avoid loading all results into memory.
        """
        # Sort by chunk_id to maintain order
        results = sorted(results, key=lambda r: r.chunk_id)

        total_tokens = 0
        total_lines = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for i, result in enumerate(results):
                if result.error or not result.output_file:
                    self.logger.warning(f"Skipping chunk {result.chunk_id} (error)")
                    continue

                result_path = Path(result.output_file)
                if not result_path.exists():
                    self.logger.warning(f"Result file missing: {result_path}")
                    continue

                # Stream copy from temp file to output
                with open(result_path, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1

                total_tokens += result.num_tokens

                # Delete temp file after copying
                try:
                    result_path.unlink()
                except:
                    pass

                if show_progress and not self.quiet and (i + 1) % 100 == 0:
                    progress = ((i + 1) / len(results)) * 100
                    print(f"\r   Merging: {i+1}/{len(results)} chunks ({progress:.1f}%)...", end='', flush=True)

        if show_progress and not self.quiet:
            print(f"\r   Merging: {len(results)}/{len(results)} chunks (100.0%)    ")

        self.logger.info(f"Merged {total_lines:,} lines, {total_tokens:,} tokens")

        return total_tokens, total_lines

    def _calculate_stats(
        self,
        total_tokens: int,
        total_lines: int,
        num_chunks: int,
        total_time: float,
        processing_time: float,
        file_size_mb: float,
        num_errors: int
    ) -> Dict[str, Any]:
        """Calculate performance statistics."""
        throughput_mb_s = file_size_mb / total_time if total_time > 0 else 0

        # Efficiency calculation
        if self.sequential or self.num_workers <= 1:
            actual_speedup = 1.0
            efficiency = 1.0
        else:
            # processing_time is sum of all worker times
            # In perfect parallelization: processing_time = total_time * num_workers
            theoretical_parallel_time = processing_time / self.num_workers
            actual_speedup = theoretical_parallel_time / total_time if total_time > 0 else 1.0
            efficiency = min(1.0, actual_speedup)  # Cap at 100%

        return {
            'total_tokens': total_tokens,
            'total_lines': total_lines,
            'num_chunks': num_chunks,
            'num_workers': self.num_workers,
            'num_errors': num_errors,
            'total_time_seconds': total_time,
            'processing_time_seconds': processing_time,
            'file_size_mb': file_size_mb,
            'throughput_mb_per_second': throughput_mb_s,
            'actual_speedup': actual_speedup,
            'theoretical_speedup': self.num_workers,
            'efficiency': efficiency,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'sequential_mode': self.sequential
        }

    def _print_statistics(self, stats: Dict[str, Any]):
        """Print tokenization statistics."""
        print()
        print("=" * 70)
        print("  Tokenization Statistics")
        print("=" * 70)

        print("\n   === PERFORMANCE ===")
        print(f"   Total time:              {stats['total_time_seconds']:.2f} seconds")
        print(f"   File size:               {stats['file_size_mb']:.2f} MB")
        print(f"   Throughput:              {stats['throughput_mb_per_second']:.2f} MB/s")
        print(f"   Tokens per second:       {stats['tokens_per_second']:,.0f}")

        if not stats['sequential_mode']:
            print("\n   === PARALLELIZATION ===")
            print(f"   Workers:                 {stats['num_workers']}")
            print(f"   Efficiency:              {stats['efficiency']:.1%}")
        else:
            print("\n   === MODE ===")
            print(f"   Processing:              SEQUENTIAL (single process)")

        if stats['num_errors'] > 0:
            print(f"\n   === ERRORS ===")
            print(f"   Failed chunks:           {stats['num_errors']}")

        print("\n   === OUTPUT ===")
        print(f"   Total tokens:            {stats['total_tokens']:,}")
        print(f"   Total lines:             {stats['total_lines']:,}")
        print(f"   Chunks processed:        {stats['num_chunks']:,}")

        print("\n" + "=" * 70)


# ============================================================
# WORKER FUNCTIONS (must be at module level for pickling)
# ============================================================

def _process_single_chunk(
    input_path: str,
    chunk: ChunkInfo,
    model_path: str,
    temp_dir: str,
    debug: bool = False
) -> TokenizationResult:
    """
    Process a single chunk - reads bytes from file, tokenizes, writes to temp file.

    This function is designed to be memory-efficient:
    - Only loads one chunk at a time
    - Writes results immediately to disk
    - Does not keep token IDs in memory
    """
    start_time = time.time()

    try:
        # Load tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        # Read chunk from file using byte positions
        with open(input_path, 'rb') as f:
            f.seek(chunk.start_byte)
            chunk_bytes = f.read(chunk.end_byte - chunk.start_byte)

        # Decode to text
        chunk_text = chunk_bytes.decode('utf-8', errors='replace')
        lines = chunk_text.strip().split('\n')

        # Create output file
        output_file = Path(temp_dir) / f"chunk_{chunk.chunk_id:06d}.txt"

        total_tokens = 0
        num_lines = 0

        # Tokenize and write immediately
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line in lines:
                line = line.strip()
                if line:
                    token_ids = sp.encode_as_ids(line)
                    out_f.write(' '.join(map(str, token_ids)) + '\n')
                    total_tokens += len(token_ids)
                    num_lines += 1

        processing_time = time.time() - start_time

        return TokenizationResult(
            chunk_id=chunk.chunk_id,
            output_file=str(output_file),
            num_tokens=total_tokens,
            num_lines=num_lines,
            processing_time=processing_time
        )

    except Exception as e:
        return TokenizationResult(
            chunk_id=chunk.chunk_id,
            output_file="",
            num_tokens=0,
            num_lines=0,
            processing_time=time.time() - start_time,
            error=f"{type(e).__name__}: {str(e)}"
        )


def _process_chunk_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments."""
    input_path, chunk, model_path, temp_dir, debug = args
    return _process_single_chunk(input_path, chunk, model_path, temp_dir, debug)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def tokenize_file_parallel(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    num_workers: Optional[int] = None,
    chunk_size: int = 10000,
    quiet: bool = False,
    debug: bool = False,
    log_file: Optional[Path] = None,
    sequential: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for parallel tokenization.

    Args:
        input_path: Input text file
        model_path: SentencePiece model file
        output_path: Output file for token IDs
        num_workers: Number of worker processes (default: auto)
        chunk_size: Lines per chunk (default: 10000)
        quiet: Suppress output
        debug: Enable debug logging
        log_file: Path to debug log file
        sequential: Force sequential processing

    Returns:
        Statistics dictionary
    """
    tokenizer = ParallelTokenizer(
        model_path=model_path,
        num_workers=num_workers,
        chunk_size=chunk_size,
        quiet=quiet,
        debug=debug,
        log_file=log_file,
        sequential=sequential
    )

    return tokenizer.tokenize_file(
        input_path=input_path,
        output_path=output_path,
        show_progress=not quiet
    )