"""
Parallel Batch Tokenization Module - Industrial Grade.

Designed for TB-scale corpus processing with:
- Streaming I/O (constant memory regardless of file size)
- Checkpointing (resume after failure)
- Real-time monitoring and statistics
- Configurable compression for intermediate files
- Batch writing with configurable buffer sizes

Architecture:
┌──────────────────────────────────────────────────────────────────────┐
│  COORDINATOR (main process)                                          │
│  - Scans input file for chunk boundaries (byte offsets only)        │
│  - Distributes work to worker pool                                   │
│  - Monitors progress and handles failures                            │
│  - Performs streaming merge of results                               │
├──────────────────────────────────────────────────────────────────────┤
│  WORKERS (N processes)                                               │
│  - Each loads own SentencePiece model instance                       │
│  - Reads assigned byte range from input file                         │
│  - Tokenizes and writes directly to temp file                        │
│  - Reports statistics back to coordinator                            │
├──────────────────────────────────────────────────────────────────────┤
│  DISK I/O                                                            │
│  - Input: Sequential read in chunks (OS page cache friendly)         │
│  - Temp: One file per chunk (parallel writes, no contention)         │
│  - Output: Sequential append (streaming merge)                       │
└──────────────────────────────────────────────────────────────────────┘

Memory Budget (per worker):
- SentencePiece model:     ~30-50 MB
- Input buffer:            ~chunk_size * avg_line_length
- Output buffer:           ~10 MB (configurable)
- Python overhead:         ~20 MB
- Total per worker:        ~100-150 MB typical

For 32 workers processing 1 TB file:
- Peak memory: ~5 GB (vs 1+ TB with naive approach)
- Disk I/O: ~2x file size (read once, write temp, write final)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
import tempfile
import shutil
import time
import json
import gzip
import os
import sys
import logging
import traceback
import signal
import hashlib

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

VERSION = "2.0.0"
DEFAULT_CHUNK_SIZE = 50_000          # Lines per chunk
DEFAULT_WRITE_BUFFER_SIZE = 10_000   # Lines before flush
CHECKPOINT_INTERVAL = 100            # Save checkpoint every N chunks


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChunkSpec:
    """
    Specification for a single processing chunk.

    Contains only metadata - no actual data loaded.
    Designed to be pickle-friendly for multiprocessing.
    """
    chunk_id: int
    start_byte: int
    end_byte: int
    estimated_lines: int

    @property
    def size_bytes(self) -> int:
        return self.end_byte - self.start_byte

    def __repr__(self) -> str:
        size_mb = self.size_bytes / (1024 * 1024)
        return f"Chunk({self.chunk_id}, {size_mb:.1f}MB, ~{self.estimated_lines} lines)"


@dataclass
class ChunkResult:
    """
    Result from processing a single chunk.

    Contains path to output file, not the actual tokens.
    """
    chunk_id: int
    output_path: str
    num_lines: int
    num_tokens: int
    processing_time_sec: float
    input_bytes: int
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        if self.processing_time_sec > 0:
            return self.num_tokens / self.processing_time_sec
        return 0.0

    @property
    def mb_per_second(self) -> float:
        if self.processing_time_sec > 0:
            return (self.input_bytes / (1024 * 1024)) / self.processing_time_sec
        return 0.0


@dataclass
class CheckpointData:
    """Checkpoint for resumable processing."""
    input_file: str
    input_file_hash: str
    output_file: str
    model_path: str
    chunk_size: int
    total_chunks: int
    completed_chunks: List[int]
    temp_dir: str
    started_at: str
    last_update: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, data: str) -> 'CheckpointData':
        return cls(**json.loads(data))


@dataclass
class ProcessingStats:
    """Aggregated processing statistics."""
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    total_lines: int = 0
    total_tokens: int = 0
    total_input_bytes: int = 0
    total_processing_time: float = 0.0
    wall_time: float = 0.0

    @property
    def progress_percent(self) -> float:
        if self.total_chunks > 0:
            return (self.completed_chunks / self.total_chunks) * 100
        return 0.0

    @property
    def throughput_mb_per_sec(self) -> float:
        if self.wall_time > 0:
            return (self.total_input_bytes / (1024 * 1024)) / self.wall_time
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.wall_time > 0:
            return self.total_tokens / self.wall_time
        return 0.0

    @property
    def avg_tokens_per_line(self) -> float:
        if self.total_lines > 0:
            return self.total_tokens / self.total_lines
        return 0.0

    @property
    def parallelization_efficiency(self) -> float:
        """Ratio of ideal speedup to actual speedup."""
        if self.wall_time > 0 and self.total_processing_time > 0:
            return self.total_processing_time / self.wall_time
        return 0.0


# ============================================================================
# LOGGING
# ============================================================================

class TokenizerLogger:
    """Structured logger for tokenization process."""

    def __init__(
        self,
        name: str = "parallel_tokenizer",
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        quiet: bool = False
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.quiet = quiet

        if not quiet:
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(level)
            console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self.logger.addHandler(console)

        if log_file:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s'
            ))
            self.logger.addHandler(file_handler)

    def debug(self, msg: str): self.logger.debug(msg)
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)

    def section(self, title: str):
        if not self.quiet:
            self.logger.info("=" * 60)
            self.logger.info(f"  {title}")
            self.logger.info("=" * 60)

    def progress(self, current: int, total: int, extra: str = ""):
        if not self.quiet:
            pct = (current / total) * 100 if total > 0 else 0
            bar_width = 30
            filled = int(bar_width * current / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            msg = f"\r   [{bar}] {pct:5.1f}% ({current}/{total}) {extra}"
            print(msg, end='', flush=True, file=sys.stderr)

    def progress_done(self):
        if not self.quiet:
            print(file=sys.stderr)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def compute_file_hash(filepath: Path, sample_size: int = 1024 * 1024) -> str:
    """
    Compute fast hash of file for checkpoint validation.
    Uses first and last sample_size bytes + file size for speed.
    """
    file_size = filepath.stat().st_size
    hasher = hashlib.md5()
    hasher.update(str(file_size).encode())

    with open(filepath, 'rb') as f:
        hasher.update(f.read(sample_size))
        if file_size > sample_size * 2:
            f.seek(-sample_size, 2)
            hasher.update(f.read(sample_size))

    return hasher.hexdigest()


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} EB"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_eta(completed: int, total: int, elapsed: float) -> str:
    """Estimate time remaining."""
    if completed == 0 or elapsed == 0:
        return "calculating..."

    rate = completed / elapsed
    remaining = total - completed
    eta_seconds = remaining / rate

    return format_duration(eta_seconds)


# ============================================================================
# CHUNK SCANNER
# ============================================================================

class ChunkScanner:
    """
    Scans input file to determine chunk boundaries.
    Memory efficient - only stores byte positions, not content.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        logger: Optional[TokenizerLogger] = None
    ):
        self.chunk_size = chunk_size
        self.logger = logger

    def scan(self, filepath: Path, show_progress: bool = True) -> List[ChunkSpec]:
        """Scan file and return chunk specifications."""
        file_size = filepath.stat().st_size
        chunks: List[ChunkSpec] = []

        chunk_id = 0
        chunk_start = 0
        lines_in_chunk = 0
        total_lines = 0
        bytes_read = 0

        last_progress = time.time()

        with open(filepath, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                lines_in_chunk += 1
                total_lines += 1
                bytes_read = f.tell()

                if lines_in_chunk >= self.chunk_size:
                    chunks.append(ChunkSpec(
                        chunk_id=chunk_id,
                        start_byte=chunk_start,
                        end_byte=bytes_read,
                        estimated_lines=lines_in_chunk
                    ))

                    chunk_id += 1
                    chunk_start = bytes_read
                    lines_in_chunk = 0

                if show_progress and self.logger:
                    now = time.time()
                    if now - last_progress >= 1.0:
                        pct = (bytes_read / file_size) * 100
                        self.logger.progress(
                            int(pct), 100,
                            f"| {total_lines:,} lines | {len(chunks)} chunks"
                        )
                        last_progress = now

        if lines_in_chunk > 0:
            chunks.append(ChunkSpec(
                chunk_id=chunk_id,
                start_byte=chunk_start,
                end_byte=file_size,
                estimated_lines=lines_in_chunk
            ))

        if show_progress and self.logger:
            self.logger.progress_done()

        if self.logger:
            self.logger.info(f"Scan complete: {total_lines:,} lines in {len(chunks)} chunks")

        return chunks


# ============================================================================
# WORKER FUNCTIONS
# ============================================================================

def _worker_process_chunk(args: Tuple) -> ChunkResult:
    """
    Worker function for processing a single chunk.

    This runs in a separate process. It:
    1. Loads the SentencePiece model
    2. Reads assigned byte range from input file
    3. Tokenizes each line
    4. Writes token IDs to temp file
    5. Returns statistics
    """
    (
        input_path,
        chunk_spec,
        model_path,
        temp_dir,
        write_buffer_size,
        compress_output
    ) = args

    start_time = time.time()

    try:
        # Load tokenizer model
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

        # Prepare output file
        output_filename = f"chunk_{chunk_spec.chunk_id:08d}.txt"
        if compress_output:
            output_filename += ".gz"
        output_path = os.path.join(temp_dir, output_filename)

        # Read chunk from input file
        with open(input_path, 'rb') as f:
            f.seek(chunk_spec.start_byte)
            chunk_bytes = f.read(chunk_spec.end_byte - chunk_spec.start_byte)

        # Decode bytes to text
        chunk_text = chunk_bytes.decode('utf-8', errors='replace')
        lines = chunk_text.split('\n')

        # Free memory
        del chunk_bytes
        del chunk_text

        # Process and write
        num_lines = 0
        num_tokens = 0
        write_buffer = []

        open_func = gzip.open if compress_output else open
        mode = 'wt' if compress_output else 'w'

        with open_func(output_path, mode, encoding='utf-8') as out_f:
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                token_ids = sp.EncodeAsIds(line)
                write_buffer.append(' '.join(map(str, token_ids)))
                num_lines += 1
                num_tokens += len(token_ids)

                if len(write_buffer) >= write_buffer_size:
                    out_f.write('\n'.join(write_buffer) + '\n')
                    write_buffer.clear()

            if write_buffer:
                out_f.write('\n'.join(write_buffer) + '\n')

        processing_time = time.time() - start_time

        return ChunkResult(
            chunk_id=chunk_spec.chunk_id,
            output_path=output_path,
            num_lines=num_lines,
            num_tokens=num_tokens,
            processing_time_sec=processing_time,
            input_bytes=chunk_spec.size_bytes
        )

    except Exception as e:
        return ChunkResult(
            chunk_id=chunk_spec.chunk_id,
            output_path="",
            num_lines=0,
            num_tokens=0,
            processing_time_sec=time.time() - start_time,
            input_bytes=chunk_spec.size_bytes,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )


# ============================================================================
# MAIN TOKENIZER CLASS
# ============================================================================

class ParallelTokenizer:
    """
    Industrial-grade parallel tokenizer for TB-scale corpus processing.

    Features:
    - Streaming I/O with constant memory usage
    - Checkpointing for crash recovery
    - Configurable parallelism
    - Real-time progress monitoring
    - Detailed statistics and logging

    Usage:
        tokenizer = ParallelTokenizer(
            model_path=Path("tokenizer.model"),
            num_workers=16,
            chunk_size=50000
        )

        stats = tokenizer.tokenize_file(
            input_path=Path("corpus.txt"),
            output_path=Path("tokens.txt")
        )
    """

    def __init__(
        self,
        model_path: Path,
        num_workers: Optional[int] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        write_buffer_size: int = DEFAULT_WRITE_BUFFER_SIZE,
        compress_temp: bool = False,
        quiet: bool = False,
        debug: bool = False,
        log_file: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize parallel tokenizer.

        Args:
            model_path: Path to SentencePiece .model file
            num_workers: Number of worker processes (default: CPU count - 1)
            chunk_size: Lines per processing chunk (default: 50000)
            write_buffer_size: Lines to buffer before disk write (default: 10000)
            compress_temp: Gzip compress temporary files (saves disk, costs CPU)
            quiet: Suppress all console output
            debug: Enable debug-level logging
            log_file: Path to write detailed log file
            checkpoint_dir: Directory for checkpoint files (enables resume)
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "SentencePiece is required. Install with: pip install sentencepiece"
            )

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Validate model can be loaded
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(str(self.model_path))
            self.vocab_size = sp.GetPieceSize()
        except Exception as e:
            raise ValueError(f"Failed to load SentencePiece model: {e}")

        # Worker configuration
        if num_workers is None:
            cpu_count = mp.cpu_count()
            self.num_workers = max(1, cpu_count - 1)
        else:
            self.num_workers = max(1, num_workers)

        self.chunk_size = chunk_size
        self.write_buffer_size = write_buffer_size
        self.compress_temp = compress_temp

        # Logging
        log_level = logging.DEBUG if debug else logging.INFO
        self.logger = TokenizerLogger(
            level=log_level,
            log_file=log_file,
            quiet=quiet
        )
        self.quiet = quiet
        self.debug = debug

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Runtime state
        self._temp_dir: Optional[Path] = None
        self._interrupted = False

        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log tokenizer configuration."""
        self.logger.section("Parallel Tokenizer Configuration")
        self.logger.info(f"Version:           {VERSION}")
        self.logger.info(f"Model:             {self.model_path.name}")
        self.logger.info(f"Vocabulary size:   {self.vocab_size:,}")
        self.logger.info(f"CPU cores:         {mp.cpu_count()}")
        self.logger.info(f"Worker processes:  {self.num_workers}")
        self.logger.info(f"Chunk size:        {self.chunk_size:,} lines")
        self.logger.info(f"Write buffer:      {self.write_buffer_size:,} lines")
        self.logger.info(f"Compress temp:     {self.compress_temp}")
        if self.checkpoint_dir:
            self.logger.info(f"Checkpoint dir:    {self.checkpoint_dir}")

    def tokenize_file(
        self,
        input_path: Path,
        output_path: Path,
        resume: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Tokenize a text file using parallel processing.

        Args:
            input_path: Input text file (one sentence per line)
            output_path: Output file for token IDs (space-separated)
            resume: Attempt to resume from checkpoint if available
            show_progress: Show progress updates

        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # File info
        file_size = input_path.stat().st_size
        file_hash = compute_file_hash(input_path)

        self.logger.section("Input File")
        self.logger.info(f"Path:    {input_path}")
        self.logger.info(f"Size:    {format_bytes(file_size)}")
        self.logger.info(f"Hash:    {file_hash[:16]}...")

        # Setup signal handlers
        self._setup_signal_handlers()

        # Initialize statistics
        stats = ProcessingStats()
        start_time = time.time()

        try:
            # Create temp directory
            self._temp_dir = Path(tempfile.mkdtemp(prefix='tokenizer_'))
            self.logger.debug(f"Temp directory: {self._temp_dir}")

            # PHASE 1: Scan input file
            self.logger.section("Phase 1: Scanning Input")
            scanner = ChunkScanner(
                chunk_size=self.chunk_size,
                logger=self.logger
            )
            chunks = scanner.scan(input_path, show_progress=show_progress)
            stats.total_chunks = len(chunks)

            # Check for checkpoint
            completed_chunk_ids = set()
            if resume and self.checkpoint_dir:
                completed_chunk_ids = self._load_checkpoint(
                    input_path, file_hash, chunks
                )
                if completed_chunk_ids:
                    self.logger.info(
                        f"Resuming: {len(completed_chunk_ids)} chunks already complete"
                    )

            # Filter chunks to process
            chunks_to_process = [
                c for c in chunks if c.chunk_id not in completed_chunk_ids
            ]

            if not chunks_to_process:
                self.logger.info("All chunks already processed!")
            else:
                # PHASE 2: Process chunks
                self.logger.section("Phase 2: Parallel Tokenization")
                self.logger.info(f"Processing {len(chunks_to_process)} chunks with {self.num_workers} workers")

                results = self._process_chunks(
                    input_path=input_path,
                    chunks=chunks_to_process,
                    show_progress=show_progress
                )

                # Update stats
                for result in results:
                    if result.error:
                        stats.failed_chunks += 1
                        self.logger.error(f"Chunk {result.chunk_id} failed: {result.error}")
                    else:
                        stats.completed_chunks += 1
                        stats.total_lines += result.num_lines
                        stats.total_tokens += result.num_tokens
                        stats.total_input_bytes += result.input_bytes
                        stats.total_processing_time += result.processing_time_sec

                # Add previously completed chunks to stats
                stats.completed_chunks += len(completed_chunk_ids)

            # PHASE 3: Merge results
            self.logger.section("Phase 3: Merging Results")
            merge_stats = self._merge_results(
                output_path=output_path,
                total_chunks=len(chunks),
                show_progress=show_progress
            )

            # Update final stats
            stats.total_lines = merge_stats['total_lines']
            stats.total_tokens = merge_stats['total_tokens']
            stats.wall_time = time.time() - start_time

            # PHASE 4: Report
            self.logger.section("Phase 4: Complete")
            self._report_stats(stats, output_path)

            return self._stats_to_dict(stats, output_path)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            if self.checkpoint_dir:
                self.logger.info("Progress saved to checkpoint")
            raise

        finally:
            # Cleanup
            if self._temp_dir and self._temp_dir.exists():
                try:
                    shutil.rmtree(self._temp_dir)
                    self.logger.debug(f"Cleaned up temp directory")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp dir: {e}")

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on interrupt."""
        def handler(signum, frame):
            self._interrupted = True
            self.logger.warning("Interrupt received, finishing current work...")

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _process_chunks(
        self,
        input_path: Path,
        chunks: List[ChunkSpec],
        show_progress: bool
    ) -> List[ChunkResult]:
        """Process chunks using multiprocessing pool."""
        # Prepare worker arguments
        worker_args = [
            (
                str(input_path),
                chunk,
                str(self.model_path),
                str(self._temp_dir),
                self.write_buffer_size,
                self.compress_temp
            )
            for chunk in chunks
        ]

        results: List[ChunkResult] = []
        completed = 0
        last_checkpoint = 0
        last_progress_time = time.time()
        start_time = time.time()

        # Calculate optimal chunksize for imap
        pool_chunksize = max(1, len(chunks) // (self.num_workers * 4))

        try:
            ctx = mp.get_context('spawn')

            with ctx.Pool(processes=self.num_workers) as pool:
                for result in pool.imap_unordered(
                    _worker_process_chunk,
                    worker_args,
                    chunksize=pool_chunksize
                ):
                    results.append(result)
                    completed += 1

                    if self._interrupted:
                        self.logger.warning("Stopping due to interrupt...")
                        pool.terminate()
                        break

                    # Progress update
                    if show_progress:
                        now = time.time()
                        if now - last_progress_time >= 0.5:
                            elapsed = now - start_time
                            eta = estimate_eta(completed, len(chunks), elapsed)

                            total_bytes = sum(r.input_bytes for r in results if not r.error)
                            throughput = total_bytes / (1024 * 1024) / elapsed if elapsed > 0 else 0

                            self.logger.progress(
                                completed, len(chunks),
                                f"| {throughput:.1f} MB/s | ETA: {eta}"
                            )
                            last_progress_time = now

                    # Checkpoint
                    if self.checkpoint_dir:
                        if completed - last_checkpoint >= CHECKPOINT_INTERVAL:
                            self._save_checkpoint(
                                input_path,
                                [r.chunk_id for r in results if not r.error]
                            )
                            last_checkpoint = completed

            if show_progress:
                self.logger.progress_done()

        except Exception as e:
            self.logger.error(f"Pool processing failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise

        return results

    def _merge_results(
        self,
        output_path: Path,
        total_chunks: int,
        show_progress: bool
    ) -> Dict[str, int]:
        """Merge all chunk results into final output file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = 0
        total_tokens = 0
        merged = 0

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for chunk_id in range(total_chunks):
                filename = f"chunk_{chunk_id:08d}.txt"
                chunk_path = self._temp_dir / filename

                if not chunk_path.exists():
                    chunk_path = self._temp_dir / (filename + ".gz")

                if not chunk_path.exists():
                    self.logger.warning(f"Missing chunk {chunk_id}")
                    continue

                open_func = gzip.open if chunk_path.suffix == '.gz' else open
                mode = 'rt' if chunk_path.suffix == '.gz' else 'r'

                with open_func(chunk_path, mode, encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1
                        total_tokens += len(line.strip().split())

                chunk_path.unlink()
                merged += 1

                if show_progress and merged % 50 == 0:
                    self.logger.progress(merged, total_chunks, "")

        if show_progress:
            self.logger.progress_done()

        self.logger.info(f"Merged {merged} chunks → {output_path}")

        return {
            'total_lines': total_lines,
            'total_tokens': total_tokens
        }

    def _load_checkpoint(
        self,
        input_path: Path,
        file_hash: str,
        chunks: List[ChunkSpec]
    ) -> set:
        """Load checkpoint and return completed chunk IDs."""
        checkpoint_file = self.checkpoint_dir / "checkpoint.json"

        if not checkpoint_file.exists():
            return set()

        try:
            with open(checkpoint_file, 'r') as f:
                cp = CheckpointData.from_json(f.read())

            if cp.input_file_hash != file_hash:
                self.logger.warning("Checkpoint hash mismatch, starting fresh")
                return set()

            if cp.chunk_size != self.chunk_size:
                self.logger.warning("Checkpoint chunk size mismatch, starting fresh")
                return set()

            return set(cp.completed_chunks)

        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return set()

    def _save_checkpoint(self, input_path: Path, completed_ids: List[int]):
        """Save checkpoint to disk."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.checkpoint_dir / "checkpoint.json"

        cp = CheckpointData(
            input_file=str(input_path),
            input_file_hash=compute_file_hash(input_path),
            output_file="",
            model_path=str(self.model_path),
            chunk_size=self.chunk_size,
            total_chunks=0,
            completed_chunks=completed_ids,
            temp_dir=str(self._temp_dir),
            started_at=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )

        with open(checkpoint_file, 'w') as f:
            f.write(cp.to_json())

    def _report_stats(self, stats: ProcessingStats, output_path: Path):
        """Print final statistics."""
        self.logger.info(f"")
        self.logger.info(f"Output file:       {output_path}")
        self.logger.info(f"Total lines:       {stats.total_lines:,}")
        self.logger.info(f"Total tokens:      {stats.total_tokens:,}")
        self.logger.info(f"Avg tokens/line:   {stats.avg_tokens_per_line:.2f}")
        self.logger.info(f"")
        self.logger.info(f"Wall time:         {format_duration(stats.wall_time)}")
        self.logger.info(f"Throughput:        {stats.throughput_mb_per_sec:.2f} MB/s")
        self.logger.info(f"Tokens/second:     {stats.tokens_per_second:,.0f}")
        self.logger.info(f"")
        self.logger.info(f"Chunks processed:  {stats.completed_chunks}/{stats.total_chunks}")
        if stats.failed_chunks > 0:
            self.logger.warning(f"Chunks failed:     {stats.failed_chunks}")
        self.logger.info(f"Parallelization:   {stats.parallelization_efficiency:.1f}x effective")

    def _stats_to_dict(self, stats: ProcessingStats, output_path: Path) -> Dict[str, Any]:
        """Convert stats to dictionary for return."""
        return {
            'output_path': str(output_path),
            'total_lines': stats.total_lines,
            'total_tokens': stats.total_tokens,
            'avg_tokens_per_line': stats.avg_tokens_per_line,
            'total_chunks': stats.total_chunks,
            'completed_chunks': stats.completed_chunks,
            'failed_chunks': stats.failed_chunks,
            'wall_time_seconds': stats.wall_time,
            'throughput_mb_per_second': stats.throughput_mb_per_sec,
            'tokens_per_second': stats.tokens_per_second,
            'parallelization_efficiency': stats.parallelization_efficiency,
            'num_workers': self.num_workers,
            'chunk_size': self.chunk_size
        }


# ============================================================================
# SHARD PROCESSOR
# ============================================================================

class ShardProcessor:
    """
    Process multiple shard files in parallel.

    This is more efficient than single-file processing for pre-sharded data:
    - No scanning phase needed (each shard = one unit of work)
    - Natural checkpoint granularity
    - Can output per-shard or merged
    - Trivially parallelizable across machines

    Usage:
        processor = ShardProcessor(
            model_path=Path("tokenizer.model"),
            num_workers=16
        )

        # Process directory of shards
        stats = processor.process_shards(
            input_pattern="./shards/shard-*.txt",
            output_dir=Path("./tokens/")
        )

        # Or merge into single output
        stats = processor.process_shards(
            input_pattern="./shards/shard-*.txt",
            output_path=Path("./tokens.txt"),
            merge_output=True
        )
    """

    def __init__(
        self,
        model_path: Path,
        num_workers: Optional[int] = None,
        quiet: bool = False,
        debug: bool = False,
        log_file: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize shard processor.

        Args:
            model_path: Path to SentencePiece .model file
            num_workers: Number of worker processes (default: CPU count - 1)
            quiet: Suppress console output
            debug: Enable debug logging
            log_file: Path to log file
            checkpoint_dir: Directory for checkpoints (enables resume)
        """
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece required: pip install sentencepiece")

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Validate model
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(str(self.model_path))
            self.vocab_size = sp.GetPieceSize()
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

        # Workers
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = max(1, num_workers)

        # Logging
        log_level = logging.DEBUG if debug else logging.INFO
        self.logger = TokenizerLogger(
            level=log_level,
            log_file=log_file,
            quiet=quiet
        )
        self.quiet = quiet
        self.debug = debug

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self._interrupted = False

    def discover_shards(self, input_pattern: str) -> List[Path]:
        """
        Discover shard files matching pattern.

        Args:
            input_pattern: Glob pattern (e.g., "./shards/shard-*.txt")
                          or directory path

        Returns:
            Sorted list of shard file paths
        """
        import glob

        pattern_path = Path(input_pattern)

        # If it's a directory, find all .txt files
        if pattern_path.is_dir():
            files = list(pattern_path.glob("*.txt"))
        else:
            # Use glob pattern
            files = [Path(f) for f in glob.glob(input_pattern)]

        # Sort naturally (shard-1, shard-2, ..., shard-10, not shard-1, shard-10, shard-2)
        def natural_sort_key(path: Path) -> List:
            import re
            parts = re.split(r'(\d+)', path.name)
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        files = sorted(files, key=natural_sort_key)

        return files

    def process_shards(
        self,
        input_pattern: str,
        output_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        merge_output: bool = False,
        show_progress: bool = True,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process shard files in parallel.

        Args:
            input_pattern: Glob pattern for shard files (e.g., "shards/*.txt")
            output_dir: Directory for per-shard output files
            output_path: Single output file (requires merge_output=True)
            merge_output: Merge all shards into single output file
            show_progress: Show progress updates
            resume: Resume from checkpoint if available

        Returns:
            Statistics dictionary
        """
        # Validate arguments
        if merge_output and not output_path:
            raise ValueError("merge_output=True requires output_path")
        if not merge_output and not output_dir:
            raise ValueError("Need either output_dir or merge_output=True with output_path")

        # Discover shards
        shards = self.discover_shards(input_pattern)

        if not shards:
            raise FileNotFoundError(f"No shards found matching: {input_pattern}")

        # Calculate total size
        total_size = sum(s.stat().st_size for s in shards)

        self.logger.section("Shard Processing")
        self.logger.info(f"Pattern:       {input_pattern}")
        self.logger.info(f"Shards found:  {len(shards)}")
        self.logger.info(f"Total size:    {format_bytes(total_size)}")
        self.logger.info(f"Workers:       {self.num_workers}")

        if merge_output:
            self.logger.info(f"Output:        {output_path} (merged)")
        else:
            self.logger.info(f"Output dir:    {output_dir}")

        # Setup
        start_time = time.time()
        temp_dir = Path(tempfile.mkdtemp(prefix='tokenizer_shards_'))

        # Check checkpoint
        completed_shards = set()
        if resume and self.checkpoint_dir:
            completed_shards = self._load_shard_checkpoint(shards)
            if completed_shards:
                self.logger.info(f"Resuming: {len(completed_shards)}/{len(shards)} shards complete")

        # Filter shards to process
        shards_to_process = [s for s in shards if s.name not in completed_shards]

        try:
            # Process shards
            if shards_to_process:
                self.logger.section("Processing Shards")

                results = self._process_shards_parallel(
                    shards=shards_to_process,
                    temp_dir=temp_dir,
                    output_dir=output_dir if not merge_output else None,
                    show_progress=show_progress
                )
            else:
                results = []
                self.logger.info("All shards already processed")

            # Merge if requested
            if merge_output:
                self.logger.section("Merging Output")
                merge_stats = self._merge_shard_outputs(
                    shards=shards,
                    temp_dir=temp_dir,
                    output_dir=output_dir,
                    output_path=output_path,
                    show_progress=show_progress
                )
            else:
                merge_stats = {'total_lines': 0, 'total_tokens': 0}
                for r in results:
                    if not r.error:
                        merge_stats['total_lines'] += r.num_lines
                        merge_stats['total_tokens'] += r.num_tokens

            # Calculate stats
            wall_time = time.time() - start_time

            stats = {
                'total_shards': len(shards),
                'processed_shards': len(shards_to_process),
                'failed_shards': sum(1 for r in results if r.error),
                'total_lines': merge_stats['total_lines'],
                'total_tokens': merge_stats['total_tokens'],
                'total_bytes': total_size,
                'wall_time_seconds': wall_time,
                'throughput_mb_per_second': (total_size / (1024*1024)) / wall_time if wall_time > 0 else 0,
                'tokens_per_second': merge_stats['total_tokens'] / wall_time if wall_time > 0 else 0
            }

            # Report
            self.logger.section("Complete")
            self.logger.info(f"Shards:      {stats['total_shards']} ({stats['failed_shards']} failed)")
            self.logger.info(f"Lines:       {stats['total_lines']:,}")
            self.logger.info(f"Tokens:      {stats['total_tokens']:,}")
            self.logger.info(f"Time:        {format_duration(wall_time)}")
            self.logger.info(f"Throughput:  {stats['throughput_mb_per_second']:.2f} MB/s")

            return stats

        finally:
            # Cleanup temp
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _process_shards_parallel(
        self,
        shards: List[Path],
        temp_dir: Path,
        output_dir: Optional[Path],
        show_progress: bool
    ) -> List[ChunkResult]:
        """Process shards using worker pool."""

        # Prepare arguments
        worker_args = []
        for i, shard in enumerate(shards):
            if output_dir:
                out_path = output_dir / f"{shard.stem}.tokens.txt"
            else:
                out_path = temp_dir / f"{shard.stem}.tokens.txt"

            worker_args.append((
                str(shard),
                str(out_path),
                str(self.model_path),
                10000  # write buffer
            ))

        results = []
        completed = 0
        start_time = time.time()
        last_progress = time.time()

        # Create output dir if needed
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            ctx = mp.get_context('spawn')

            with ctx.Pool(processes=self.num_workers) as pool:
                for result in pool.imap_unordered(
                    _process_shard_worker,
                    worker_args,
                    chunksize=1
                ):
                    results.append(result)
                    completed += 1

                    if self._interrupted:
                        pool.terminate()
                        break

                    # Save checkpoint
                    if self.checkpoint_dir and not result.error:
                        self._save_shard_checkpoint(
                            [r for r in results if not r.error]
                        )

                    # Progress
                    if show_progress:
                        now = time.time()
                        if now - last_progress >= 0.5:
                            elapsed = now - start_time
                            eta = estimate_eta(completed, len(shards), elapsed)

                            total_bytes = sum(r.input_bytes for r in results if not r.error)
                            throughput = (total_bytes / (1024*1024)) / elapsed if elapsed > 0 else 0

                            self.logger.progress(
                                completed, len(shards),
                                f"| {throughput:.1f} MB/s | ETA: {eta}"
                            )
                            last_progress = now

            if show_progress:
                self.logger.progress_done()

        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise

        return results

    def _merge_shard_outputs(
        self,
        shards: List[Path],
        temp_dir: Path,
        output_dir: Optional[Path],
        output_path: Path,
        show_progress: bool
    ) -> Dict[str, int]:
        """Merge all shard outputs into single file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = 0
        total_tokens = 0

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for i, shard in enumerate(shards):
                # Find the output file
                if output_dir:
                    shard_output = output_dir / f"{shard.stem}.tokens.txt"
                else:
                    shard_output = temp_dir / f"{shard.stem}.tokens.txt"

                if not shard_output.exists():
                    self.logger.warning(f"Missing output for {shard.name}")
                    continue

                # Stream copy
                with open(shard_output, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1
                        total_tokens += len(line.strip().split())

                if show_progress and (i + 1) % 10 == 0:
                    self.logger.progress(i + 1, len(shards), "")

        if show_progress:
            self.logger.progress_done()

        self.logger.info(f"Merged {len(shards)} shards → {output_path}")

        return {
            'total_lines': total_lines,
            'total_tokens': total_tokens
        }

    def _load_shard_checkpoint(self, shards: List[Path]) -> set:
        """Load checkpoint for shard processing."""
        if not self.checkpoint_dir:
            return set()

        cp_file = self.checkpoint_dir / "shard_checkpoint.json"
        if not cp_file.exists():
            return set()

        try:
            with open(cp_file, 'r') as f:
                data = json.load(f)
            return set(data.get('completed_shards', []))
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return set()

    def _save_shard_checkpoint(self, results: List[ChunkResult]):
        """Save checkpoint for shard processing."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cp_file = self.checkpoint_dir / "shard_checkpoint.json"

        # Extract shard names from output paths
        completed = []
        for r in results:
            if r.output_path:
                name = Path(r.output_path).stem.replace('.tokens', '')
                completed.append(name)

        data = {
            'completed_shards': completed,
            'last_update': datetime.now().isoformat()
        }

        with open(cp_file, 'w') as f:
            json.dump(data, f, indent=2)


def _process_shard_worker(args: Tuple) -> ChunkResult:
    """Worker function for processing a single shard file."""
    input_path, output_path, model_path, write_buffer_size = args

    start_time = time.time()
    input_path = Path(input_path)

    try:
        # Load model
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

        # Process file
        num_lines = 0
        num_tokens = 0
        write_buffer = []

        with open(output_path, 'w', encoding='utf-8') as out_f:
            with open(input_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue

                    token_ids = sp.EncodeAsIds(line)
                    write_buffer.append(' '.join(map(str, token_ids)))
                    num_lines += 1
                    num_tokens += len(token_ids)

                    if len(write_buffer) >= write_buffer_size:
                        out_f.write('\n'.join(write_buffer) + '\n')
                        write_buffer.clear()

            if write_buffer:
                out_f.write('\n'.join(write_buffer) + '\n')

        return ChunkResult(
            chunk_id=0,
            output_path=output_path,
            num_lines=num_lines,
            num_tokens=num_tokens,
            processing_time_sec=time.time() - start_time,
            input_bytes=input_path.stat().st_size
        )

    except Exception as e:
        return ChunkResult(
            chunk_id=0,
            output_path="",
            num_lines=0,
            num_tokens=0,
            processing_time_sec=time.time() - start_time,
            input_bytes=input_path.stat().st_size if input_path.exists() else 0,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def tokenize_shards_parallel(
    input_pattern: str,
    model_path: Path,
    output_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    merge_output: bool = False,
    num_workers: Optional[int] = None,
    quiet: bool = False,
    debug: bool = False,
    log_file: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function for parallel shard tokenization.

    Args:
        input_pattern: Glob pattern for shards (e.g., "shards/*.txt")
        model_path: SentencePiece model file
        output_dir: Directory for per-shard outputs
        output_path: Single merged output file
        merge_output: Merge all outputs into one file
        num_workers: Worker processes (default: auto)
        quiet: Suppress output
        debug: Enable debug logging
        log_file: Log file path
        checkpoint_dir: Checkpoint directory

    Returns:
        Statistics dictionary
    """
    processor = ShardProcessor(
        model_path=model_path,
        num_workers=num_workers,
        quiet=quiet,
        debug=debug,
        log_file=log_file,
        checkpoint_dir=checkpoint_dir
    )

    return processor.process_shards(
        input_pattern=input_pattern,
        output_dir=output_dir,
        output_path=output_path,
        merge_output=merge_output,
        show_progress=not quiet
    )


def tokenize_file_parallel(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    num_workers: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    quiet: bool = False,
    debug: bool = False,
    log_file: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function for parallel tokenization.

    Args:
        input_path: Input text file
        model_path: SentencePiece model file
        output_path: Output file for token IDs
        num_workers: Number of worker processes (default: auto)
        chunk_size: Lines per chunk (default: 50000)
        quiet: Suppress output
        debug: Enable debug logging
        log_file: Path to debug log file
        checkpoint_dir: Directory for checkpoints (enables resume)

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
        checkpoint_dir=checkpoint_dir
    )

    return tokenizer.tokenize_file(
        input_path=input_path,
        output_path=output_path,
        show_progress=not quiet
    )