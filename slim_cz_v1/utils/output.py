"""
Universal output module for SLiM-CZ-V1.

Provides centralized console output with:
- Verbose/quiet mode switching
- Automatic environment detection (Kaggle, Jupyter, Terminal)
- Automatic I/O flushing for problematic environments
- Consistent formatting across CLI and programmatic use

Usage:
    from slim_cz_v1.utils import console

    console.info("Loading data...")
    console.success("Model trained!")
    console.verbose("Detailed info...")  # Only shown in verbose mode

    # Enable verbose mode:
    console.set_verbose(True)
    # Or via environment variable:
    # export SLIM_VERBOSE=1

Configuration:
    Environment variables:
        SLIM_VERBOSE=1      Enable verbose output
        SLIM_QUIET=1        Suppress all non-error output
        SLIM_NO_COLOR=1     Disable colored output
        SLIM_FLUSH=1        Force flush after every print
"""

import os
import sys
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager


class Console:
    """
    Centralized console output handler.

    Singleton-like design - import and use directly:
        from slim_cz_v1.utils import console
        console.info("message")
    """

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[0;33m',
        'blue': '\033[0;34m',
        'cyan': '\033[0;36m',
        'gray': '\033[0;90m',
    }

    def __init__(self):
        """Initialize console with environment detection."""
        self._verbose = False
        self._quiet = False
        self._use_color = True
        self._force_flush = False
        self._last_flush = time.time()
        self._flush_interval = 30  # seconds

        # Auto-configure from environment
        self._detect_environment()
        self._load_env_config()

    def _detect_environment(self):
        """Detect execution environment and set appropriate defaults."""
        self.in_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        self.in_colab = 'COLAB_GPU' in os.environ
        self.in_notebook = False

        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                self.in_notebook = True
        except ImportError:
            pass

        # Auto-enable flush in problematic environments
        if self.in_kaggle or self.in_colab or self.in_notebook:
            self._force_flush = True

    def _load_env_config(self):
        """Load configuration from environment variables."""
        if os.environ.get('SLIM_VERBOSE', '').lower() in ('1', 'true', 'yes'):
            self._verbose = True

        if os.environ.get('SLIM_QUIET', '').lower() in ('1', 'true', 'yes'):
            self._quiet = True

        if os.environ.get('SLIM_NO_COLOR', '').lower() in ('1', 'true', 'yes'):
            self._use_color = False

        if os.environ.get('SLIM_FLUSH', '').lower() in ('1', 'true', 'yes'):
            self._force_flush = True

    # ================================================================
    # CONFIGURATION
    # ================================================================

    def set_verbose(self, enabled: bool = True):
        """Enable or disable verbose output."""
        self._verbose = enabled
        return self

    def set_quiet(self, enabled: bool = True):
        """Enable or disable quiet mode (suppresses info/success)."""
        self._quiet = enabled
        return self

    def set_color(self, enabled: bool = True):
        """Enable or disable colored output."""
        self._use_color = enabled
        return self

    def set_flush(self, enabled: bool = True, interval: int = 30):
        """Enable or disable force flush mode."""
        self._force_flush = enabled
        self._flush_interval = interval
        return self

    def configure(
            self,
            verbose: Optional[bool] = None,
            quiet: Optional[bool] = None,
            color: Optional[bool] = None,
            flush: Optional[bool] = None,
            flush_interval: Optional[int] = None
    ):
        """
        Configure console settings.

        Args:
            verbose: Enable verbose output
            quiet: Enable quiet mode
            color: Enable colored output
            flush: Enable force flush
            flush_interval: Flush interval in seconds
        """
        if verbose is not None:
            self._verbose = verbose
        if quiet is not None:
            self._quiet = quiet
        if color is not None:
            self._use_color = color
        if flush is not None:
            self._force_flush = flush
        if flush_interval is not None:
            self._flush_interval = flush_interval
        return self

    @property
    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self._verbose

    @property
    def is_quiet(self) -> bool:
        """Check if quiet mode is enabled."""
        return self._quiet

    # ================================================================
    # OUTPUT METHODS
    # ================================================================

    def _flush(self):
        """Flush stdout and stderr."""
        sys.stdout.flush()
        sys.stderr.flush()
        self._last_flush = time.time()

    def _should_flush(self) -> bool:
        """Check if we should flush now."""
        if self._force_flush:
            return True
        return time.time() - self._last_flush > self._flush_interval

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self._use_color and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def _print(self, message: str, color: Optional[str] = None,
               force: bool = False, file=None):
        """Internal print with optional color and flush."""
        if file is None:
            file = sys.stdout

        if color:
            message = self._colorize(message, color)

        print(message, file=file)

        if force or self._should_flush():
            self._flush()

    # ----------------------------------------------------------------
    # Standard output methods
    # ----------------------------------------------------------------

    def print(self, *args, **kwargs):
        """Print with automatic flush handling."""
        print(*args, **kwargs)
        if self._should_flush():
            self._flush()

    def info(self, message: str):
        """Print info message (suppressed in quiet mode)."""
        if not self._quiet:
            self._print(f"[INFO]    {message}", 'blue')

    def success(self, message: str):
        """Print success message (suppressed in quiet mode)."""
        if not self._quiet:
            self._print(f"[SUCCESS] ✔ {message}", 'green')

    def warning(self, message: str):
        """Print warning message (always shown)."""
        self._print(f"[WARNING] ⚠ {message}", 'yellow')

    def error(self, message: str):
        """Print error message (always shown)."""
        self._print(f"[ERROR]   ✖ {message}", 'red', file=sys.stderr)

    def verbose(self, message: str):
        """Print verbose message (only in verbose mode)."""
        if self._verbose:
            self._print(f"[VERBOSE] {message}", 'gray')

    def debug(self, message: str):
        """Alias for verbose."""
        self.verbose(message)

    # ----------------------------------------------------------------
    # Structured output
    # ----------------------------------------------------------------

    def header(self, title: str, width: int = 70):
        """Print section header."""
        if not self._quiet:
            self._print("")
            self._print(self._colorize("=" * width, 'bold'))
            self._print(self._colorize(title, 'bold'))
            self._print(self._colorize("=" * width, 'bold'))
            self._print("")

    def section(self, title: str, width: int = 70):
        """Print subsection header."""
        if not self._quiet:
            self._print("")
            self._print(self._colorize(title, 'cyan'))
            self._print("-" * width)

    def kv(self, key: str, value: Any, indent: int = 2):
        """Print key-value pair."""
        if not self._quiet:
            spaces = " " * indent
            if isinstance(value, float):
                self._print(f"{spaces}{key}: {value:.6f}")
            elif isinstance(value, int) and value > 1000:
                self._print(f"{spaces}{key}: {value:,}")
            else:
                self._print(f"{spaces}{key}: {value}")

    def table(self, data: Dict[str, Any], indent: int = 2, key_width: int = 30):
        """Print dictionary as aligned table."""
        if not self._quiet:
            spaces = " " * indent
            for key, value in data.items():
                if isinstance(value, float):
                    formatted = f"{value:.6f}"
                elif isinstance(value, int) and value > 1000:
                    formatted = f"{value:,}"
                else:
                    formatted = str(value)
                self._print(f"{spaces}{key:<{key_width}} {formatted}")

    # ----------------------------------------------------------------
    # Progress reporting
    # ----------------------------------------------------------------

    def progress(self, current: int, total: int, prefix: str = "",
                 suffix: str = "", width: int = 30):
        """Print progress bar (verbose mode or forced)."""
        if self._quiet:
            return

        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)

        line = f"\r{prefix} [{bar}] {percent * 100:.1f}% {suffix}"

        if self._verbose or current == total:
            print(line, end="" if current < total else "\n")
            if self._should_flush():
                self._flush()

    def epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """Print epoch summary."""
        if self._quiet:
            return

        parts = [f"[EPOCH {epoch}/{total_epochs}]"]
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")

        self._print(" | ".join(parts), force=True)

    def batch(self, batch: int, total: int, loss: float,
              extra: Optional[Dict[str, float]] = None):
        """Print batch progress (verbose mode only)."""
        if not self._verbose:
            # Still flush periodically even without output
            if self._should_flush():
                self._flush()
            return

        percent = (batch + 1) / total * 100
        msg = f"[BATCH] {batch + 1}/{total} ({percent:.1f}%) | loss: {loss:.4f}"

        if extra:
            for key, value in extra.items():
                if isinstance(value, float):
                    msg += f" | {key}: {value:.4f}"
                else:
                    msg += f" | {key}: {value}"

        self._print(msg, 'gray')

    # ----------------------------------------------------------------
    # Context managers
    # ----------------------------------------------------------------

    @contextmanager
    def status(self, message: str):
        """
        Context manager for operations with status.

        Usage:
            with console.status("Training model..."):
                train()
        """
        self.info(f"{message}...")
        start = time.time()
        try:
            yield
            elapsed = time.time() - start
            self.success(f"{message} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            self.error(f"{message} failed ({elapsed:.1f}s): {e}")
            raise

    @contextmanager
    def verbose_mode(self):
        """Temporarily enable verbose mode."""
        old_verbose = self._verbose
        self._verbose = True
        try:
            yield
        finally:
            self._verbose = old_verbose

    @contextmanager
    def quiet_mode(self):
        """Temporarily enable quiet mode."""
        old_quiet = self._quiet
        self._quiet = True
        try:
            yield
        finally:
            self._quiet = old_quiet

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    def flush(self):
        """Force flush output."""
        self._flush()

    def heartbeat(self, message: str = ""):
        """
        Send heartbeat to prevent I/O timeout.
        Prints only if flush interval elapsed.
        """
        if time.time() - self._last_flush > self._flush_interval:
            if message and self._verbose:
                self._print(f"[HEARTBEAT] {message}", 'gray', force=True)
            else:
                self._flush()

    def newline(self):
        """Print empty line."""
        if not self._quiet:
            print()

    def rule(self, char: str = "-", width: int = 70):
        """Print horizontal rule."""
        if not self._quiet:
            self._print(char * width)

    def env_info(self):
        """Print detected environment info (verbose mode)."""
        if not self._verbose:
            return

        env = []
        if self.in_kaggle:
            env.append("Kaggle")
        if self.in_colab:
            env.append("Colab")
        if self.in_notebook:
            env.append("Notebook")
        if not env:
            env.append("Terminal")

        self.verbose(f"Environment: {', '.join(env)}")
        self.verbose(f"Force flush: {self._force_flush}")
        self.verbose(f"Colors: {self._use_color}")


# ================================================================
# GLOBAL INSTANCE
# ================================================================

# Create global console instance
console = Console()