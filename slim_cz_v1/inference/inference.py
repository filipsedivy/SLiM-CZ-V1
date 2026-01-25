"""
SLiM-CZ-V1 inference module.

Provides model loading, text generation, and interactive mode functionality.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

try:
    import sentencepiece as spm

    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False

from ..model import SLiM_CZ_V1


# ============================================================
# SPINNER
# ============================================================

class Spinner:
    """Simple spinner for loading operations."""

    def __init__(self, message: str = "Processing"):
        self.message = message
        self.frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.current = 0
        self.active = False

    def start(self):
        """Start spinner."""
        self.active = True
        sys.stdout.write(f"\r{self.message} {self.frames[0]}")
        sys.stdout.flush()

    def update(self):
        """Update spinner frame."""
        if not self.active:
            return
        self.current = (self.current + 1) % len(self.frames)
        sys.stdout.write(f"\r{self.message} {self.frames[self.current]}")
        sys.stdout.flush()

    def stop(self, final_message: str = None):
        """Stop spinner."""
        self.active = False
        if final_message:
            sys.stdout.write(f"\r{final_message}\n")
        else:
            sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()


# ============================================================
# MODEL LOADER
# ============================================================

class ModelLoader:
    """Load trained SLiM-CZ-V1 model and tokenizer."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = None):
        """
        Initialize model loader.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            tokenizer_path: Path to tokenizer model (.model file)
            device: Device to use (cuda/cpu, auto-detect if None)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.tokenizer = None
        self.config = None
        self.checkpoint_info = None

    def load(self, cli_formatter=None):
        """
        Load model and tokenizer.

        Args:
            cli_formatter: Optional CLIFormatter instance for output
        """
        # Use provided formatter or create dummy functions
        if cli_formatter:
            fmt = cli_formatter
        else:
            # Dummy formatter for non-CLI usage
            class DummyFormatter:
                @staticmethod
                def section(msg): pass

                @staticmethod
                def error(msg): print(f"ERROR: {msg}")

                @staticmethod
                def info(msg): pass

                @staticmethod
                def success(msg): pass

                @staticmethod
                def warning(msg): pass

                GREEN = ''
                RESET = ''

            fmt = DummyFormatter()

        fmt.section("Loading Model & Tokenizer")

        # Check files exist
        if not self.checkpoint_path.exists():
            fmt.error(f"Checkpoint not found: {self.checkpoint_path}")
            sys.exit(1)

        if not self.tokenizer_path.exists():
            fmt.error(f"Tokenizer not found: {self.tokenizer_path}")
            sys.exit(1)

        # Load tokenizer
        spinner = Spinner("Loading tokenizer")
        spinner.start()

        if not HAS_SENTENCEPIECE:
            spinner.stop()
            fmt.error("SentencePiece not installed!")
            fmt.info("Install with: pip install sentencepiece")
            sys.exit(1)

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(self.tokenizer_path))
        vocab_size = self.tokenizer.get_piece_size()
        spinner.stop(f"{fmt.GREEN}✔ Tokenizer loaded: {vocab_size:,} tokens{fmt.RESET}")

        # Load checkpoint
        spinner = Spinner("Loading model checkpoint")
        spinner.start()

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'val_perplexity': checkpoint.get('best_val_perplexity', 'unknown')
        }

        spinner.stop(f"{fmt.GREEN}✔ Checkpoint loaded{fmt.RESET}")

        # Detect max_seq_len from checkpoint
        max_seq_len_detected = None
        if 'model_state_dict' in checkpoint:
            if 'pos_encoding.pe' in checkpoint['model_state_dict']:
                pe_shape = checkpoint['model_state_dict']['pos_encoding.pe'].shape
                max_seq_len_detected = pe_shape[0]

        # Create model
        spinner = Spinner("Initializing model")
        spinner.start()

        model_config = self.config.get('model', {})
        max_seq_len = max_seq_len_detected or model_config.get('max_seq_len', 512)

        self.model = SLiM_CZ_V1(
            vocab_size=vocab_size,
            d_model=model_config.get('d_model', 256),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 4),
            d_ff=model_config.get('d_ff', 1024),
            max_seq_len=max_seq_len,
            dropout=model_config.get('dropout', 0.1),
            weight_tying=model_config.get('weight_tying', True)
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        spinner.stop(f"{fmt.GREEN}✔ Model initialized{fmt.RESET}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for display.

        Returns:
            Dictionary with model architecture and checkpoint info
        """
        params = self.model.count_parameters()
        model_cfg = self.config.get('model', {})

        return {
            'architecture': {
                'name': 'SLiM-CZ-V1',
                'description': 'Slavic Linguistic integrated Micro-model for Czechia',
                'num_layers': model_cfg.get('num_layers', 'N/A'),
                'num_heads': model_cfg.get('num_heads', 'N/A'),
                'd_model': model_cfg.get('d_model', 'N/A'),
                'd_ff': model_cfg.get('d_ff', 'N/A'),
                'max_seq_len': self.model.max_seq_len,
                'vocab_size': self.tokenizer.get_piece_size(),
            },
            'parameters': {
                'total': params['total'],
                'total_millions': params['total'] / 1e6,
                'memory_fp32_mb': (params['total'] * 4) / (1024 ** 2),
                'memory_fp16_mb': (params['total'] * 2) / (1024 ** 2),
            },
            'checkpoint': {
                'epoch': self.checkpoint_info['epoch'],
                'val_loss': self.checkpoint_info['val_loss'],
                'val_perplexity': self.checkpoint_info['val_perplexity'],
            },
            'device': str(self.device)
        }


# ============================================================
# TEXT GENERATOR
# ============================================================

class TextGenerator:
    """Generate text using SLiM-CZ-V1 model."""

    def __init__(self, model: SLiM_CZ_V1, tokenizer, device: torch.device):
        """
        Initialize text generator.

        Args:
            model: Loaded SLiM-CZ-V1 model
            tokenizer: SentencePiece tokenizer
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 100,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.2,
            do_sample: bool = True,
            show_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            do_sample: Use sampling (False = greedy)
            show_prompt: Include prompt in output

        Returns:
            Dictionary with 'text' and 'statistics' keys
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Track token history for repetition penalty
        generated_tokens = []

        start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get logits from model
                context = input_tensor[:, -self.model.max_seq_len:]
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated_tokens:
                    for token_id in set(generated_tokens):
                        logits[0, token_id] /= repetition_penalty

                if do_sample:
                    # Top-k filtering
                    if top_k > 0:
                        top_k_actual = min(top_k, logits.size(-1))
                        indices_to_remove = logits < torch.topk(logits, top_k_actual)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float('-inf')

                    # Sample
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to sequence
                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                generated_tokens.append(next_token.item())

                # Stop at EOS token
                if next_token.item() == self.tokenizer.eos_id():
                    break

        elapsed_time = time.time() - start_time

        # Decode
        generated_ids = input_tensor[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        if not show_prompt:
            generated_only_ids = generated_ids[len(input_ids):]
            generated_text = self.tokenizer.decode(generated_only_ids)

        # Calculate statistics
        actual_tokens = len(generated_tokens)
        tokens_per_sec = actual_tokens / elapsed_time if elapsed_time > 0 else 0

        return {
            'text': generated_text,
            'statistics': {
                'prompt_tokens': len(input_ids),
                'generated_tokens': actual_tokens,
                'total_tokens': len(generated_ids),
                'elapsed_time': elapsed_time,
                'tokens_per_second': tokens_per_sec,
                'stopped_early': actual_tokens < max_new_tokens
            }
        }

    def batch_generate(
            self,
            prompts: List[str],
            max_new_tokens: int = 100,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.2,
            do_sample: bool = True,
            progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            progress_callback: Optional callback(current, total, stats) for progress updates
            Other args: Same as generate()

        Returns:
            List of result dictionaries
        """
        results = []

        for i, prompt in enumerate(prompts):
            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                show_prompt=False
            )

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(prompts), result['statistics'])

        return results


# ============================================================
# INTERACTIVE MODE
# ============================================================

class InteractiveMode:
    """Interactive text generation interface."""

    def __init__(
            self,
            generator: TextGenerator,
            config: Dict[str, Any],
            cli_formatter=None
    ):
        """
        Initialize interactive mode.

        Args:
            generator: TextGenerator instance
            config: Generation config
            cli_formatter: Optional CLIFormatter for output
        """
        self.generator = generator
        self.gen_config = config.get('generation', {})
        self.cli_formatter = cli_formatter

        # Default parameters
        self.max_new_tokens = self.gen_config.get('max_new_tokens', 100)
        self.temperature = self.gen_config.get('temperature', 0.8)
        self.top_k = self.gen_config.get('top_k', 50)
        self.top_p = self.gen_config.get('top_p', 0.95)
        self.repetition_penalty = self.gen_config.get('repetition_penalty', 1.2)
        self.do_sample = self.gen_config.get('do_sample', True)

    def run(self):
        """Run interactive generation loop."""
        fmt = self.cli_formatter

        if fmt:
            fmt.header("Interactive Generation Mode")
            fmt.info("SLiM-CZ-V1 - Slavic Linguistic integrated Micro-model for Czechia")
            print()

        self._print_help()

        while True:
            try:
                print()
                if fmt:
                    prompt = input(f"{fmt.CYAN}> {fmt.RESET}").strip()
                else:
                    prompt = input("> ").strip()

                if not prompt:
                    continue

                # Handle commands
                if prompt.startswith('/'):
                    self._handle_command(prompt)
                    continue

                # Generate
                print()
                if fmt:
                    fmt.info("Generating...")
                print("-" * 70)

                result = self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=self.do_sample,
                    show_prompt=True
                )

                print(result['text'])
                print("-" * 70)

                # Display statistics
                stats = result['statistics']
                if fmt:
                    print(f"{fmt.GREEN}[COMPLETE]{fmt.RESET} " +
                          f"{stats['generated_tokens']} tokens in {stats['elapsed_time']:.2f}s " +
                          f"({stats['tokens_per_second']:.1f} tokens/sec)")
                else:
                    print(f"[COMPLETE] {stats['generated_tokens']} tokens in " +
                          f"{stats['elapsed_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)")

            except KeyboardInterrupt:
                print()
                print()
                if fmt:
                    fmt.info("Exiting interactive mode...")
                else:
                    print("Exiting...")
                break
            except Exception as e:
                print()
                if fmt:
                    fmt.error(f"Generation failed: {e}")
                else:
                    print(f"ERROR: {e}")

    def _print_help(self):
        """Print help message."""
        print("Commands:")
        print("  /help              - Show this help message")
        print("  /params            - Show current generation parameters")
        print("  /set <param> <val> - Set generation parameter")
        print("  /quit, /exit       - Exit interactive mode")
        print()
        print("Parameters:")
        print("  max_new_tokens      - Maximum tokens to generate")
        print("  temperature         - Sampling temperature (0.0-2.0)")
        print("  top_k               - Top-k sampling (0 = disabled)")
        print("  top_p               - Nucleus sampling (0.0-1.0)")
        print("  repetition_penalty  - Penalty for repeated tokens")
        print("  do_sample           - Enable sampling (0 or 1)")

    def _handle_command(self, cmd: str):
        """Handle interactive commands."""
        fmt = self.cli_formatter
        parts = cmd.split()
        cmd = parts[0].lower()

        if cmd == '/help':
            print()
            self._print_help()

        elif cmd == '/params':
            print()
            if fmt:
                fmt.section("Current Generation Parameters")
                fmt.metric("max_new_tokens:", str(self.max_new_tokens))
                fmt.metric("temperature:", f"{self.temperature:.2f}")
                fmt.metric("top_k:", str(self.top_k))
                fmt.metric("top_p:", f"{self.top_p:.2f}")
                fmt.metric("repetition_penalty:", f"{self.repetition_penalty:.2f}")
                fmt.metric("do_sample:", str(self.do_sample))
            else:
                print("Current Parameters:")
                print(f"  max_new_tokens:      {self.max_new_tokens}")
                print(f"  temperature:         {self.temperature}")
                print(f"  top_k:               {self.top_k}")
                print(f"  top_p:               {self.top_p}")
                print(f"  repetition_penalty:  {self.repetition_penalty}")
                print(f"  do_sample:           {self.do_sample}")
            print()

        elif cmd == '/set':
            if len(parts) < 3:
                if fmt:
                    fmt.error("Usage: /set <parameter> <value>")
                else:
                    print("ERROR: Usage: /set <parameter> <value>")
                return

            param = parts[1]
            try:
                value = float(parts[2]) if '.' in parts[2] else int(parts[2])

                if param == 'max_new_tokens':
                    self.max_new_tokens = int(value)
                elif param == 'temperature':
                    self.temperature = float(value)
                elif param == 'top_k':
                    self.top_k = int(value)
                elif param == 'top_p':
                    self.top_p = float(value)
                elif param == 'repetition_penalty':
                    self.repetition_penalty = float(value)
                elif param == 'do_sample':
                    self.do_sample = bool(int(value))
                else:
                    if fmt:
                        fmt.error(f"Unknown parameter: {param}")
                    else:
                        print(f"ERROR: Unknown parameter: {param}")
                    return

                if fmt:
                    fmt.success(f"Set {param} = {value}")
                else:
                    print(f"SUCCESS: Set {param} = {value}")

            except ValueError:
                if fmt:
                    fmt.error("Invalid value")
                else:
                    print("ERROR: Invalid value")

        elif cmd == '/quit' or cmd == '/exit':
            if fmt:
                fmt.info("Exiting...")
            else:
                print("Exiting...")
            sys.exit(0)

        else:
            if fmt:
                fmt.error(f"Unknown command: {cmd}")
                fmt.info("Type /help for available commands")
            else:
                print(f"ERROR: Unknown command: {cmd}")
                print("Type /help for available commands")