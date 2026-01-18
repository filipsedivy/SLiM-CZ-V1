import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False

from slim_cz_v1 import SLiM_CZ_V1


# ============================================================================
# ANSI Color Codes
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ============================================================================
# Print Utilities
# ============================================================================

def print_header(text: str):
    """Print formatted header."""
    width = 80
    print(f"\n{'=' * width}")
    print(f"{text:^{width}}")
    print(f"{'=' * width}")


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 80)


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}[SUCCESS] {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}[INFO] {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}[ERROR] {text}{Colors.ENDC}")


# ============================================================================
# Progress Spinner
# ============================================================================

class Spinner:
    """Simple spinner for operations."""
    
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


# ============================================================================
# Model Loader
# ============================================================================

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
    
    def load(self):
        """Load model and tokenizer."""
        print_section("Loading Model & Tokenizer")
        
        # Check files exist
        if not self.checkpoint_path.exists():
            print_error(f"Checkpoint not found: {self.checkpoint_path}")
            sys.exit(1)
        
        if not self.tokenizer_path.exists():
            print_error(f"Tokenizer not found: {self.tokenizer_path}")
            sys.exit(1)
        
        # Load tokenizer
        spinner = Spinner("Loading tokenizer")
        spinner.start()
        
        if not HAS_SENTENCEPIECE:
            spinner.stop()
            print_error("SentencePiece not installed!")
            print_info("Install with: pip install sentencepiece")
            sys.exit(1)
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(self.tokenizer_path))
        spinner.stop(f"{Colors.GREEN}✔ Tokenizer loaded: {self.tokenizer.get_piece_size()} tokens{Colors.ENDC}")
        
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
        
        spinner.stop(f"{Colors.GREEN}✔ Checkpoint loaded{Colors.ENDC}")
        
        # Detect max_seq_len from checkpoint (from positional encoding size)
        # This is necessary because max_seq_len comes from data stats during training,
        # not from the config file
        max_seq_len_detected = None
        if 'model_state_dict' in checkpoint:
            if 'pos_encoding.pe' in checkpoint['model_state_dict']:
                pe_shape = checkpoint['model_state_dict']['pos_encoding.pe'].shape
                max_seq_len_detected = pe_shape[0]

        # Create model
        spinner = Spinner("Initializing model")
        spinner.start()

        model_config = self.config.get('model', {})

        # Use detected max_seq_len if available, otherwise fall back to config
        max_seq_len = max_seq_len_detected or model_config.get('max_seq_len', 512)

        self.model = SLiM_CZ_V1(
            vocab_size=self.tokenizer.get_piece_size(),
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

        spinner.stop(f"{Colors.GREEN}✔ Model initialized{Colors.ENDC}")

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print model information."""
        params = self.model.count_parameters()
        model_cfg = self.config.get('model', {})

        print()
        print(f"  Model Architecture:")
        print(f"    Name:        SLiM-CZ-V1")
        print(f"    Description: Slavic Linguistic integrated Micro-model for Czechia")
        print(f"    Layers:      {model_cfg.get('num_layers', 'N/A')}")
        print(f"    Heads:       {model_cfg.get('num_heads', 'N/A')}")
        print(f"    Embedding:   {model_cfg.get('d_model', 'N/A')}")
        print(f"    FFN:         {model_cfg.get('d_ff', 'N/A')}")
        print(f"    Max Seq Len: {self.model.max_seq_len}")
        print(f"    Parameters:  {params['total']:,} ({params['total']/1e6:.2f}M)")
        print()
        print(f"  Checkpoint Info:")
        print(f"    Epoch:       {self.checkpoint_info['epoch']}")
        print(f"    Val Loss:    {self.checkpoint_info['val_loss']}")
        print(f"    Val PPL:     {self.checkpoint_info['val_perplexity']}")
        print()
        print(f"  Device:        {self.device}")
        print()


# ============================================================================
# Text Generator
# ============================================================================

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
    ) -> str:
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
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Track token history for repetition penalty
        generated_tokens = []
        
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
        
        # Decode
        generated_ids = input_tensor[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        if not show_prompt:
            # Try to extract only generated part
            generated_only_ids = generated_ids[len(input_ids):]
            generated_text = self.tokenizer.decode(generated_only_ids)
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            Other args: Same as generate()
        
        Returns:
            List of generated texts
        """
        results = []
        
        print_info(f"Generating {len(prompts)} completions...")
        print()
        
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Generating...", end=' ')
            sys.stdout.flush()
            
            start_time = time.time()
            text = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                show_prompt=False
            )
            elapsed = time.time() - start_time
            
            print(f"{Colors.GREEN}✔{Colors.ENDC} ({elapsed:.2f}s)")
            results.append(text)
        
        print()
        return results


# ============================================================================
# Interactive Mode
# ============================================================================

class InteractiveMode:
    """Interactive text generation interface."""
    
    def __init__(self, generator: TextGenerator, config: Dict[str, Any]):
        """
        Initialize interactive mode.
        
        Args:
            generator: TextGenerator instance
            config: Generation config
        """
        self.generator = generator
        self.gen_config = config.get('generation', {})
        
        # Default parameters
        self.max_new_tokens = self.gen_config.get('max_new_tokens', 100)
        self.temperature = self.gen_config.get('temperature', 0.8)
        self.top_k = self.gen_config.get('top_k', 50)
        self.top_p = self.gen_config.get('top_p', 0.95)
        self.repetition_penalty = self.gen_config.get('repetition_penalty', 1.2)
        self.do_sample = self.gen_config.get('do_sample', True)
    
    def run(self):
        """Run interactive generation loop."""
        print_header("Interactive Generation Mode")
        print(f"{Colors.BOLD}SLiM-CZ-V1 - Slavic Linguistic integrated Micro-model for Czechia{Colors.ENDC}".center(80))
        print()
        
        self._print_help()
        
        while True:
            try:
                print()
                prompt = input(f"{Colors.CYAN}> {Colors.ENDC}").strip()
                
                if not prompt:
                    continue
                
                # Handle commands
                if prompt.startswith('/'):
                    self._handle_command(prompt)
                    continue
                
                # Generate
                print()
                print(f"{Colors.YELLOW}[GENERATING]{Colors.ENDC}")
                print("-" * 80)
                
                start_time = time.time()
                generated = self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=self.do_sample,
                    show_prompt=True
                )
                elapsed = time.time() - start_time
                
                print(generated)
                print("-" * 80)
                print(f"{Colors.GREEN}[COMPLETE]{Colors.ENDC} Generated in {elapsed:.2f}s")
                
            except KeyboardInterrupt:
                print()
                print()
                print_info("Exiting interactive mode...")
                break
            except EOFError:
                print()
                break
            except Exception as e:
                print_error(f"Generation failed: {e}")
    
    def _print_help(self):
        """Print help message."""
        print("Commands:")
        print(f"  {Colors.CYAN}/help{Colors.ENDC}        Show this help")
        print(f"  {Colors.CYAN}/params{Colors.ENDC}      Show current parameters")
        print(f"  {Colors.CYAN}/set{Colors.ENDC}         Set parameter (e.g., /set temperature 0.9)")
        print(f"  {Colors.CYAN}/quit{Colors.ENDC}        Exit")
        print()
        print("Just type your prompt and press Enter to generate!")
    
    def _handle_command(self, command: str):
        """Handle special commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            self._print_help()
        
        elif cmd == '/params':
            print()
            print("Current Parameters:")
            print(f"  max_new_tokens:      {self.max_new_tokens}")
            print(f"  temperature:         {self.temperature}")
            print(f"  top_k:               {self.top_k}")
            print(f"  top_p:               {self.top_p}")
            print(f"  repetition_penalty:  {self.repetition_penalty}")
            print(f"  do_sample:           {self.do_sample}")
        
        elif cmd == '/set':
            if len(parts) < 3:
                print_error("Usage: /set <parameter> <value>")
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
                    self.do_sample = bool(value)
                else:
                    print_error(f"Unknown parameter: {param}")
                    return
                
                print_success(f"Set {param} = {value}")
            
            except ValueError:
                print_error("Invalid value")
        
        elif cmd == '/quit' or cmd == '/exit':
            print_info("Exiting...")
            sys.exit(0)
        
        else:
            print_error(f"Unknown command: {cmd}")
            print_info("Type /help for available commands")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🇨🇿 SLiM-CZ-V1 Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python inference.py --checkpoint model.pt --tokenizer tokenizer.model
  
  # Generate from prompt
  python inference.py --checkpoint model.pt --tokenizer tokenizer.model \\
    --prompt "Dnes je krásný den" --max-tokens 150
  
  # Batch generation from file
  python inference.py --checkpoint model.pt --tokenizer tokenizer.model \\
    --prompts-file prompts.txt --output results.txt
  
  # Custom parameters
  python inference.py --checkpoint model.pt --tokenizer tokenizer.model \\
    --prompt "Praha je" --temperature 0.9 --top-k 40 --top-p 0.95
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Path to tokenizer model (.model file)'
    )
    
    # Generation mode
    parser.add_argument(
        '--prompt',
        type=str,
        help='Single prompt for generation'
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        help='File with prompts (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for batch generation'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50, 0 = disabled)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling threshold (default: 0.95)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.2,
        help='Repetition penalty (default: 1.2)'
    )
    parser.add_argument(
        '--no-sample',
        action='store_true',
        help='Use greedy decoding instead of sampling'
    )
    
    # Other options
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cuda, cpu (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header("🇨🇿 SLiM-CZ-V1 Inference")
    print(f"{Colors.BOLD}Slavic Linguistic integrated Micro-model for Czechia{Colors.ENDC}".center(80))
    
    # Load model and tokenizer
    loader = ModelLoader(args.checkpoint, args.tokenizer, args.device)
    loader.load()
    
    # Create generator
    generator = TextGenerator(loader.model, loader.tokenizer, loader.device)
    
    # Determine mode
    if args.prompt:
        # Single prompt generation
        print_section("Single Prompt Generation")
        print_info(f"Prompt: {args.prompt}")
        print_info(f"Max tokens: {args.max_tokens}")
        print()
        
        print(f"{Colors.YELLOW}[GENERATING]{Colors.ENDC}")
        print("-" * 80)
        
        start_time = time.time()
        generated = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            show_prompt=True
        )
        elapsed = time.time() - start_time
        
        print(generated)
        print("-" * 80)
        print(f"{Colors.GREEN}[COMPLETE]{Colors.ENDC} Generated in {elapsed:.2f}s")
    
    elif args.prompts_file:
        # Batch generation from file
        print_section("Batch Generation")
        
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print_error(f"Prompts file not found: {args.prompts_file}")
            sys.exit(1)
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print_info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        print()
        
        results = generator.batch_generate(
            prompts=prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample
        )
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt, result in zip(prompts, results):
                    f.write(f"PROMPT: {prompt}\n")
                    f.write(f"GENERATED: {result}\n")
                    f.write("-" * 80 + "\n\n")
            
            print_success(f"Results saved to: {args.output}")
        else:
            # Print results
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                print(f"\n[{i+1}] PROMPT: {prompt}")
                print(f"    GENERATED: {result}")
    
    else:
        # Interactive mode
        interactive = InteractiveMode(generator, loader.config)
        interactive.run()


if __name__ == "__main__":
    main()
