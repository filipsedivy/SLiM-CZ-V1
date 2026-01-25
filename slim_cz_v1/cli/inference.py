#!/usr/bin/env python3
"""
CLI tool for SLiM-CZ-V1 model inference.

Supports single prompt generation, batch processing, and interactive mode.
"""

import argparse
import sys
from pathlib import Path

from ..inference import ModelLoader, TextGenerator, InteractiveMode


# ============================================================
# CLI OUTPUT FORMATTING
# ============================================================

class CLIFormatter:
    """
    CLI output formatter with ANSI color support.

    Follows terminal UI/UX best practices:
    - Plaintext only (no Markdown)
    - Clear visual hierarchy
    - ANSI color codes for emphasis
    - Unicode symbols for quick scanning
    """

    # ANSI color codes
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def info(message: str):
        """Print info message."""
        print(f"{CLIFormatter.BLUE}[INFO]{CLIFormatter.RESET}    {message}")

    @staticmethod
    def success(message: str):
        """Print success message."""
        print(f"{CLIFormatter.GREEN}[SUCCESS]{CLIFormatter.RESET} ✔ {message}")

    @staticmethod
    def warning(message: str):
        """Print warning message."""
        print(f"{CLIFormatter.YELLOW}[WARNING]{CLIFormatter.RESET} ⚠ {message}")

    @staticmethod
    def error(message: str):
        """Print error message."""
        print(f"{CLIFormatter.RED}[ERROR]{CLIFormatter.RESET}   ✖ {message}")

    @staticmethod
    def header(title: str):
        """Print section header."""
        separator = "=" * 70
        print(f"\n{CLIFormatter.BOLD}{separator}{CLIFormatter.RESET}")
        print(f"{CLIFormatter.BOLD}{title}{CLIFormatter.RESET}")
        print(f"{CLIFormatter.BOLD}{separator}{CLIFormatter.RESET}\n")

    @staticmethod
    def section(title: str):
        """Print subsection."""
        print(f"\n{CLIFormatter.CYAN}{title}{CLIFormatter.RESET}")
        print("-" * 70)

    @staticmethod
    def metric(name: str, value: str, unit: str = ""):
        """Print metric in aligned format."""
        unit_str = f" {unit}" if unit else ""
        print(f"  {name:30s} {value}{unit_str}")


# ============================================================
# MAIN CLI FUNCTION
# ============================================================

def main():
    """Main entry point for inference CLI."""

    parser = argparse.ArgumentParser(
        description='SLiM-CZ-V1 Model Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  slim-infer --checkpoint model.pt --tokenizer tokenizer.model

  # Single prompt generation
  slim-infer --checkpoint model.pt --tokenizer tokenizer.model \\
    --prompt "Dnes je krásný den" --max-tokens 150

  # Batch generation from file
  slim-infer --checkpoint model.pt --tokenizer tokenizer.model \\
    --prompts-file prompts.txt --output results.txt

  # Custom generation parameters
  slim-infer --checkpoint model.pt --tokenizer tokenizer.model \\
    --prompt "Praha je" --temperature 0.9 --top-k 40 --top-p 0.95

Generation Parameters:
  - temperature: Controls randomness (0.0 = deterministic, 2.0 = very random)
  - top_k: Limits vocabulary to top K tokens (0 = disabled)
  - top_p: Nucleus sampling threshold (0.0-1.0)
  - repetition_penalty: Penalizes repeated tokens (1.0 = no penalty)

Requirements:
  - Trained model checkpoint (.pt file)
  - Tokenizer model (.model file)
  - PyTorch >= 2.0.0
  - SentencePiece
        """
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--tokenizer', '-t',
        type=str,
        required=True,
        help='Path to tokenizer model (.model file)'
    )

    # Generation mode
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Single prompt for generation'
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        help='File with prompts (one per line)'
    )
    parser.add_argument(
        '--output', '-o',
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

    # ============================================================
    # HEADER
    # ============================================================

    CLIFormatter.header("SLiM-CZ-V1 Inference")
    CLIFormatter.info("Slavic Linguistic integrated Micro-model for Czechia")
    print()

    # ============================================================
    # LOAD MODEL
    # ============================================================

    loader = ModelLoader(args.checkpoint, args.tokenizer, args.device)
    loader.load(cli_formatter=CLIFormatter)

    # Display model info
    info = loader.get_model_info()

    print()
    CLIFormatter.metric("Model name:", info['architecture']['name'])
    CLIFormatter.metric("Description:", info['architecture']['description'])
    print()

    CLIFormatter.section("Architecture")
    CLIFormatter.metric("Layers:", str(info['architecture']['num_layers']))
    CLIFormatter.metric("Attention heads:", str(info['architecture']['num_heads']))
    CLIFormatter.metric("Embedding dimension:", str(info['architecture']['d_model']))
    CLIFormatter.metric("Feed-forward dimension:", str(info['architecture']['d_ff']))
    CLIFormatter.metric("Max sequence length:", str(info['architecture']['max_seq_len']))
    CLIFormatter.metric("Vocabulary size:", f"{info['architecture']['vocab_size']:,}")
    print()

    CLIFormatter.section("Model Size")
    CLIFormatter.metric("Total parameters:", f"{info['parameters']['total']:,}")
    CLIFormatter.metric("Size (M parameters):", f"{info['parameters']['total_millions']:.2f}M")
    CLIFormatter.metric("Memory (FP32):", f"{info['parameters']['memory_fp32_mb']:.1f} MB")
    CLIFormatter.metric("Memory (FP16):", f"{info['parameters']['memory_fp16_mb']:.1f} MB")
    print()

    CLIFormatter.section("Training Checkpoint")
    CLIFormatter.metric("Epoch:", str(info['checkpoint']['epoch']))

    # Display validation loss
    val_loss = info['checkpoint']['val_loss']
    if isinstance(val_loss, (int, float)):
        CLIFormatter.metric("Validation loss:", f"{val_loss:.6f}")
    else:
        CLIFormatter.metric("Validation loss:", str(val_loss))

    # Display perplexity with mathematical interpretation
    val_ppl = info['checkpoint']['val_perplexity']
    if isinstance(val_ppl, (int, float)):
        CLIFormatter.metric("Validation perplexity:", f"{val_ppl:.4f}")
        print()
        CLIFormatter.info("Formula: PPL = exp(loss)")

        # Interpretation based on empirical research
        if val_ppl < 10:
            CLIFormatter.success("Excellent model quality (PPL < 10)")
        elif val_ppl < 50:
            CLIFormatter.info("Good model quality (10 <= PPL < 50)")
        elif val_ppl < 100:
            CLIFormatter.warning("Moderate model quality (50 <= PPL < 100)")
        else:
            CLIFormatter.warning("High perplexity (PPL >= 100)")
    else:
        CLIFormatter.metric("Validation perplexity:", str(val_ppl))

    print()
    CLIFormatter.metric("Device:", info['device'])
    print()

    # Create generator
    generator = TextGenerator(loader.model, loader.tokenizer, loader.device)

    # ============================================================
    # DETERMINE MODE AND EXECUTE
    # ============================================================

    if args.prompt:
        # Single prompt generation
        CLIFormatter.section("Single Prompt Generation")

        CLIFormatter.metric("Prompt:", args.prompt)
        CLIFormatter.metric("Max tokens:", str(args.max_tokens))
        CLIFormatter.metric("Temperature:", f"{args.temperature:.2f}")
        CLIFormatter.metric("Top-k:", str(args.top_k))
        CLIFormatter.metric("Top-p:", f"{args.top_p:.2f}")
        CLIFormatter.metric("Repetition penalty:", f"{args.repetition_penalty:.2f}")
        CLIFormatter.metric("Sampling:", "Disabled" if args.no_sample else "Enabled")
        print()

        CLIFormatter.info("Generating...")
        print("-" * 70)

        result = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            show_prompt=True
        )

        print(result['text'])
        print("-" * 70)

        # Display statistics
        stats = result['statistics']
        print()
        CLIFormatter.section("Generation Statistics")
        CLIFormatter.metric("Prompt tokens:", str(stats['prompt_tokens']))
        CLIFormatter.metric("Generated tokens:", str(stats['generated_tokens']))
        CLIFormatter.metric("Total tokens:", str(stats['total_tokens']))
        CLIFormatter.metric("Time elapsed:", f"{stats['elapsed_time']:.3f}s")
        CLIFormatter.metric("Generation speed:", f"{stats['tokens_per_second']:.2f} tokens/sec")

        # Mathematical interpretation of speed
        print()
        CLIFormatter.info("Formula: speed = tokens / time")

        if stats['tokens_per_second'] > 100:
            CLIFormatter.success("Excellent generation speed (>100 tok/s)")
        elif stats['tokens_per_second'] > 50:
            CLIFormatter.info("Good generation speed (50-100 tok/s)")
        elif stats['tokens_per_second'] > 20:
            CLIFormatter.warning("Moderate generation speed (20-50 tok/s)")
        else:
            CLIFormatter.warning("Low generation speed (<20 tok/s)")
            CLIFormatter.info("Consider: GPU acceleration or model optimization")

        if stats['stopped_early']:
            print()
            CLIFormatter.info("Generation stopped at EOS token (before max_tokens)")

        print()
        CLIFormatter.success("Generation completed")
        print("=" * 70)
        print()

    elif args.prompts_file:
        # Batch generation from file
        CLIFormatter.section("Batch Generation")

        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            CLIFormatter.error(f"Prompts file not found: {args.prompts_file}")
            return 1

        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        CLIFormatter.info(f"Loaded {len(prompts)} prompts from {prompts_path.name}")
        print()

        # Progress callback
        def progress_callback(current, total, stats):
            print(f"  [{current}/{total}] {stats['generated_tokens']} tokens in " +
                  f"{stats['elapsed_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)")

        results = generator.batch_generate(
            prompts=prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
            progress_callback=progress_callback
        )

        # Calculate batch statistics
        total_time = sum(r['statistics']['elapsed_time'] for r in results)
        total_tokens = sum(r['statistics']['generated_tokens'] for r in results)
        avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        print()
        CLIFormatter.section("Batch Statistics")
        CLIFormatter.metric("Total prompts:", str(len(prompts)))
        CLIFormatter.metric("Total tokens generated:", f"{total_tokens:,}")
        CLIFormatter.metric("Total time:", f"{total_time:.2f}s")
        CLIFormatter.metric("Average speed:", f"{avg_tokens_per_sec:.1f} tokens/sec")
        print()

        # Save or display results
        if args.output:
            output_path = Path(args.output)

            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt, result in zip(prompts, results):
                    f.write(f"PROMPT: {prompt}\n")
                    f.write(f"GENERATED: {result['text']}\n")
                    f.write("-" * 70 + "\n\n")

            CLIFormatter.success(f"Results saved to: {args.output}")
            CLIFormatter.metric("Output file:", str(output_path))
        else:
            # Display results
            CLIFormatter.section("Generated Results")
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                print(f"\n[{i + 1}] PROMPT:")
                print(f"    {prompt}")
                print(f"    GENERATED:")
                print(f"    {result['text']}")

        print()
        CLIFormatter.success("Batch generation completed")
        print("=" * 70)
        print()

    else:
        # Interactive mode
        interactive = InteractiveMode(generator, loader.config, cli_formatter=CLIFormatter)
        interactive.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())