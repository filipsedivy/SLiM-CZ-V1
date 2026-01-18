import sys
from pathlib import Path
import json

try:
    import torch
    import sentencepiece as spm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch sentencepiece")
    sys.exit(1)


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"{text:^80}")
    print(f"{'=' * 80}\n")


def print_section(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print("-" * 80)


def check_checkpoint(checkpoint_path):
    """Analyze checkpoint file."""
    print_section("1. Checkpoint Analysis")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic info
        epoch = checkpoint.get('epoch', 'unknown')
        train_loss = checkpoint.get('train_loss', 'unknown')
        val_loss = checkpoint.get('best_val_loss', 'unknown')
        val_ppl = checkpoint.get('best_val_perplexity', 'unknown')
        
        print(f"Checkpoint file: {checkpoint_path}")
        print(f"Training epoch: {epoch}")
        print(f"Training loss: {train_loss}")
        print(f"Validation loss: {val_loss}")
        print(f"Validation perplexity: {val_ppl}")
        
        # Assess quality
        issues = []
        warnings = []
        
        if epoch == 0 or epoch == 'unknown':
            issues.append("Model appears to be at epoch 0 (UNTRAINED!)")
        elif isinstance(epoch, int) and epoch < 5:
            warnings.append(f"Model only trained for {epoch} epochs (recommended: 20-50)")
        
        if isinstance(val_loss, (int, float)):
            if val_loss > 5.0:
                issues.append(f"Validation loss very high: {val_loss:.4f} (should be < 3.0)")
            elif val_loss > 3.0:
                warnings.append(f"Validation loss high: {val_loss:.4f} (target: < 2.0)")
        
        if isinstance(val_ppl, (int, float)):
            if val_ppl > 150:
                issues.append(f"Perplexity very high: {val_ppl:.2f} (should be < 50)")
            elif val_ppl > 50:
                warnings.append(f"Perplexity high: {val_ppl:.2f} (target: < 30)")
        
        # Model state
        if 'model_state_dict' not in checkpoint:
            issues.append("No model_state_dict found in checkpoint!")
        else:
            state_dict = checkpoint['model_state_dict']
            
            # Check if weights look initialized
            if 'token_embedding.weight' in state_dict:
                emb = state_dict['token_embedding.weight']
                mean = emb.mean().item()
                std = emb.std().item()
                
                print(f"\nEmbedding statistics:")
                print(f"  Mean: {mean:.6f}")
                print(f"  Std:  {std:.6f}")
                
                if abs(mean) < 0.001 and std < 0.01:
                    issues.append("Embeddings appear uninitialized (near zero)")
        
        # Print issues
        print()
        if issues:
            print(f"{Colors.RED}[CRITICAL ISSUES]{Colors.ENDC}")
            for issue in issues:
                print(f"  {Colors.RED}✗{Colors.ENDC} {issue}")
        
        if warnings:
            print(f"\n{Colors.YELLOW}[WARNINGS]{Colors.ENDC}")
            for warning in warnings:
                print(f"  {Colors.YELLOW}⚠{Colors.ENDC} {warning}")
        
        if not issues and not warnings:
            print(f"{Colors.GREEN}✓ Checkpoint looks good!{Colors.ENDC}")
        
        return len(issues) == 0, checkpoint
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Failed to load checkpoint: {e}{Colors.ENDC}")
        return False, None


def check_tokenizer(tokenizer_path, data_dir=None):
    """Analyze tokenizer."""
    print_section("2. Tokenizer Analysis")
    
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))
        
        vocab_size = tokenizer.get_piece_size()
        print(f"Tokenizer file: {tokenizer_path}")
        print(f"Vocabulary size: {vocab_size:,}")
        
        # Check if tokenizer matches training data
        if data_dir:
            stats_file = Path(data_dir) / 'stats.json'
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                
                expected_vocab = stats.get('vocab_size')
                if expected_vocab and expected_vocab != vocab_size:
                    print(f"{Colors.RED}✗ MISMATCH: Training data expects vocab_size={expected_vocab}{Colors.ENDC}")
                    return False
                else:
                    print(f"{Colors.GREEN}✓ Vocab size matches training data{Colors.ENDC}")
        
        # Test tokenization
        test_texts = [
            "Praha je hlavní město České republiky",
            "Dnes je krásný den",
            "Ahoj, jak se máš?"
        ]
        
        print(f"\nTokenization test:")
        issues = []
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            # Count unknown tokens
            unknown_count = sum(1 for t in tokens if tokenizer.id_to_piece(t) == '<unk>')
            unknown_pct = (unknown_count / len(tokens)) * 100 if tokens else 0
            
            print(f"\n  Input:    {text}")
            print(f"  Tokens:   {len(tokens)} tokens")
            print(f"  Unknown:  {unknown_count} ({unknown_pct:.1f}%)")
            print(f"  Decoded:  {decoded}")
            
            if unknown_pct > 20:
                issues.append(f"High unknown token rate ({unknown_pct:.1f}%) for Czech text")
            
            if text != decoded:
                issues.append(f"Tokenization not reversible")
        
        # Check special tokens
        print(f"\nSpecial tokens:")
        print(f"  PAD: {tokenizer.pad_id()} = '{tokenizer.id_to_piece(tokenizer.pad_id())}'")
        print(f"  UNK: {tokenizer.unk_id()} = '{tokenizer.id_to_piece(tokenizer.unk_id())}'")
        print(f"  BOS: {tokenizer.bos_id()} = '{tokenizer.id_to_piece(tokenizer.bos_id())}'")
        print(f"  EOS: {tokenizer.eos_id()} = '{tokenizer.id_to_piece(tokenizer.eos_id())}'")
        
        if issues:
            print(f"\n{Colors.RED}[ISSUES]{Colors.ENDC}")
            for issue in issues:
                print(f"  {Colors.RED}✗{Colors.ENDC} {issue}")
            return False
        else:
            print(f"\n{Colors.GREEN}✓ Tokenizer working correctly{Colors.ENDC}")
            return True
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Failed to load tokenizer: {e}{Colors.ENDC}")
        return False


def check_training_data(data_dir):
    """Analyze training data statistics."""
    print_section("3. Training Data Analysis")
    
    stats_file = Path(data_dir) / 'stats.json'
    
    if not stats_file.exists():
        print(f"{Colors.YELLOW}[WARNING] No stats.json found in {data_dir}{Colors.ENDC}")
        return True
    
    try:
        with open(stats_file) as f:
            stats = json.load(f)
        
        print(f"Data directory: {data_dir}")
        print(f"Vocabulary size: {stats.get('vocab_size', 'N/A'):,}")
        print(f"Sequence length: {stats.get('seq_len', 'N/A')}")
        print(f"Training sequences: {stats.get('train_sequences', 'N/A'):,}")
        print(f"Validation sequences: {stats.get('val_sequences', 'N/A'):,}")
        
        # Calculate dataset size
        train_seqs = stats.get('train_sequences', 0)
        seq_len = stats.get('seq_len', 512)
        total_tokens = train_seqs * seq_len
        
        print(f"\nDataset size:")
        print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e6:.2f}M)")
        
        # Assess quality
        issues = []
        warnings = []
        
        if total_tokens < 1_000_000:
            issues.append(f"Dataset very small ({total_tokens/1e6:.2f}M tokens, need 2-5M)")
        elif total_tokens < 2_000_000:
            warnings.append(f"Dataset small ({total_tokens/1e6:.2f}M tokens, recommended 2-5M)")
        
        if train_seqs < 1000:
            warnings.append(f"Few training sequences ({train_seqs}, need more data)")
        
        print()
        if issues:
            print(f"{Colors.RED}[CRITICAL ISSUES]{Colors.ENDC}")
            for issue in issues:
                print(f"  {Colors.RED}✗{Colors.ENDC} {issue}")
        
        if warnings:
            print(f"\n{Colors.YELLOW}[WARNINGS]{Colors.ENDC}")
            for warning in warnings:
                print(f"  {Colors.YELLOW}⚠{Colors.ENDC} {warning}")
        
        if not issues and not warnings:
            print(f"{Colors.GREEN}✓ Training data looks adequate{Colors.ENDC}")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Failed to read stats: {e}{Colors.ENDC}")
        return False


def test_generation(checkpoint_path, tokenizer_path):
    """Test actual text generation."""
    print_section("4. Generation Test")
    
    try:
        from model import SLiM_CZ_V1
        
        # Load tokenizer
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(tokenizer_path))
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Detect max_seq_len
        if 'pos_encoding.pe' in checkpoint['model_state_dict']:
            max_seq_len = checkpoint['model_state_dict']['pos_encoding.pe'].shape[0]
        else:
            max_seq_len = 512
        
        # Create model
        config = checkpoint.get('config', {}).get('model', {})
        model = SLiM_CZ_V1(
            vocab_size=tokenizer.get_piece_size(),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            d_ff=config.get('d_ff', 1024),
            max_seq_len=max_seq_len,
            dropout=0.0,  # Disable dropout for inference
            weight_tying=config.get('weight_tying', True)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test generation
        test_prompts = [
            "Praha je",
            "Dnes je",
            "Česká republika"
        ]
        
        print("Generating short samples...\n")
        
        for prompt in test_prompts:
            # Encode
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                for _ in range(10):  # Generate just 10 tokens
                    logits, _ = model(input_tensor)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Decode
            generated_ids = input_tensor[0].tolist()
            generated_text = tokenizer.decode(generated_ids)
            
            print(f"Prompt:    {prompt}")
            print(f"Generated: {generated_text}")
            
            # Check for issues
            unknown_count = generated_text.count('⁇')
            if unknown_count > 0:
                print(f"{Colors.RED}  ✗ Contains {unknown_count} unknown characters (⁇){Colors.ENDC}")
            else:
                print(f"{Colors.GREEN}  ✓ No unknown characters{Colors.ENDC}")
            print()
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Generation test failed: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python diagnose.py <checkpoint.pt> <tokenizer.model> [data_dir]")
        print("\nExample:")
        print("  python diagnose.py output/best_model.pt data/tokenizer.model data/")
        sys.exit(1)
    
    checkpoint_path = Path(sys.argv[1])
    tokenizer_path = Path(sys.argv[2])
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print_header("SLiM-CZ-V1 Diagnostic Tool")
    
    # Run checks
    checkpoint_ok, checkpoint = check_checkpoint(checkpoint_path)
    tokenizer_ok = check_tokenizer(tokenizer_path, data_dir)
    data_ok = check_training_data(data_dir) if data_dir else True
    generation_ok = test_generation(checkpoint_path, tokenizer_path)
    
    # Summary
    print_section("Summary & Recommendations")
    
    all_checks = {
        "Checkpoint": checkpoint_ok,
        "Tokenizer": tokenizer_ok,
        "Training Data": data_ok,
        "Generation": generation_ok
    }
    
    for name, status in all_checks.items():
        status_str = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if status else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
        print(f"  {name:20s} {status_str}")
    
    print()
    
    if not checkpoint_ok:
        print(f"{Colors.RED}[ACTION REQUIRED]{Colors.ENDC}")
        print("Your model checkpoint has issues:")
        print("  1. Check if training completed successfully")
        print("  2. Use a checkpoint from a later epoch (best_model.pt)")
        print("  3. If at epoch 0, you need to train the model first")
        print("  4. Check training logs for errors")
    
    if not tokenizer_ok:
        print(f"{Colors.RED}[ACTION REQUIRED]{Colors.ENDC}")
        print("Your tokenizer has issues:")
        print("  1. Ensure tokenizer.model was created during data preparation")
        print("  2. Check that Czech text is properly encoded")
        print("  3. Verify vocab_size matches between tokenizer and model")
        print("  4. Consider retraining tokenizer with more data")
    
    if not data_ok:
        print(f"{Colors.YELLOW}[RECOMMENDATION]{Colors.ENDC}")
        print("Your training data is small:")
        print("  1. Collect more Czech text (aim for 2-5M tokens)")
        print("  2. Use data augmentation techniques")
        print("  3. Consider using a smaller model (tiny config)")
        print("  4. Increase training epochs (30-50)")
    
    if not generation_ok:
        print(f"{Colors.RED}[ACTION REQUIRED]{Colors.ENDC}")
        print("Generation test failed. Check above errors.")
    
    if all(all_checks.values()):
        print(f"{Colors.GREEN}[ALL CHECKS PASSED]{Colors.ENDC}")
        print("Your model should be working correctly!")
        print("If you still see issues, try:")
        print("  1. Adjusting generation parameters (temperature, top_k)")
        print("  2. Using more training epochs")
        print("  3. Increasing training data size")


if __name__ == "__main__":
    main()
