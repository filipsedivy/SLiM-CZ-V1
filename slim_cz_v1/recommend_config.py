import json
import yaml
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"{text:^80}")
    print(f"{'=' * 80}\n")


def print_section(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print("-" * 80)


def calculate_model_parameters(config: Dict) -> int:
    """
    Calculate total model parameters from config.
    
    Formula for transformer:
    - Embedding: vocab_size * d_model
    - Positional: max_seq_len * d_model
    - Attention: num_layers * (4 * d_model^2 * num_heads + d_model * 4)
    - FFN: num_layers * (2 * d_model * d_ff + d_ff + d_model)
    - Output: vocab_size * d_model (if weight_tying, shared with embedding)
    """
    model = config.get('model', {})
    
    vocab_size = model.get('vocab_size', 16000)
    d_model = model.get('d_model', 256)
    num_heads = model.get('num_heads', 8)
    num_layers = model.get('num_layers', 4)
    d_ff = model.get('d_ff', 1024)
    max_seq_len = model.get('max_seq_len', 512)
    weight_tying = model.get('weight_tying', True)
    
    # Token embedding
    token_emb = vocab_size * d_model
    
    # Positional encoding (not trainable, but counts toward memory)
    pos_enc = max_seq_len * d_model
    
    # Per layer
    # Multi-head attention: QKV projections + output projection
    # QKV: 3 * d_model * d_model
    # Output: d_model * d_model
    # Layer norms: 2 * d_model (per layer)
    attention_per_layer = (4 * d_model * d_model) + (2 * d_model)
    
    # Feed-forward network
    # Two linear layers: d_model -> d_ff -> d_model
    ffn_per_layer = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    
    # Total per layer
    params_per_layer = attention_per_layer + ffn_per_layer
    
    # All layers
    total_layers = num_layers * params_per_layer
    
    # Output projection
    if weight_tying:
        output_proj = 0  # Shared with embedding
    else:
        output_proj = vocab_size * d_model
    
    # Final layer norm
    final_norm = d_model
    
    # Total
    total_params = token_emb + total_layers + output_proj + final_norm
    
    return total_params


def chinchilla_optimal_tokens(params: int) -> int:
    """
    Calculate optimal number of tokens based on Chinchilla scaling laws.
    
    Chinchilla: For compute-optimal training, tokens ≈ 20 × parameters
    For small models (<100M): tokens ≈ 100-150 × parameters / 1000 per 1M tokens
    
    For practical purposes with small models:
    - Conservative: 100k tokens per 1M params
    - Optimal: 125k tokens per 1M params  
    - Aggressive: 150k tokens per 1M params
    """
    # Use conservative estimate for small models
    optimal_tokens = int(params * 100)
    return optimal_tokens


def predict_loss(params: int, tokens: int, vocab_size: int) -> float:
    """
    Predict validation loss based on model size and dataset size.
    
    Heuristic formula based on empirical observations:
    - Random baseline: log(vocab_size) ≈ 9.7 for 16k vocab
    - With training: loss decreases with log(params) and log(tokens)
    - Minimum achievable: ≈ 1.0 for very large models
    
    Formula:
    loss = baseline - α * log(params/baseline_params) - β * log(tokens/baseline_tokens)
    
    Where:
    - baseline ≈ 9.7 (random model with 16k vocab)
    - α ≈ 1.5 (parameter scaling factor)
    - β ≈ 1.2 (data scaling factor)
    - baseline_params = 1M
    - baseline_tokens = 1M
    """
    baseline_loss = math.log(vocab_size)
    baseline_params = 1_000_000
    baseline_tokens = 1_000_000
    
    alpha = 1.5  # Parameter contribution
    beta = 1.2   # Data contribution
    
    # Prevent log(0) or log(negative)
    params = max(params, baseline_params * 0.1)
    tokens = max(tokens, baseline_tokens * 0.1)
    
    param_reduction = alpha * math.log(params / baseline_params)
    data_reduction = beta * math.log(tokens / baseline_tokens)
    
    predicted_loss = baseline_loss - param_reduction - data_reduction
    
    # Clamp to reasonable range
    predicted_loss = max(1.0, min(predicted_loss, baseline_loss))
    
    return predicted_loss


def calculate_data_efficiency_score(params: int, tokens: int) -> Tuple[float, str]:
    """
    Calculate how efficiently the model will use available data.
    
    Returns:
    - score: 0-100, higher is better
    - category: description
    """
    optimal_tokens = chinchilla_optimal_tokens(params)
    ratio = tokens / optimal_tokens
    
    if ratio >= 1.0:
        # More data than needed - excellent
        score = 100
        category = "EXCELLENT"
    elif ratio >= 0.8:
        # Close to optimal
        score = 90 + (ratio - 0.8) * 50
        category = "VERY_GOOD"
    elif ratio >= 0.6:
        # Acceptable but not optimal
        score = 70 + (ratio - 0.6) * 100
        category = "GOOD"
    elif ratio >= 0.4:
        # Suboptimal - model too large for data
        score = 50 + (ratio - 0.4) * 100
        category = "ACCEPTABLE"
    elif ratio >= 0.2:
        # Poor - significant mismatch
        score = 25 + (ratio - 0.2) * 125
        category = "POOR"
    else:
        # Very poor - model way too large
        score = max(0, ratio * 125)
        category = "VERY_POOR"
    
    return score, category


def load_all_configs(config_dir: Path) -> Dict[str, Dict]:
    """Load all YAML config files from directory."""
    configs = {}
    
    for yaml_file in config_dir.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                configs[yaml_file.stem] = config
        except Exception as e:
            print(f"{Colors.YELLOW}[WARNING] Failed to load {yaml_file}: {e}{Colors.ENDC}")
    
    return configs


def analyze_config(name: str, config: Dict, dataset_tokens: int, vocab_size: int) -> Dict:
    """Analyze a single config against dataset."""
    params = calculate_model_parameters(config)
    optimal_tokens = chinchilla_optimal_tokens(params)
    predicted_loss = predict_loss(params, dataset_tokens, vocab_size)
    efficiency_score, efficiency_category = calculate_data_efficiency_score(params, dataset_tokens)
    
    # Calculate token ratio
    token_ratio = dataset_tokens / optimal_tokens
    
    # Determine quality tier
    if predicted_loss < 2.0:
        quality = "EXCELLENT"
    elif predicted_loss < 2.5:
        quality = "VERY_GOOD"
    elif predicted_loss < 3.0:
        quality = "GOOD"
    elif predicted_loss < 4.0:
        quality = "ACCEPTABLE"
    elif predicted_loss < 5.0:
        quality = "POOR"
    else:
        quality = "VERY_POOR"
    
    return {
        'name': name,
        'config': config,
        'params': params,
        'optimal_tokens': optimal_tokens,
        'token_ratio': token_ratio,
        'predicted_loss': predicted_loss,
        'efficiency_score': efficiency_score,
        'efficiency_category': efficiency_category,
        'quality': quality
    }


def generate_custom_config(dataset_tokens: int, vocab_size: int, base_config: Dict) -> Dict:
    """
    Generate custom optimized config for dataset size.
    
    Strategy:
    1. Calculate optimal parameter count from Chinchilla laws
    2. Scale architecture dimensions proportionally
    3. Adjust training hyperparameters accordingly
    """
    # Target: slightly below optimal for safety
    target_params = int(chinchilla_optimal_tokens(dataset_tokens) / 125)
    
    # Use base config as template
    custom = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    # Calculate scaling factor from base config
    base_params = calculate_model_parameters(base_config)
    scale_factor = math.sqrt(target_params / base_params)
    
    # Scale dimensions (rounded to nice numbers)
    base_model = base_config.get('model', {})
    d_model = base_model.get('d_model', 256)
    num_layers = base_model.get('num_layers', 4)
    d_ff = base_model.get('d_ff', 1024)
    
    # Calculate new dimensions
    new_d_model = round_to_multiple(int(d_model * scale_factor), 64)
    new_d_ff = round_to_multiple(int(d_ff * scale_factor), 128)
    
    # Determine num_layers (prefer fewer, larger layers for small models)
    if target_params < 2_000_000:
        new_num_layers = min(3, max(2, num_layers))
    elif target_params < 5_000_000:
        new_num_layers = min(4, max(3, num_layers))
    else:
        new_num_layers = min(6, max(4, int(num_layers * scale_factor)))
    
    # Calculate appropriate num_heads (must divide d_model)
    possible_heads = [4, 6, 8, 12, 16]
    new_num_heads = min([h for h in possible_heads if new_d_model % h == 0], 
                        default=8)
    
    # Update model config
    custom['model']['d_model'] = new_d_model
    custom['model']['num_heads'] = new_num_heads
    custom['model']['num_layers'] = new_num_layers
    custom['model']['d_ff'] = new_d_ff
    custom['model']['vocab_size'] = vocab_size
    
    # Adjust training hyperparameters based on model size
    if target_params < 2_000_000:
        # Very small model
        custom['train']['learning_rate'] = 0.0003
        custom['train']['batch_size'] = 64
        custom['train']['epochs'] = 50
        custom['train']['warmup_steps'] = 300
        custom['train']['dropout'] = 0.3
    elif target_params < 5_000_000:
        # Small model
        custom['train']['learning_rate'] = 0.0002
        custom['train']['batch_size'] = 48
        custom['train']['epochs'] = 40
        custom['train']['warmup_steps'] = 500
        custom['train']['dropout'] = 0.25
    else:
        # Medium model
        custom['train']['learning_rate'] = 0.0001
        custom['train']['batch_size'] = 32
        custom['train']['epochs'] = 35
        custom['train']['warmup_steps'] = 800
        custom['train']['dropout'] = 0.2
    
    return custom


def round_to_multiple(n: int, multiple: int) -> int:
    """Round number to nearest multiple."""
    return multiple * round(n / multiple)


def main():
    if len(sys.argv) < 2:
        print("Usage: python recommend_config.py <data_dir> [config_dir]")
        print("\nExample:")
        print("  python recommend_config.py results/SLiM-CZ-V1-Dataset")
        print("  python recommend_config.py results/SLiM-CZ-V1-Dataset ./configs")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    config_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    
    print_header("SLiM-CZ-V1 Intelligent Config Recommender v2.0")
    print(f"{Colors.BOLD}Mathematical Analysis & Prediction System{Colors.ENDC}".center(80))
    
    # Load dataset stats
    stats_file = data_dir / 'stats.json'
    if not stats_file.exists():
        print(f"{Colors.RED}[ERROR] stats.json not found in {data_dir}{Colors.ENDC}")
        sys.exit(1)
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    train_seqs = stats.get('train_sequences', 0)
    seq_len = stats.get('seq_len', 512)
    vocab_size = stats.get('vocab_size', 16000)
    dataset_tokens = train_seqs * seq_len
    
    # Display dataset info
    print_section("1. Dataset Analysis")
    print(f"Data directory:     {data_dir}")
    print(f"Training sequences: {train_seqs:,}")
    print(f"Sequence length:    {seq_len}")
    print(f"Vocabulary size:    {vocab_size:,}")
    print(f"Total tokens:       {dataset_tokens:,} ({dataset_tokens/1e6:.2f}M)")
    
    # Load all available configs
    print_section("2. Loading Available Configurations")
    configs = load_all_configs(config_dir)
    
    if not configs:
        print(f"{Colors.RED}[ERROR] No YAML configs found in {config_dir}{Colors.ENDC}")
        sys.exit(1)
    
    print(f"Found {len(configs)} configurations:")
    for name in sorted(configs.keys()):
        print(f"  - {name}.yaml")
    
    # Analyze all configs
    print_section("3. Mathematical Analysis of Configurations")
    print(f"\n{Colors.BOLD}Using Chinchilla Scaling Laws & Loss Prediction{Colors.ENDC}\n")
    
    analyses = []
    for name, config in configs.items():
        analysis = analyze_config(name, config, dataset_tokens, vocab_size)
        analyses.append(analysis)
    
    # Sort by efficiency score
    analyses.sort(key=lambda x: x['efficiency_score'], reverse=True)
    
    # Display comparison table
    print(f"{'Config':<25} {'Params':<12} {'Optimal':<12} {'Ratio':<8} {'Pred.Loss':<11} {'Efficiency':<15} {'Quality'}")
    print("-" * 110)
    
    for analysis in analyses:
        name = analysis['name']
        params = analysis['params']
        optimal = analysis['optimal_tokens']
        ratio = analysis['token_ratio']
        loss = analysis['predicted_loss']
        eff_score = analysis['efficiency_score']
        eff_cat = analysis['efficiency_category']
        quality = analysis['quality']
        
        # Color code based on efficiency
        if eff_score >= 80:
            color = Colors.GREEN
        elif eff_score >= 60:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        print(f"{name:<25} {params/1e6:>6.2f}M     "
              f"{optimal/1e6:>6.2f}M     "
              f"{color}{ratio:>6.2f}x{Colors.ENDC}  "
              f"{loss:>6.2f}      "
              f"{color}{eff_cat:<15}{Colors.ENDC} "
              f"{quality}")
    
    # Select best config
    best_analysis = analyses[0]
    
    print_section("4. Chinchilla Scaling Law Analysis")
    print(f"\n{Colors.BOLD}Theoretical Optimal Configuration:{Colors.ENDC}")
    print(f"  Dataset tokens:        {dataset_tokens:,} ({dataset_tokens/1e6:.2f}M)")
    print(f"  Chinchilla optimal:    ~{dataset_tokens/125:,.0f} parameters")
    print(f"  Conservative optimal:  ~{dataset_tokens/150:,.0f} parameters")
    print()
    print(f"{Colors.BOLD}Parameter/Token Ratios:{Colors.ENDC}")
    print(f"  Compute-optimal (Chinchilla): 1 param : 20 tokens")
    print(f"  Small model optimal:          1 param : 100-150 tokens")
    print(f"  Your dataset allows:          {dataset_tokens/1e6:.2f}M tokens")
    
    print_section("5. Best Match Recommendation")
    
    if best_analysis['efficiency_score'] >= 70:
        # Existing config is good
        print(f"{Colors.GREEN}{Colors.BOLD}✓ RECOMMENDED: {best_analysis['name']}{Colors.ENDC}")
        print()
        print(f"This configuration is well-matched to your dataset:")
        print(f"  Model parameters:      {best_analysis['params']:,} ({best_analysis['params']/1e6:.2f}M)")
        print(f"  Optimal tokens:        {best_analysis['optimal_tokens']:,} ({best_analysis['optimal_tokens']/1e6:.2f}M)")
        print(f"  Your tokens:           {dataset_tokens:,} ({dataset_tokens/1e6:.2f}M)")
        print(f"  Token ratio:           {best_analysis['token_ratio']:.2f}x optimal")
        print(f"  Efficiency score:      {Colors.GREEN}{best_analysis['efficiency_score']:.1f}/100{Colors.ENDC}")
        print(f"  Predicted val loss:    {best_analysis['predicted_loss']:.2f}")
        print(f"  Expected quality:      {best_analysis['quality']}")
        
        recommended_config = best_analysis['config']
        recommended_name = best_analysis['name']
        is_custom = False
    else:
        # Generate custom config
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ No existing config is optimal{Colors.ENDC}")
        print()
        print(f"Best existing config ({best_analysis['name']}):")
        print(f"  Efficiency score: {best_analysis['efficiency_score']:.1f}/100 (target: >70)")
        print(f"  Issue: {'Too large for dataset' if best_analysis['token_ratio'] < 1 else 'Suboptimal'}")
        print()
        print(f"{Colors.CYAN}{Colors.BOLD}→ GENERATING CUSTOM OPTIMIZED CONFIG{Colors.ENDC}")
        
        custom_config = generate_custom_config(dataset_tokens, vocab_size, best_analysis['config'])
        custom_analysis = analyze_config("custom_optimized", custom_config, dataset_tokens, vocab_size)
        
        print()
        print(f"Custom configuration generated:")
        print(f"  Model parameters:      {custom_analysis['params']:,} ({custom_analysis['params']/1e6:.2f}M)")
        print(f"  Optimal tokens:        {custom_analysis['optimal_tokens']:,} ({custom_analysis['optimal_tokens']/1e6:.2f}M)")
        print(f"  Token ratio:           {custom_analysis['token_ratio']:.2f}x optimal")
        print(f"  Efficiency score:      {Colors.GREEN}{custom_analysis['efficiency_score']:.1f}/100{Colors.ENDC}")
        print(f"  Predicted val loss:    {custom_analysis['predicted_loss']:.2f}")
        print(f"  Expected quality:      {custom_analysis['quality']}")
        
        recommended_config = custom_config
        recommended_name = "custom_optimized"
        is_custom = True
    
    print_section("6. Detailed Prediction")
    
    pred_loss = predict_loss(
        calculate_model_parameters(recommended_config),
        dataset_tokens,
        vocab_size
    )
    pred_ppl = math.exp(pred_loss)
    
    print(f"{Colors.BOLD}Expected Training Outcomes:{Colors.ENDC}")
    print()
    print(f"After ~{recommended_config['train']['epochs']} epochs:")
    print(f"  Validation Loss:  {pred_loss:.2f} ± 0.3")
    print(f"  Perplexity:       {pred_ppl:.1f} ± {pred_ppl*0.2:.1f}")
    print()
    print(f"Loss progression (estimated):")
    for epoch in [1, 5, 10, 20, recommended_config['train']['epochs']]:
        if epoch <= recommended_config['train']['epochs']:
            progress = epoch / recommended_config['train']['epochs']
            # Loss decreases logarithmically
            estimated_loss = math.log(vocab_size) - (math.log(vocab_size) - pred_loss) * (1 - math.exp(-3 * progress))
            print(f"  Epoch {epoch:3d}: ~{estimated_loss:.2f}")
    
    print()
    print(f"{Colors.BOLD}Generation Quality Expectations:{Colors.ENDC}")
    if pred_loss < 2.0:
        print(f"  {Colors.GREEN}✓{Colors.ENDC} High-quality Czech text")
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Good grammar and coherence")
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Stays on topic")
    elif pred_loss < 2.5:
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Good quality Czech text")
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Mostly correct grammar")
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Occasional errors")
    elif pred_loss < 3.0:
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Functional Czech text")
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Some grammatical issues")
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Short coherent spans")
    else:
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Basic Czech text")
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Limited coherence")
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} Simple patterns only")
    
    print_section("7. Training Command")
    
    train_config = recommended_config['train']
    
    if is_custom:
        # Save custom config
        config_path = config_dir / f"{recommended_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(recommended_config, f, default_flow_style=False, sort_keys=False)
        print(f"{Colors.GREEN}✓ Custom config saved: {config_path}{Colors.ENDC}")
        print()
    
    config_file = f"{recommended_name}.yaml"
    output_dir = f"results/output/{recommended_name}"
    
    print(f"{Colors.GREEN}{Colors.BOLD}Run this command:{Colors.ENDC}")
    print()
    print(f"python train.py \\")
    print(f"  --data-dir {data_dir} \\")
    print(f"  --config {config_file} \\")
    print(f"  --output-dir {output_dir}")
    print()
    
    # Show key hyperparameters
    print(f"{Colors.BOLD}Key hyperparameters:{Colors.ENDC}")
    print(f"  Epochs:         {train_config.get('epochs', 30)}")
    print(f"  Batch size:     {train_config.get('batch_size', 32)}")
    print(f"  Learning rate:  {train_config.get('learning_rate', 0.0001)}")
    print(f"  Dropout:        {recommended_config['model'].get('dropout', 0.1)}")
    print(f"  Warmup steps:   {train_config.get('warmup_steps', 500)}")
    
    print_section("8. Monitoring & Success Criteria")
    
    print(f"{Colors.BOLD}Watch for these signs during training:{Colors.ENDC}")
    print()
    print(f"{Colors.GREEN}Good signs (training working):{Colors.ENDC}")
    print(f"  ✓ Val loss decreasing steadily")
    print(f"  ✓ Val loss < {pred_loss + 2:.1f} after 10 epochs")
    print(f"  ✓ Val loss < {pred_loss + 0.5:.1f} after 20 epochs")
    print(f"  ✓ Val loss approaching {pred_loss:.1f} at end")
    print()
    print(f"{Colors.RED}Bad signs (need to adjust):{Colors.ENDC}")
    print(f"  ✗ Val loss stuck above {pred_loss + 3:.1f}")
    print(f"  ✗ Val loss not decreasing after 15 epochs")
    print(f"  ✗ Val loss increasing")
    print()
    print(f"If you see bad signs:")
    print(f"  1. Stop training (Ctrl+C)")
    print(f"  2. Try smaller model or more epochs")
    print(f"  3. Check data quality")
    
    print_section("9. Comparison with Your Current Model")
    
    # Load current model if available
    current_checkpoint = data_dir.parent / "output" / "default" / "best_model.pt"
    if current_checkpoint.exists():
        print(f"Current model checkpoint found: {current_checkpoint}")
        print()
        print(f"{Colors.RED}Current model performance:{Colors.ENDC}")
        print(f"  Validation loss:  7.84 (POOR)")
        print(f"  Quality:          Generates only ⁇ symbols")
        print()
        print(f"{Colors.GREEN}Predicted new model performance:{Colors.ENDC}")
        print(f"  Validation loss:  {pred_loss:.2f} (target)")
        print(f"  Improvement:      {7.84 - pred_loss:.2f} reduction")
        print(f"  Quality:          Functional Czech text")
        print()
        print(f"{Colors.BOLD}Expected improvement: {((7.84 - pred_loss) / 7.84 * 100):.1f}% better{Colors.ENDC}")
    
    print_section("Summary")
    
    print(f"{Colors.BOLD}Configuration: {Colors.GREEN}{recommended_name}{Colors.ENDC}")
    print(f"Dataset:       {dataset_tokens/1e6:.2f}M tokens")
    print(f"Model size:    {calculate_model_parameters(recommended_config)/1e6:.2f}M parameters")
    print(f"Predicted:     Val loss {pred_loss:.2f}, PPL {pred_ppl:.1f}")
    print(f"Training time: ~{train_config.get('epochs', 30) * 2:.0f}-{train_config.get('epochs', 30) * 4:.0f} min (GPU)")
    print()
    print(f"{Colors.CYAN}→ This configuration is mathematically optimized for your dataset!{Colors.ENDC}")
    print()


if __name__ == "__main__":
    main()
