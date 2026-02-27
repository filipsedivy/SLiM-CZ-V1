import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.proj.NANNGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out, attn


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.NANNGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        # Attention
        attn_out, attn_weights = self.attn(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x, attn_weights


class SLiM_CZ_V1(nn.Module):
    """
    SLiM-CZ-V1: Transformer language model for Czech.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        weight_tying: Tie input/output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        weight_tying: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.weight_tying = weight_tying
        
        # Components
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        if weight_tying:
            self.output.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'NANNGPT_SCALE_INIT'):
                    std *= (2 * self.num_layers) ** -0.5
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Token IDs (batch_size, seq_len)
            mask: Attention mask
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            attention: Attention weights from last layer
        """
        B, T = x.shape
        
        # Create causal mask if none is provided
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.emb_dropout(x)
        
        attention = None
        for block in self.blocks:
            x, attention = block(x, mask)
        
        x = self.norm(x)
        x = self.emb_dropout(x)
        logits = self.output(x)
        
        return logits, attention
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Starting tokens (1, seq_len)
            max_length: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus top-p sampling
            repetition_penalty: Repetition penalty coefficient
            eos_token_id: Optional ID of EndOfSequence token to stop early
        
        Returns:
            Generated token IDs
        """
        was_training = self.training
        self.eval()
        device = start_tokens.device
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.forward(start_tokens[:, -self.max_seq_len:])
                logits = logits[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(start_tokens[0].tolist()):
                        val = logits[0, token_id]
                        logits[0, token_id] = val * repetition_penalty if val < 0 else val / repetition_penalty
                        
                logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_actual = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_actual)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                    
                # Top-p nucleus sampling
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
                    
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                start_tokens = torch.cat([start_tokens, next_token], dim=1)
                
                # EOS stop check
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        
        if was_training:
            self.train()
            
        return start_tokens
    
    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        saved = self.vocab_size * self.d_model if self.weight_tying else 0
        
        return {
            'total': total,
            'trainable': trainable,
            'saved': saved
        }


if __name__ == "__main__":
    # Test
    model = SLiM_CZ_V1(vocab_size=16000)
    
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")
    print(f"Saved by tying: {params['saved']:,}")
    
    x = torch.randint(0, 16000, (2, 128))
    logits, attn = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {logits.shape}")
