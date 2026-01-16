"""
SLiM-CZ-V1 Model
Version: 0.1.0

Just the model architecture, nothing else.
"""

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
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.proj(out)
        
        return out, attn


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
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
                nn.init.normal_(module.weight, std=0.02)
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
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.emb_dropout(x)
        
        attention = None
        for block in self.blocks:
            x, attention = block(x, mask)
        
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, attention
    
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Starting tokens (1, seq_len)
            max_length: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            Generated token IDs
        """
        self.eval()
        device = start_tokens.device
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.forward(start_tokens[:, -self.max_seq_len:])
                logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_idx)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                start_tokens = torch.cat([start_tokens, next_token], dim=1)
        
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
