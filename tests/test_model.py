import torch
import pytest
from slim_cz_v1.model import SLiM_CZ_V1

def test_model_initialization():
    """Test if the model can be initialized with default parameters."""
    vocab_size = 1000
    model = SLiM_CZ_V1(vocab_size=vocab_size)
    assert model is not None
    assert isinstance(model, SLiM_CZ_V1)

def test_model_forward_pass():
    """Test a forward pass with dummy data."""
    vocab_size = 1000
    d_model = 128
    batch_size = 2
    seq_len = 32
    
    model = SLiM_CZ_V1(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2
    )
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, _ = model(x)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)

def test_model_parameter_count():
    """Test if parameter count returns expected structure."""
    model = SLiM_CZ_V1(vocab_size=1000)
    params = model.count_parameters()
    
    assert "total" in params
    assert "trainable" in params
    assert "saved" in params
    assert isinstance(params["total"], int)
