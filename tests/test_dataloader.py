import torch
import pytest
import numpy as np
from pathlib import Path
import tempfile
from slim_cz_v1.dataloader import LanguageModelDataset, MemmapDataset

def test_language_model_dataset():
    """Test the basic LanguageModelDataset."""
    sequences = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ]
    dataset = LanguageModelDataset(sequences)
    
    assert len(dataset) == 2
    
    input_ids, labels = dataset[0]
    # input is [1, 2, 3, 4], label is [2, 3, 4, 5]
    assert torch.equal(input_ids, torch.tensor([1, 2, 3, 4], dtype=torch.long))
    assert torch.equal(labels, torch.tensor([2, 3, 4, 5], dtype=torch.long))

def test_memmap_dataset():
    """Test the MemmapDataset with a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        # Create 100 random tokens
        data = np.random.randint(0, 1000, 101, dtype=np.uint16)
        data.tofile(tmp.name)
        tmp_path = Path(tmp.name)
    
    try:
        seq_len = 10
        dataset = MemmapDataset(tmp_path, seq_len=seq_len)
        
        # 100 // 10 = 10 sequences
        assert len(dataset) == 10
        
        input_ids, labels = dataset[0]
        assert len(input_ids) == seq_len
        assert len(labels) == seq_len
        
        # Check shifting
        assert torch.equal(input_ids[1:], labels[:-1])
        
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
