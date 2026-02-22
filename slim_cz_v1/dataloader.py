import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LanguageModelDataset(Dataset):
    """Dataset for autoregressive language modeling."""
    
    def __init__(self, sequences: List[List[int]]):
        """
        Args:
            sequences: List of tokenized sequences
        """
        self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training example.
        
        Returns:
            input_ids: Input tokens (seq_len)
            labels: Target tokens (seq_len) - shifted by 1
        """
        seq = self.sequences[idx]
        
        # Autoregressive: input[t] predicts label[t+1]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_ids, labels


class MemmapDataset(Dataset):
    """Memory-mapped dataset for large-scale training."""

    def __init__(self, bin_path: Path, seq_len: int, dtype=np.uint16):
        """
        Args:
            bin_path: Path to binary token file
            seq_len: Sequence length
            dtype: Data type of tokens (default: uint16)
        """
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.dtype = dtype

        # Memory map the file
        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode='r')
        
        # Calculate number of sequences
        # Each sequence needs seq_len + 1 tokens for input/label shift
        self.num_sequences = len(self.data) // seq_len
        
        # Trim data to exact multiple of seq_len
        self.data = self.data[:self.num_sequences * seq_len]

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training example.
        
        Returns:
            input_ids: Input tokens (seq_len)
            labels: Target tokens (seq_len) - shifted by 1
        """
        # For language modeling, we usually want to predict next token.
        # If we have a flat stream of tokens, we can take a window.
        
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Ensure we don't go out of bounds for labels (which needs +1)
        if end_idx >= len(self.data):
             # Wrap around or handle end
             start_idx = 0
             end_idx = self.seq_len
             
        chunk = self.data[start_idx:end_idx+1]
        
        # If we don't have enough for labels, pad or wrap
        if len(chunk) < self.seq_len + 1:
            chunk = np.concatenate([chunk, self.data[:self.seq_len + 1 - len(chunk)]])

        input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
        labels = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return input_ids, labels


def load_preprocessed_data(data_dir: str) -> Tuple[List, List, Dict]:
    """
    Load preprocessed sequences and stats.
    
    Args:
        data_dir: Directory with train.json, val.json, stats.json
    
    Returns:
        train_sequences: Training sequences
        val_sequences: Validation sequences
        stats: Dataset statistics
    """
    data_path = Path(data_dir)
    
    with open(data_path / 'train.json', 'r') as f:
        train_sequences = json.load(f)
    
    with open(data_path / 'val.json', 'r') as f:
        val_sequences = json.load(f)
    
    with open(data_path / 'stats.json', 'r') as f:
        stats = json.load(f)
    
    return train_sequences, val_sequences, stats


def create_dataloaders(
    train_sequences: List[List[int]],
    val_sequences: List[List[int]],
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_sequences: Training sequences
        val_sequences: Validation sequences
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    train_dataset = LanguageModelDataset(train_sequences)
    val_dataset = LanguageModelDataset(val_sequences)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataloader.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print(f"Loading from {data_dir}")
    train_seqs, val_seqs, stats = load_preprocessed_data(data_dir)
    
    print(f"Train: {len(train_seqs):,}")
    print(f"Val: {len(val_seqs):,}")
    print(f"Vocab: {stats['vocab_size']:,}")
    print(f"Seq len: {stats['seq_len']}")
    
    train_loader, val_loader = create_dataloaders(train_seqs, val_seqs, batch_size=32)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Sample batch
    input_ids, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Labels: {labels.shape}")
