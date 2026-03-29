"""
Training infrastructure for SLiM-CZ-V1.

This module provides core training functionality including:
- Trainer class for model training
- Data loading utilities
- TensorBoard logging
- Checkpoint management

Usage (Programmatic):
    from slim_cz_v1.training import Trainer

    trainer = Trainer(config, output_dir)
    for progress in trainer.train(tokens_path, seq_len, vocab_size):
        print(f"Epoch {progress['epoch']}: PPL={progress['val_perplexity']:.2f}")

Usage (CLI):
    slim-train --config config.yaml --tokens tokens.txt --output models/
"""

from .train import (
    SequenceDataset,
    TensorBoardLogger,
    Trainer,
    create_dataloaders,
    load_tokenized_data,
)

__all__ = [
    "SequenceDataset",
    "TensorBoardLogger",
    "Trainer",
    "create_dataloaders",
    "load_tokenized_data",
]
