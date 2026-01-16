"""
Training script
Version: 0.1.0
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from model import SLiM_CZ_V1
from dataloader import load_preprocessed_data, create_dataloaders


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        
        log_probs = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return (-true_dist * log_probs).sum(dim=-1).mean()


class Trainer:
    """Training manager."""
    
    def __init__(self, model, train_loader, val_loader, config, device, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        train_cfg = config['train']
        self.epochs = train_cfg['epochs']
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)
        self.patience = train_cfg.get('patience', 5)
        
        # Loss
        label_smoothing = train_cfg.get('label_smoothing', 0.0)
        self.criterion = LabelSmoothingLoss(smoothing=label_smoothing) if label_smoothing > 0 else nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=train_cfg.get('warmup_steps', 500),
            T_mult=2
        )
        
        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD and train_cfg.get('use_tensorboard', False):
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch}") if HAS_TQDM else self.train_loader
        
        for input_ids, labels in iterator:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            if HAS_TQDM:
                iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        
        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop."""
        print(f"\nTraining for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}\n")
        
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(lr)
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('PPL/train', train_ppl, epoch)
                self.writer.add_scalar('PPL/val', val_ppl, epoch)
                self.writer.add_scalar('LR', lr, epoch)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{self.epochs} ({elapsed:.1f}s)")
            print(f"  Train: loss={train_loss:.4f} ppl={train_ppl:.2f}")
            print(f"  Val:   loss={val_loss:.4f} ppl={val_ppl:.2f}")
            print(f"  LR: {lr:.2e}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping after {epoch} epochs")
                    break
            
            print()
        
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        if self.writer:
            self.writer.close()
        
        print(f"Training complete")
        print(f"Best val loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        """Save training history."""
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def load_config(config_path, args):
    """Load config from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
    if args.dropout:
        config['model']['dropout'] = args.dropout
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SLiM-CZ-V1')
    
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--dropout', type=float, help='Override dropout')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    train_seqs, val_seqs, stats = load_preprocessed_data(args.data_dir)
    
    print(f"  Train: {len(train_seqs):,} sequences")
    print(f"  Val: {len(val_seqs):,} sequences")
    print(f"  Vocab: {stats['vocab_size']:,}")
    print(f"  Seq len: {stats['seq_len']}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_seqs,
        val_seqs,
        batch_size=config['train']['batch_size'],
        pin_memory=torch.cuda.is_available()
    )
    
    # Device
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create model
    print(f"\nCreating model")
    model = SLiM_CZ_V1(
        vocab_size=stats['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=stats['seq_len'],
        dropout=config['model']['dropout'],
        weight_tying=config['model'].get('weight_tying', True)
    ).to(device)
    
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,}")
    if model.weight_tying:
        print(f"  Saved by tying: {params['saved']:,}")
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")
        trainer.save_checkpoint('interrupted.pt')


if __name__ == "__main__":
    main()
