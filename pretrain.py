"""
WirelessJEPA Pretraining Module
Implements complete ALgorithm 1 training loop with loss functions and optimization
Based on: "WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"
Reference: Paper Section 4, Algorithm 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time
import json
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mask import MaskGenerator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class JEPALoss(nn.Module):
    """
    JEPA L2 regression loss for masked position prediction.
    
    Paper Reference: Section 4 - Loss function
    "L = 1/|M| * ∑_{(i,j)∈M} ||ŷ_{i,j} - y_{i,j}||₂²"
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
               mask: torch.Tensor) -> torch.Tensor:
        """
        Compute JEPA loss only on masked positions.
        
        Args:
            predicted: Predicted features (B, C, H, W)
            target: Target features (B, C, H, W)
            mask: Mask tensor (B, 1, H, W) or (B, H, W), 0=masked, 1=visible
        
        Returns:
            Scalar loss value
        """
        # Ensure mask is proper shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Invert mask: we want to compute loss only on masked regions (0 → 1)
        loss_mask = 1.0 - mask
        
        # Compute L2 loss
        diff = predicted - target
        loss = (diff ** 2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Apply mask and compute mean over masked positions
        masked_loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        
        return masked_loss


class PretrainingTrainer:
    """
    Complete pretraining trainer implementing Algorithm 1 from the paper.
    """
    
    def __init__(self, model, config, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: WirelessJEPA model instance
            config: Configuration object
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Loss and optimizer
        self.loss_fn = JEPALoss().to(device)
        self.mask_gen = MaskGenerator(
            height=config.mask.height,
            width=config.mask.width,
            patch_height=config.mask.patch_height,
            patch_width=config.mask.patch_width,
            seed=config.mask.random_seed
        )
        
        # Setup optimizer (Paper: AdamW)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas,
            eps=config.training.eps
        )
        
        # Setup learning rate scheduler (Paper: warmup + cosine annealing)
        warmup_epochs = min(config.training.warmup_epochs, max(0, config.training.num_epochs - 1))
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, config.training.num_epochs - warmup_epochs),
                eta_min=config.training.min_lr
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, config.training.num_epochs),
                eta_min=config.training.min_lr
            )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.ema_momentum_schedule = []
        self.current_step = 0
        self.total_steps = None
        
        # Create output directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def compute_ema_momentum(self) -> float:
        """
        Compute EMA momentum for the current optimizer step.
        
        Paper Algorithm 1: "Momentum τ linearly increases from 0.996 to 1.0"
        """
        init_tau = self.config.model.ema_momentum
        final_tau = self.config.model.ema_momentum_final
        
        total_steps = self.total_steps or max(1, self.config.training.num_epochs)
        progress = min(self.current_step / max(1, total_steps), 1.0)
        tau = init_tau + (final_tau - init_tau) * progress
        self.current_step += 1
        return min(tau, final_tau)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Paper Algorithm 1: Complete training step
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (signals, labels) in enumerate(pbar):
            signals = signals.to(self.device)  # (B, 2, 256, 256)
            
            # Generate masks
            masks = []
            masks_latent = []
            
            for _ in range(signals.size(0)):
                mask = self.mask_gen.generate_mask(
                    mask_type=self.config.mask.mask_type,
                    mask_ratio=self.config.mask.mask_ratio
                )
                mask_latent = self.mask_gen.get_latent_mask(mask, factor=16)
                masks.append(mask)
                masks_latent.append(mask_latent)
            
            mask = torch.stack(masks).to(self.device)  # (B, 256, 256)
            mask_latent = torch.stack(masks_latent).to(self.device)  # (B, 16, 16)
            
            # Create masked input (paper: zeros at masked positions)
            x_masked = signals * mask.unsqueeze(1)
            
            # Forward pass
            output = self.model(x_masked, signals, mask, mask_latent)
            
            # Compute loss (only on masked positions)
            loss = self.loss_fn(
                output['predicted_features'],
                output['target_features'],
                mask_latent
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip_norm
                )
            
            self.optimizer.step()
            
            # Update EMA teacher
            tau = self.compute_ema_momentum()
            self.model.update_teacher(tau=tau)
            self.ema_momentum_schedule.append(tau)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log interval
            if (batch_idx + 1) % self.config.training.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validation pass.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for signals, labels in pbar:
            signals = signals.to(self.device)
            
            # Generate masks
            masks = []
            masks_latent = []
            
            for _ in range(signals.size(0)):
                mask = self.mask_gen.generate_mask(
                    mask_type=self.config.mask.mask_type,
                    mask_ratio=self.config.mask.mask_ratio
                )
                mask_latent = self.mask_gen.get_latent_mask(mask, factor=16)
                masks.append(mask)
                masks_latent.append(mask_latent)
            
            mask = torch.stack(masks).to(self.device)
            mask_latent = torch.stack(masks_latent).to(self.device)
            
            x_masked = signals * mask.unsqueeze(1)
            
            # Forward pass
            output = self.model(x_masked, signals, mask, mask_latent)
            
            # Compute loss
            loss = self.loss_fn(
                output['predicted_features'],
                output['target_features'],
                mask_latent
            )
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = self.config.get_checkpoint_path(epoch)
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.config.get_best_checkpoint_path()
            torch.save(checkpoint, best_path)
            print(f"✓ Best checkpoint saved: {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Complete training loop for specified number of epochs.
        
        Paper Algorithm 1: Full training procedure
        """
        print(f"Starting pretraining for {self.config.training.num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        best_val_loss = float('inf')
        self.total_steps = self.config.training.num_epochs * max(1, len(train_loader))
        
        for epoch in range(self.config.training.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader is not None and (epoch + 1) % self.config.training.val_interval == 0:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(epoch)
        
        print("✓ Pretraining completed!")
        self.plot_losses()
        
        return self.train_losses, self.val_losses
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', marker='s')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('WirelessJEPA Pretraining Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        loss_plot_path = Path(self.config.output_dir) / 'training_losses.png'
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Loss plot saved: {loss_plot_path}")
        plt.close()


def pretrain_wirelessjepa(config, train_loader, val_loader=None):
    """
    High-level function to run pretraining.
    
    Args:
        config: Configuration object
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
    
    Returns:
        Trained model
    """
    from model import create_model
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    trainer = PretrainingTrainer(model, config, device=device)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    return model, trainer


if __name__ == "__main__":
    print("Pretraining module test")
    print("This module should be called from main training scripts")
