"""
WirelessJEPA Utility Functions
Common utilities for data processing, training, and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_parameters(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze model parameters"""
    for param in model.parameters():
        param.requires_grad = not freeze


def get_parameter_groups(model: nn.Module, weight_decay: float = 0.0) -> List[Dict]:
    """
    Group parameters for differential learning rates or weight decay.
    
    Useful for applying weight decay to conv/linear weights but not biases/norms.
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias and normalization layers
        if 'bias' in name or 'norm' in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                   save_path: str, extras: Optional[Dict] = None):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
    }
    
    if extras:
        checkpoint.update(extras)
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                   optimizer=None, scheduler=None, device='cpu') -> int:
    """
    Load checkpoint and return epoch.
    
    Returns:
        Starting epoch for resuming training
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    if scheduler and checkpoint.get('scheduler_state'):
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✓ Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
    
    return epoch


class AverageMeter:
    """Track running averages"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'


class ExperimentTracker:
    """Track experiment metrics and metadata"""
    
    def __init__(self, exp_name: str, config_dict: Optional[Dict] = None):
        self.exp_name = exp_name
        self.timestamp = datetime.now().isoformat()
        self.config = config_dict or {}
        self.metrics = {}
    
    def add_metric(self, phase: str, epoch: int, name: str, value: float):
        """Add metric value"""
        if phase not in self.metrics:
            self.metrics[phase] = {}
        
        if epoch not in self.metrics[phase]:
            self.metrics[phase][epoch] = {}
        
        self.metrics[phase][epoch][name] = value
    
    def save(self, save_path: str):
        """Save experiment tracking data as JSON"""
        data = {
            'experiment': self.exp_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'metrics': self.metrics
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Experiment tracking saved: {save_path}")


def get_device(use_cuda: bool = True) -> torch.device:
    """Get appropriate device"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def set_reproducibility(seed: int = 42):
    """Set reproducible random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_directories(paths: Dict[str, str]):
    """Create necessary directories"""
    for name, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {name}: {path}")


def estimate_model_params_flops(model: nn.Module, input_size: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Estimate model parameters and FLOPs.
    
    Note: This is a rough estimate; exact computation requires specialized tools.
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Approximate FLOPs (heuristic)
    total_flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # FLOPs = (H * W * Cin * Kh * Kw) * 2 (multiply-add)
            output_size = m.out_channels
            kernel_ops = m.kernel_size[0] * m.kernel_size[1]
            input_ops = total_params * kernel_ops / m.out_channels
            total_flops += input_ops * 2
    
    return total_params, total_flops


def summary_model(model: nn.Module, input_size: Tuple[int, ...]):
    """Print model summary"""
    total, trainable = count_parameters(model)
    
    print("="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"Input size: {input_size}")
    print("="*60)


if __name__ == "__main__":
    print("Utility functions module")
    print("Import as: from utils import *")
