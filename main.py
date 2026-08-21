"""
WirelessJEPA Main Entry Point
Command-line interface for pretraining, evaluation, and ablation experiments

Usage:
    python main.py --mode pretrain --config_file config.yaml
    python main.py --mode eval --checkpoint best_model.pt
    python main.py --mode ablation --mask_type random
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config import WirelessJEPAConfig, get_default_config
from dataset import create_dataloader, WirelessJEPADataLoader
from model import create_model
from pretrain import PretrainingTrainer
from eval import DownstreamEvaluator
from mask import MaskGenerator, visualize_masks


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state'])
    
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✓ Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
    
    return epoch


def pretrain_mode(config: WirelessJEPAConfig):
    """Run pretraining"""
    print("\n" + "="*60)
    print("WirelessJEPA PRETRAINING MODE")
    print("="*60)
    
    setup_seed(config.training.seed)
    config.create_directories()
    
    # Create device
    device = torch.device(
        config.training.device if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading dataset from {config.data.data_root}...")
    try:
        train_loader, val_loader, _ = create_dataloader(
            data_path=f"{config.data.data_root}/RML2016.10a_dict.pkl",
            batch_size=config.data.batch_size,
            few_shot=None,
            num_workers=config.data.num_workers
        )
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"Please ensure RML2016.10a_dict.pkl is in {config.data.data_root}/")
        return
    
    # Create model
    print("\nCreating WirelessJEPA model...")
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params/1e6:.2f}M parameters")
    
    # Move to device
    model = model.to(device)
    
    # Create trainer
    trainer = PretrainingTrainer(model, config, device=device)
    
    # Resume from checkpoint if specified
    if config.training.resume_from:
        load_checkpoint(config.training.resume_from, model, trainer.optimizer, trainer.scheduler)
    
    # Train
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    print("\n✓ Pretraining completed!")
    print(f"Best checkpoint: {config.get_best_checkpoint_path()}")


def eval_mode(config: WirelessJEPAConfig, checkpoint_path: Optional[str] = None):
    """Run evaluation"""
    print("\n" + "="*60)
    print("WirelessJEPA EVALUATION MODE")
    print("="*60)
    
    setup_seed(config.training.seed)
    
    # Device
    device = torch.device(
        config.training.device if torch.cuda.is_available() else 'cpu'
    )
    
    # Load data
    print(f"\nLoading dataset...")
    try:
        train_loader, val_loader, test_loader = create_dataloader(
            data_path=f"{config.data.data_root}/RML2016.10a_dict.pkl",
            batch_size=config.data.batch_size,
            few_shot=None,
            num_workers=config.data.num_workers
        )
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    # Load model
    print("\nCreating model...")
    model = create_model(config).to(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.get_best_checkpoint_path()
    
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"✓ Loaded pretrained model: {checkpoint_path}")
    
    # Evaluate
    data_loader = WirelessJEPADataLoader(
        f"{config.data.data_root}/RML2016.10a_dict.pkl",
        batch_size=config.data.batch_size
    )
    
    evaluator = DownstreamEvaluator(model, config, device=device)
    results = evaluator.evaluate(
        train_loader, test_loader,
        num_classes=len(data_loader.get_modulation_list()),
        modulation_list=data_loader.get_modulation_list()
    )
    
    print("\n✓ Evaluation completed!")


def ablation_mode(config: WirelessJEPAConfig, mask_type: Optional[str] = None):
    """Run ablation experiments"""
    print("\n" + "="*60)
    print("WirelessJEPA ABLATION EXPERIMENTS")
    print("="*60)
    
    ablation_results = {}
    
    # Test each mask type
    mask_types = [mask_type] if mask_type else config.ablation.ablation_mask_types
    
    for mask_t in mask_types:
        print(f"\n{'='*60}")
        print(f"Testing mask type: {mask_t}")
        print(f"{'='*60}")
        
        # Update config
        config.mask.mask_type = mask_t
        
        # Run pretraining (short version for ablation)
        _config = WirelessJEPAConfig()
        _config.mask.mask_type = mask_t
        _config.training.num_epochs = 5  # Reduced epochs for ablation
        
        print(f"Pretraining with {mask_t} masking...")
        try:
            pretrain_mode(_config)
            ablation_results[mask_t] = "Completed"
        except Exception as e:
            print(f"✗ Error: {e}")
            ablation_results[mask_t] = f"Failed: {str(e)}"
    
    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for mask_t, result in ablation_results.items():
        print(f"{mask_t}: {result}")


def demo_mode():
    """Run demo with synthetic data"""
    print("\n" + "="*60)
    print("WirelessJEPA DEMO MODE")
    print("="*60)
    
    print("\nGenerating synthetic data...")
    
    # Create config
    config = get_default_config()
    config.training.num_epochs = 2
    config.create_directories()
    
    # Create synthetic data loader
    from torch.utils.data import TensorDataset, DataLoader as PyTorchDataLoader
    
    # Synthetic signals
    X = torch.randn(128, 2, 256, 256)  # 128 samples
    y = torch.randint(0, 11, (128,))  # 11 modulation classes
    
    dataset = TensorDataset(X, y)
    train_loader = PyTorchDataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = PyTorchDataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create and train model
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config).to(device)
    
    print("Training...")
    trainer = PretrainingTrainer(model, config, device=device)
    trainer.train(train_loader, val_loader)
    
    print("✓ Demo completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="WirelessJEPA: Multi-Antenna Foundation Model for Wireless Signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pretraining
  python main.py --mode pretrain
  
  # Evaluation
  python main.py --mode eval
  
  # Ablation studies
  python main.py --mode ablation --mask_type random
  
  # Demo with synthetic data
  python main.py --mode demo
  
  # Visualize masking strategies
  python main.py --mode visualize_masks
        """
    )
    
    parser.add_argument('--mode', type=str, default='pretrain',
                       choices=['pretrain', 'eval', 'ablation', 'demo', 'visualize_masks'],
                       help='Mode of operation')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--mask_type', type=str, default=None,
                       choices=['random', 'antenna', 'temporal', 'multiblock'],
                       help='Mask type for ablation')
    parser.add_argument('--data_path', type=str, default='./data/RML2016.10a_dict.pkl',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    setup_seed(args.seed)
    
    # Load/create configuration
    config = get_default_config()
    config.data.data_root = str(Path(args.data_path).parent)
    config.output_dir = args.output_dir
    config.data.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.seed = args.seed
    
    # Run mode
    if args.mode == 'pretrain':
        if args.mask_type:
            config.mask.mask_type = args.mask_type
        pretrain_mode(config)
    elif args.mode == 'eval':
        eval_mode(config, args.checkpoint)
    elif args.mode == 'ablation':
        ablation_mode(config, args.mask_type)
    elif args.mode == 'demo':
        demo_mode()
    elif args.mode == 'visualize_masks':
        print("Generating mask visualizations...")
        visualize_masks(save_path='./mask_visualization.png')
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
