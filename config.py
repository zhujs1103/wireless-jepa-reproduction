"""
WirelessJEPA Configuration Module
Configuration management for WirelessJEPA pretraining and evaluation
Based on: "WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"
Reference: Section 4 (Experimental Setup), Table II
"""

import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration (Paper Section 3.1)"""
    # Dataset paths
    data_root: str = "./data"
    dataset_name: str = "RML2016.10a"
    
    # Input signal format (Paper: x ∈ R^(C×H×W))
    num_channels: int = 2  # I/Q components
    num_antennas: int = 4  # H: antenna dimension
    num_time_samples: int = 256  # W: time dimension
    
    # For RML2016.10a single-channel compatibility (2×1×128)
    rml_num_time_samples: int = 128
    rml_num_antennas: int = 1
    
    # Upsampling configuration (Paper: antenna-time grid upsampling)
    target_antenna_dim: int = 256  # Upsample to square grid (256×256)
    target_time_dim: int = 256
    upsample_ratio: float = 64.0  # Nearest-neighbor interp ratio for antenna dim
    
    # Normalization: unitmax (Paper Section 3.1)
    normalize_type: str = "unitmax"  # x ← x / max(|x|)
    
    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Few-shot settings (Paper: Table I, few-shot scenarios)
    few_shot_samples: Dict[str, int] = field(default_factory=lambda: {
        "1-shot": 1,
        "100-shot": 100,
        "500-shot": 500,
        "full": -1
    })
    
    # SNR conditions
    snr_conditions: List[str] = field(default_factory=lambda: [
        "-20dB", "-18dB", "-16dB", "-14dB", "-12dB", 
        "-10dB", "-8dB", "-6dB", "-4dB", "-2dB", "0dB", "2dB"
    ])
    
    # DataLoader settings
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration (Paper Section 3, Algorithm 2)"""
    # Context and teacher encoders (Paper: ShuffleNetV2-x0.5)
    encoder_type: str = "ShuffleNetV2"
    encoder_depth: str = "x0.5"  # Channel multiplier for ShuffleNetV2
    
    # Sparse convolution mechanism (Paper Algorithm 2)
    use_sparse_conv: bool = True
    mask_propagation: bool = True
    
    # Predictor architecture (Paper: Section 3.2, Lightweight depthwise separable conv)
    predictor_depth: int = 3  # 3-layer depthwise separable conv
    predictor_hidden_dim: int = 256
    predictor_use_bn: bool = True
    
    # Learnable mask token (Paper: masked latent token insertion)
    learnable_mask_token: bool = True
    mask_token_dim: int = 256  # Match feature_dim after latent projection
    
    # Feature dimension
    feature_dim: int = 256
    
    # Encoder input
    input_channels: int = 2  # I/Q
    
    # Momentum teacher (Paper: Section 3.3, EMA update)
    ema_momentum: float = 0.996  # Initial τ, grows to 1.0
    ema_momentum_final: float = 1.0
    ema_update_freq: int = 1  # Update every N batches
    
    # Normalization layers
    use_batchnorm: bool = True
    use_layernorm: bool = False
    norm_bias: bool = True
    eps: float = 1e-5


@dataclass
class MaskConfig:
    """Masking strategy configuration (Paper Section 3.2, Figure 2)"""
    # Grid dimensions matching input (256×256 antenna-time plane)
    height: int = 256
    width: int = 256
    
    # Mask types (Paper Figure 2: 4 masking strategies)
    mask_type: str = "random"  # Options: "random", "antenna", "temporal", "multiblock"
    
    # Patch-based masking configuration
    patch_height: int = 64
    patch_width: int = 32
    
    # Masking ratio (Paper: typically 0.75)
    mask_ratio: float = 0.75
    
    # Spatial masking parameters
    antenna_masking_height: int = 64  # Whole row of antennas
    antenna_masking_width: int = 256
    
    # Temporal masking parameters
    temporal_masking_height: int = 256
    temporal_masking_width: int = 32  # Whole column of time
    
    # Multi-block masking
    num_blocks: int = 3
    block_aspect_ratio: float = 2.0  # H/W ratio for blocks
    
    # Operations
    random_seed: int = 42
    allow_overlap: bool = False


@dataclass
class TrainingConfig:
    """Pretraining configuration (Paper Section 4, Algorithm 1)"""
    # Optimizer settings (Paper: AdamW optimizer)
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduler (Paper: Cosine annealing)
    lr_scheduler: str = "cosine"
    num_epochs: int = 100
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Gradient settings
    grad_clip_norm: float = 1.0
    
    # Loss function (Paper: L2 regression loss on masked positions)
    loss_type: str = "l2"  # L = 1/|M| * ∑ ||ŷ - y||₂²
    
    # Training loop configuration
    log_interval: int = 50
    val_interval: int = 5
    save_interval: int = 5
    
    # Device settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Random seeds
    seed: int = 42
    deterministic: bool = True
    
    # Checkpointing
    save_best: bool = True
    resume_from: str = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration (Paper Section 4, Table I, II)"""
    # Evaluation methods
    eval_type: str = "linear_probing"  # Options: "linear_probing", "knn"
    
    # Linear probing settings
    linear_probe_epochs: int = 100
    linear_probe_lr: float = 1e-3
    linear_probe_weight_decay: float = 0.0
    
    # k-NN evaluation
    knn_k: int = 20
    knn_temperature: float = 0.07
    
    # Few-shot evaluation (Paper: 1-shot, 100-shot, 500-shot)
    few_shot_config: Dict[str, int] = field(default_factory=lambda: {
        "1-shot": 1,
        "100-shot": 100,
        "500-shot": 500
    })
    
    # Downstream tasks
    downstream_tasks: List[str] = field(default_factory=lambda: [
        "modulation_classification",
        "aoa_estimation",
        "rf_fingerprinting",
        "gnss_interference",
        "wifi_protocol",
        "5g_nr_interference"
    ])
    
    # Results output
    output_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_score", "confusion_matrix"
    ])
    
    # Visualization
    plot_results: bool = True
    save_plots: bool = True


@dataclass
class AblationConfig:
    """Ablation experiment configuration"""
    # Mask type ablation (Paper: Figure 3, 4)
    ablation_mask_types: List[str] = field(default_factory=lambda: [
        "random", "antenna", "temporal", "multiblock"
    ])
    
    # Run configuration
    num_runs_per_config: int = 3
    save_ablation_results: bool = True
    compare_baselines: bool = True
    
    # Baselines (Paper: IQFM baseline reference)
    baseline_types: List[str] = field(default_factory=lambda: [
        "supervised_baseline",
        "iqfm_baseline",
        "wirelessjepa"
    ])


@dataclass
class WirelessJEPAConfig:
    """Complete WirelessJEPA configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    
    # General settings
    project_name: str = "WirelessJEPA"
    experiment_name: str = "default"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    def create_directories(self):
        """Create necessary output directories"""
        for dir_path in [self.output_dir, self.log_dir, self.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, epoch: int) -> str:
        """Get checkpoint path for given epoch"""
        return os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pt")
    
    def get_best_checkpoint_path(self) -> str:
        """Get best model checkpoint path"""
        return os.path.join(self.checkpoint_dir, "best_model.pt")


def get_default_config() -> WirelessJEPAConfig:
    """Get default configuration matching paper settings"""
    return WirelessJEPAConfig()


def load_config_from_dict(config_dict: Dict) -> WirelessJEPAConfig:
    """Load configuration from dictionary (for command-line args)"""
    config = get_default_config()
    
    # Recursively update nested dataclass fields
    for key, value in config_dict.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            else:
                setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Default WirelessJEPA Configuration:")
    print(f"Data Config: batch_size={config.data.batch_size}, target_dim=(256, 256)")
    print(f"Model Config: encoder={config.model.encoder_type}, ema_momentum={config.model.ema_momentum}")
    print(f"Mask Config: type={config.mask.mask_type}, ratio={config.mask.mask_ratio}")
    print(f"Training Config: epochs={config.training.num_epochs}, lr={config.training.learning_rate}")
    print(f"Evaluation Config: type={config.evaluation.eval_type}")
    
    config.create_directories()
    print("✓ Configuration validated and directories created")
