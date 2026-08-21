# WirelessJEPA: Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions

Complete PyTorch implementation of the paper:
**"WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"**

This repository provides a full reproduction of the WirelessJEPA architecture with all core components, training pipeline, and downstream task evaluation.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Model Architecture](#model-architecture)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

## ✨ Features

### Core Implementation
- ✅ **Complete WirelessJEPA Architecture** - ShuffleNetV2-x0.5 encoder with sparse convolution
- ✅ **4 Masking Strategies** - Random, Antenna, Temporal, Multi-block masking
- ✅ **JEPA Training** - Complete Algorithm 1 implementation with EMA teacher
- ✅ **Downstream Evaluation** - Linear probing and k-NN classification
- ✅ **Few-shot Support** - 1-shot, 100-shot, 500-shot scenarios

### Data & Preprocessing
- ✅ **RML2016.10a Support** - Load and preprocess wireless signals
- ✅ **IQ Signal Handling** - Complex to real conversion and normalization
- ✅ **Antenna-Time Upsampling** - Nearest-neighbor interpolation to 256×256 grid

### Training & Evaluation
- ✅ **AdamW Optimizer** - With cosine learning rate scheduling
- ✅ **EMA Teacher Update** - Momentum coefficient 0.996→1.0
- ✅ **Ablation Experiments** - Compare different masking strategies
- ✅ **Result Visualization** - Confusion matrices and performance plots

## 🔧 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone or download the project
cd wirelessJEPA

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn pandas tqdm

# Prepare data - place RML2016.10a_dict.pkl in ./data/
mkdir -p data
# Copy your RML2016.10a_dict.pkl to ./data/
```

## 📁 Project Structure

```
wirelessJEPA/
├── config.py              # Configuration management (all hyperparameters)
├── dataset.py             # Data loading and preprocessing (Stage 1)
├── mask.py                # Masking strategies (Stage 3)
├── model.py               # WirelessJEPA architecture (Stage 2)
├── pretrain.py            # Training loop (Stage 4)
├── eval.py                # Downstream evaluation (Stage 5)
├── utils.py               # Utility functions
├── main.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── REPRODUCTION_REPORT.md # Experiment record and limitations
└── data/
    ├── RML2016.10a_dict.pkl    # Dataset (provided by user)
    └── LICENSE.TXT              # Data license
```

## 🚀 Quick Start

### 1. Pretraining

Train the model with default configuration:

```bash
python main.py --mode pretrain --epochs 100 --batch_size 64
```

Check `outputs/training_losses.png` for loss curves.

### 2. Evaluation

Evaluate pretrained model on downstream tasks:

```bash
python main.py --mode eval --checkpoint checkpoints/best_model.pt
```

Results include:
- Linear probing accuracy
- k-NN classification accuracy (k=20)
- Confusion matrix visualization

### 3. Ablation Experiments

Compare different masking strategies:

```bash
# Test specific mask type
python main.py --mode ablation --mask_type random

# Test all mask types
python main.py --mode ablation
```

### 4. Demo

Run demo with synthetic data:

```bash
python main.py --mode demo
```

### 5. Visualize Masking Strategies

Generate Figure 2 from the paper:

```bash
python main.py --mode visualize_masks
```

Creates `mask_visualization.png` showing all 4 masking strategies.

## 📖 Detailed Usage

### Configuration

Edit `config.py` to customize hyperparameters:

```python
from config import WirelessJEPAConfig

config = WirelessJEPAConfig()

# Data configuration
config.data.batch_size = 64
config.data.normalize_type = "unitmax"

# Model configuration
config.model.input_channels = 2
config.model.feature_dim = 256

# Training configuration
config.training.learning_rate = 1e-4
config.training.num_epochs = 100
config.training.optimizer = "adamw"

# Masking configuration
config.mask.mask_type = "random"  # or "antenna", "temporal", "multiblock"
config.mask.mask_ratio = 0.75
```

### Pretraining from Code

```python
from config import get_default_config
from dataset import create_dataloader
from model import create_model
from pretrain import pretrain_wirelessjepa

# Load configuration
config = get_default_config()

# Create data loaders
train_loader, val_loader, test_loader = create_dataloader(
    data_path="./data/RML2016.10a_dict.pkl",
    batch_size=64
)

# Pretrain
model, trainer = pretrain_wirelessjepa(config, train_loader, val_loader)
```

### Evaluation from Code

```python
from config import get_default_config
from dataset import WirelessJEPADataLoader, create_dataloader
from model import create_model
from eval import evaluate_pretrained_model
import torch

# Load configuration and data
config = get_default_config()
_, val_loader, test_loader = create_dataloader(
    data_path="./data/RML2016.10a_dict.pkl"
)

# Load pretrained model
model = create_model(config)
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state'])

# Get modulation list
data_loader = WirelessJEPADataLoader("./data/RML2016.10a_dict.pkl")

# Evaluate
results = evaluate_pretrained_model(
    model, config, val_loader, test_loader,
    num_classes=len(data_loader.get_modulation_list()),
    modulation_list=data_loader.get_modulation_list()
)

print(results)
```

## 🏗️ Model Architecture

### Encoder: ShuffleNetV2-x0.5

- Input: (B, 2, 256, 256) - 2 channel IQ signals
- 4 stages of ShuffleNetV2 blocks
- Depthwise separable convolutions
- Output feature dimension: 1024 → 256 (projected)

### Sparse Convolution Mechanism

- Mask propagation through layers (Algorithm 2)
- Mask-aware BatchNorm on unmasked regions
- Learnable mask tokens inserted at masked positions

### Predictor: 3-layer Depthwise Separable Conv

- Input: Context features (B, 256, H, W)
- 3 depthwise separable convolutional layers
- Output: Predicted features for masked positions

### Teacher Encoder: Momentum Updated

- Same architecture as student encoder
- Parameters updated via EMA: θ'_t ← τ·θ'_t + (1-τ)·θ_t
- Momentum τ linearly increases from 0.996 to 1.0

## 📊 Experimental Results

### Paper Benchmarks (Table I - RML2016.10a)

| Scenario | Method | Accuracy |
|----------|--------|----------|
| Full | WirelessJEPA | 91.2% |
| Full | IQFM Baseline | 87.5% |
| 500-shot | WirelessJEPA | 88.3% |
| 100-shot | WirelessJEPA | 78.9% |
| 1-shot | WirelessJEPA | 45.2% |

### Masking Strategy Comparison (Figure 3)

- Random Masking: Baseline CNN-JEPA approach
- Antenna Masking: ↑ Learn spatial antenna dependencies
- Temporal Masking: ↑ Learn temporal correlations
- Multi-block: Balanced spatial-temporal inductive bias

## 🔬 Masking Strategies (Paper Section 3.2, Figure 2)

### 1. Random Masking
- Patch-based random masking (64×32 patches)
- Masking ratio: 75%
- Baseline CNN-JEPA approach

### 2. Antenna Masking
- Mask entire antenna rows (64×256)
- Forces learning of spatial dependencies across antennas
- Specialized for multi-antenna wireless systems

### 3. Temporal Masking
- Mask entire time columns (256×32)
- Forces learning of temporal correlations
- Captures signal dynamics over time

### 4. Multi-block Masking
- Multiple continuous spatio-temporal blocks
- Balanced spatial-temporal inductive bias
- Adaptive block sizes

## 📝 Paper Details

### Input Signal Format (Section 3.1)
- Complex baseband IQ signals → real-valued tensors
- Format: x ∈ ℝ^(C×H×W) where C=2 (I/Q), H=antenna, W=time
- Input: (2, 4, 256) → Upsampled: (2, 256, 256)
- Normalization: Unitmax x ← x/max(|x|)

### Training Algorithm (Algorithm 1)
1. Input upsampling to 256×256
2. Generate context and target masks
3. Encode context with masking
4. Insert learnable mask tokens
5. Predict masked latents with predictor
6. Encode full representation with teacher
7. Compute JEPA L2 loss at masked positions
8. Update student and EMA teacher

### Loss Function
L = 1/|M| * Σ_(i,j)∈M ||ŷ_(i,j) - y_(i,j)||²_2

Loss computed only on masked positions M.

## 🎓 Citation

```bibtex
@article{wirelessjepa2024,
  title={WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions},
  author={[Author Information]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## 📄 License

The original code in this repository is released under the MIT License. The
referenced paper, datasets and any downloaded third-party assets remain subject
to their respective terms.

## 🤝 Contributing

For issues, questions, or improvements, please open an issue on the repository.

## ✅ Verification Checklist

- [x] Core source files pass syntax checks
- [x] Main algorithms, parameters and architecture are documented against the paper
- [x] Paper source citations in comments
- [x] Supports RML2016.10a dataset
- [x] 4 masking strategies implemented
- [x] Training + evaluation pipeline complete
- [x] Few-shot support (1/100/500-shot)
- [x] Ablation experiment framework
- [x] Result visualization and metrics

## 📞 Support

For questions or issues:
1. Check `REPRODUCTION_REPORT.md` for the verified setup and known limitations
2. Verify data is in `./data/RML2016.10a_dict.pkl`
3. Run demo mode: `python main.py --mode demo`
4. Check `outputs/` for training logs and results

---

**Last Updated**: April 2026
**Status**: Core pipeline implemented; see `PUBLICATION_NOTES.md` for the current
environment verification boundary.
