"""
WirelessJEPA Data Preprocessing and Loading Module
Implements data loading, normalization, and upsampling as described in the paper
Based on: "WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"
Reference: Paper Section 3.1 - Input signal format and preprocessing
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings('ignore')


class SignalPreprocessor:
    """
    Signal preprocessing module for complex baseband IQ signals.
    Handles conversion to real-valued tensors, normalization, and upsampling.
    
    Paper Reference: Section 3.1 - "Input signal format: x ∈ R^(C×H×W)"
    """
    
    def __init__(self, 
                 normalize_type: str = "unitmax",
                 target_antenna_dim: int = 256,
                 target_time_dim: int = 256):
        """
        Initialize signal preprocessor.
        
        Args:
            normalize_type: Type of normalization ('unitmax': x ← x / max(|x|))
            target_antenna_dim: Target antenna dimension (H) for upsampling
            target_time_dim: Target time dimension (W) for upsampling
        """
        self.normalize_type = normalize_type
        self.target_antenna_dim = target_antenna_dim
        self.target_time_dim = target_time_dim
    
    def complex_to_real(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert complex IQ signal to real-valued tensor.
        
        Paper: x ∈ R^(C×H×W) where C=2 represents I/Q components
        
        Input: Complex signal of shape (H, W) or (W,)
        Output: Real tensor of shape (2, H, W) with [I, Q] components
        """
        is_torch = isinstance(signal, torch.Tensor)
        
        if is_torch:
            signal_np = signal.cpu().numpy()
        else:
            signal_np = signal
        
        # Handle different input shapes
        if signal_np.ndim == 1:
            # Single time series: expand to 2D
            signal_np = signal_np[np.newaxis, :]
        
        # Extract I (real) and Q (imaginary) components
        if np.iscomplexobj(signal_np):
            iq_real = np.real(signal_np)
            iq_imag = np.imag(signal_np)
        else:
            # Already real, split into I/Q if needed
            if signal_np.shape[0] == 2:
                iq_real = signal_np[0]
                iq_imag = signal_np[1]
            else:
                iq_real = signal_np
                iq_imag = np.zeros_like(signal_np)
        
        # Stack I and Q: shape (2, H, W)
        iq_stacked = np.stack([iq_real, iq_imag], axis=0)
        
        if is_torch:
            iq_stacked = torch.from_numpy(iq_stacked).float()
        
        return iq_stacked
    
    def normalize_unitmax(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply unitmax normalization: x ← x / max(|x|)
        
        Paper: Section 3.1 - Normalization: "unitmax normalization x ← x / max(|x|)"
        Ensures signal amplitude is bounded to [-1, 1]
        """
        is_torch = isinstance(signal, torch.Tensor)
        
        if is_torch:
            max_val = torch.max(torch.abs(signal))
            if max_val > 0:
                return signal / (max_val + 1e-8)
            return signal
        else:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                return signal / (max_val + 1e-8)
            return signal
    
    def normalize_zstandard(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Z-score normalization (alternative)"""
        is_torch = isinstance(signal, torch.Tensor)
        
        if is_torch:
            mean = torch.mean(signal)
            std = torch.std(signal)
            return (signal - mean) / (std + 1e-8)
        else:
            mean = np.mean(signal)
            std = np.std(signal)
            return (signal - mean) / (std + 1e-8)
    
    def upsample_antenna_time(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Upsample antenna-time grid using nearest-neighbor interpolation.
        
        Paper: Section 3.1 - "antenna-time grid upsampling"
        "Nearest-neighbor interpolation x_{c,i,t} = (x_raw)_{c,⌊i/64⌋,t}"
        
        For RML2016.10a: Input is (2, 128) [I/Q, time]
        We need to expand to (2, 256, 256) by:
        - Treating the 128 time samples as a 1D signal
        - Upsampling time dimension to 256
        - Creating a pseudo-antenna dimension of 256 (repeating the signal)
        """
        is_torch = isinstance(signal, torch.Tensor)
        
        if is_torch:
            signal_np = signal.cpu().numpy()
        else:
            signal_np = signal
        
        # Handle RML2016.10a format: (2, 128) -> I/Q channels, time samples
        if signal_np.ndim == 2 and signal_np.shape[0] == 2:
            # RML2016.10a compatibility: (2, 128) -> (2, 1, 128)
            signal_np = signal_np[:, np.newaxis, :]

        if signal_np.ndim == 3 and signal_np.shape[0] == 2:
            # Paper format: (2, H, W), commonly (2, 4, 256). Upsample only
            # the antenna plane with nearest-neighbor indexing; interpolate
            # time only for single-channel downstream data such as RML16.
            channels, original_h, original_w = signal_np.shape
            if original_w == self.target_time_dim:
                time_upsampled = signal_np
            else:
                time_indices = np.floor(
                    np.arange(self.target_time_dim) * original_w / self.target_time_dim
                ).astype(int)
                time_indices = np.clip(time_indices, 0, original_w - 1)
                time_upsampled = signal_np[:, :, time_indices]
            
            antenna_indices = np.floor(
                np.arange(self.target_antenna_dim) * original_h / self.target_antenna_dim
            ).astype(int)
            antenna_indices = np.clip(antenna_indices, 0, original_h - 1)
            upsampled = time_upsampled[:, antenna_indices, :]
            
        elif signal_np.ndim == 3:
            # Handle 3D input (batch_size, H, W)
            batch_size, original_h, original_w = signal_np.shape
            
            if original_h == self.target_antenna_dim and original_w == self.target_time_dim:
                upsampled = signal_np
            else:
                # Use PyTorch interpolation for 3D inputs
                signal_torch = torch.from_numpy(signal_np).float() if not is_torch else signal
                
                # Add batch and channel dimensions for interpolation
                signal_expanded = signal_torch.unsqueeze(0)  # (1, batch_size, H, W)
                
                # Bilinear interpolation for antenna-time plane
                upsampled = torch.nn.functional.interpolate(
                    signal_expanded,
                    size=(self.target_antenna_dim, self.target_time_dim),
                    mode='nearest',
                    align_corners=None
                )
                
                upsampled = upsampled.squeeze(0)  # Remove batch dimension
                
                if not is_torch:
                    upsampled = upsampled.numpy()
                
                return upsampled
        else:
            raise ValueError(f"Unsupported signal shape: {signal_np.shape}. Expected (2, 128) for RML2016.10a or (batch, H, W)")
        
        if is_torch:
            upsampled = torch.from_numpy(upsampled).float()
        
        return upsampled
    
    def preprocess(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Complete preprocessing pipeline:
        1. Convert complex to real (I/Q)
        2. Normalize (unitmax)
        3. Upsample antenna-time grid
        
        Returns tensor of shape (2, 256, 256)
        """
        # Convert to real IQ representation
        signal = self.complex_to_real(signal)
        
        # Apply normalization
        if self.normalize_type == "unitmax":
            signal = self.normalize_unitmax(signal)
        elif self.normalize_type == "zstd":
            signal = self.normalize_zstandard(signal)
        
        # Upsample to target dimensions
        signal = self.upsample_antenna_time(signal)
        
        return signal


class RML2016Dataset(Dataset):
    """
    RML2016.10a Dataset Loader
    
    Paper Reference: Section 4 - Experimental Setup on RML2016.10a
    Supports full dataset and few-shot scenarios (1-shot, 100-shot, 500-shot)
    """
    
    def __init__(self, 
                 data_dict: Dict,
                 modulations: List[str],
                 snr_list: List[float],
                 split: str = "train",
                 few_shot: Optional[int] = None,
                 transform: Optional[callable] = None,
                 seed: int = 42):
        """
        Initialize RML2016 dataset.
        
        Args:
            data_dict: Loaded pickle dictionary from RML2016.10a
            modulations: List of modulation types
            snr_list: List of SNR values
            split: 'train', 'val', or 'test'
            few_shot: Number of samples per class for few-shot (None for full dataset)
            transform: Optional preprocessing function
            seed: Random seed for reproducibility
        """
        self.data_dict = data_dict
        self.modulations = modulations
        self.snr_list = snr_list
        self.split = split
        self.few_shot = few_shot
        self.transform = transform
        self.seed = seed
        
        self.preprocessor = SignalPreprocessor()
        self.samples = []
        self.labels = []
        
        self._build_dataset()
    
    def _build_dataset(self):
        """Build dataset index and labels"""
        np.random.seed(self.seed)
        
        for mod_idx, modulation in enumerate(self.modulations):
            for snr in self.snr_list:
                key = (modulation, snr)
                
                if key not in self.data_dict:
                    continue
                
                signals = self.data_dict[key]
                num_signals = len(signals)
                
                # Shuffle before splitting so each split samples the same
                # modulation/SNR distribution without relying on pickle order.
                all_indices = np.arange(num_signals)
                np.random.shuffle(all_indices)
                train_end = int(0.7 * num_signals)
                val_end = int(0.85 * num_signals)
                
                if self.split == "train":
                    indices = all_indices[:train_end]
                elif self.split == "val":
                    indices = all_indices[train_end:val_end]
                elif self.split == "test":
                    indices = all_indices[val_end:]
                else:
                    indices = all_indices
                
                # Apply few-shot sampling
                if self.few_shot is not None and self.few_shot > 0:
                    indices = list(indices)
                    np.random.shuffle(indices)
                    indices = indices[:self.few_shot]
                
                # Add samples
                for idx in indices:
                    self.samples.append((signals[idx], mod_idx))
                    self.labels.append(mod_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample and label"""
        signal, label = self.samples[idx]
        
        # Preprocess: complex → real (I/Q) → normalize → upsample
        signal = self.preprocessor.preprocess(signal)
        
        # Apply optional transform
        if self.transform is not None:
            signal = self.transform(signal)
        
        # Convert to torch tensor if needed
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(signal).float()
        
        return signal, label


class WirelessJEPADataLoader:
    """
    Complete data loading pipeline for WirelessJEPA training and evaluation
    """
    
    def __init__(self, 
                 data_path: str = "./data/RML2016.10a_dict.pkl",
                 batch_size: int = 64,
                 num_workers: int = 0,
                 seed: int = 42):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to RML2016.10a pickle file
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            seed: Random seed
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Check data file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load dataset
        print(f"Loading dataset from {self.data_path}...")
        with open(self.data_path, 'rb') as f:
            self.data_dict = pickle.load(f, encoding='latin1')
        
        # Extract metadata
        self._extract_metadata()
        
        print(f"✓ Dataset loaded: {len(self.modulations)} modulations, {len(self.snr_list)} SNR levels")
    
    def _extract_metadata(self):
        """Extract modulations and SNR values from data dictionary"""
        self.modulations = []
        self.snrs = {}
        
        for key in self.data_dict.keys():
            if isinstance(key, tuple) and len(key) == 2:
                mod, snr = key
                if mod not in self.modulations:
                    self.modulations.append(mod)
                if snr not in self.snrs:
                    self.snrs[snr] = True
        
        self.snr_list = sorted(self.snrs.keys())
        self.modulations = sorted(self.modulations)
        self.num_classes = len(self.modulations)
    
    def get_loaders(self, 
                   few_shot: Optional[int] = None,
                   train_batch_size: Optional[int] = None,
                   val_batch_size: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        Get DataLoaders for train/val/test splits.
        
        Args:
            few_shot: Number of samples per class (None for full dataset)
            train_batch_size: Override batch size for training
            val_batch_size: Override batch size for validation/test
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        train_bs = train_batch_size or self.batch_size
        val_bs = val_batch_size or self.batch_size
        
        # Create datasets
        train_dataset = RML2016Dataset(
            self.data_dict, self.modulations, self.snr_list,
            split="train", few_shot=few_shot, seed=self.seed
        )
        
        val_dataset = RML2016Dataset(
            self.data_dict, self.modulations, self.snr_list,
            split="val", few_shot=few_shot, seed=self.seed
        )
        
        test_dataset = RML2016Dataset(
            self.data_dict, self.modulations, self.snr_list,
            split="test", few_shot=few_shot, seed=self.seed
        )
        
        # Create DataLoaders
        loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=train_bs,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=val_bs,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=val_bs,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )
        }
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        if few_shot:
            print(f"Few-shot: {few_shot} samples per class")
        
        return loaders
    
    def get_modulation_list(self) -> List[str]:
        """Get list of modulation types"""
        return self.modulations
    
    def get_snr_list(self) -> List[float]:
        """Get list of SNR values"""
        return self.snr_list


def create_dataloader(data_path: str = "./data/RML2016.10a_dict.pkl",
                     batch_size: int = 64,
                     few_shot: Optional[int] = None,
                     num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create data loaders.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    loader = WirelessJEPADataLoader(data_path, batch_size, num_workers)
    loaders = loader.get_loaders(few_shot=few_shot)
    return loaders['train'], loaders['val'], loaders['test']


if __name__ == "__main__":
    # Test data loading
    print("Testing dataset loading...")
    
    try:
        loader = WirelessJEPADataLoader("./data/RML2016.10a_dict.pkl", batch_size=32)
        loaders = loader.get_loaders(few_shot=None)
        
        print(f"\nModulations: {loader.get_modulation_list()}")
        print(f"SNR levels: {loader.get_snr_list()}")
        
        # Get a batch
        train_loader = loaders['train']
        for batch_idx, (signals, labels) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Signal shape: {signals.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Signal range: [{signals.min():.4f}, {signals.max():.4f}]")
            break
        
        print("\n✓ Dataset loading test passed!")
    
    except Exception as e:
        print(f"✗ Error: {e}")
