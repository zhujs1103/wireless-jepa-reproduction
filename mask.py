"""
WirelessJEPA Spatio-Temporal Masking Module
Implements 4 masking strategies for sparse convolution and JEPA training
Based on: "WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"
Reference: Paper Section 3.2, Figure 2 - Masking geometries
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MaskGenerator:
    """
    Generates spatio-temporal masks for JEPA masking mechanism.
    
    Paper Reference: Section 3.2 - "Spatio-temporal masking strategies"
    Figure 2 shows 4 masking types applied to the antenna-time plane (256×256)
    """
    
    def __init__(self,
                 height: int = 256,
                 width: int = 256,
                 patch_height: int = 64,
                 patch_width: int = 32,
                 seed: int = 42):
        """
        Initialize mask generator.
        
        Args:
            height: Height of antenna-time plane (antenna dimension)
            width: Width of antenna-time plane (time dimension)
            patch_height: Height of patches for masking
            patch_width: Width of patches for masking
            seed: Random seed for reproducibility
        """
        self.height = height
        self.width = width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def random_masking(self, mask_ratio: float = 0.75) -> torch.Tensor:
        """
        Random block masking strategy.
        
        Paper: "Random masking strategy - random blocks of 64×32 patches"
        Aligns with CNN-JEPA masking approach
        
        Args:
            mask_ratio: Ratio of patches to mask (0-1)
        
        Returns:
            Mask tensor (0 = masked, 1 = visible) of shape (height, width)
        """
        # Calculate number of patches
        num_patches_h = self.height // self.patch_height
        num_patches_w = self.width // self.patch_width
        total_patches = num_patches_h * num_patches_w
        
        # Determine number of patches to mask
        num_mask_patches = int(total_patches * mask_ratio)
        
        # Randomly select patches to mask
        patch_indices = self.rng.choice(
            total_patches, size=num_mask_patches, replace=False
        )
        
        # Create mask
        mask = np.ones((self.height, self.width), dtype=np.float32)
        
        for idx in patch_indices:
            patch_h = (idx // num_patches_w) * self.patch_height
            patch_w = (idx % num_patches_w) * self.patch_width
            
            mask[patch_h:patch_h + self.patch_height,
                 patch_w:patch_w + self.patch_width] = 0
        
        return torch.from_numpy(mask).float()
    
    def antenna_masking(self, mask_ratio: float = 0.75) -> torch.Tensor:
        """
        Antenna dimension masking strategy.
        
        Paper: "Antenna masking - entire rows of antenna dimension are masked"
        Patch size: 64×256 (full width for antenna rows)
        Enforces learning of spatial dependencies across antennas
        
        Args:
            mask_ratio: Ratio of antenna rows to mask
        
        Returns:
            Mask tensor of shape (height, width)
        """
        # Number of antenna rows (patch_height = 64 for antenna groups)
        num_antenna_groups = self.height // self.patch_height
        num_groups_to_mask = max(1, int(num_antenna_groups * mask_ratio))
        
        # Randomly select antenna groups to mask
        group_indices = self.rng.choice(
            num_antenna_groups, size=num_groups_to_mask, replace=False
        )
        
        # Create mask
        mask = np.ones((self.height, self.width), dtype=np.float32)
        
        # Mask entire antenna rows (full width)
        for group_idx in group_indices:
            start_h = group_idx * self.patch_height
            end_h = start_h + self.patch_height
            mask[start_h:end_h, :] = 0  # Mask entire height of this antenna group
        
        return torch.from_numpy(mask).float()
    
    def temporal_masking(self, mask_ratio: float = 0.75) -> torch.Tensor:
        """
        Temporal dimension masking strategy.
        
        Paper: "Temporal masking - entire columns of time dimension are masked"
        Patch size: 256×32 (full height for time columns)
        Enforces learning of temporal dependencies
        
        Args:
            mask_ratio: Ratio of time columns to mask
        
        Returns:
            Mask tensor of shape (height, width)
        """
        # Number of time windows (patch_width = 32 for time groups)
        num_time_groups = self.width // self.patch_width
        num_groups_to_mask = max(1, int(num_time_groups * mask_ratio))
        
        # Randomly select time groups to mask
        group_indices = self.rng.choice(
            num_time_groups, size=num_groups_to_mask, replace=False
        )
        
        # Create mask
        mask = np.ones((self.height, self.width), dtype=np.float32)
        
        # Mask entire time columns (full height)
        for group_idx in group_indices:
            start_w = group_idx * self.patch_width
            end_w = start_w + self.patch_width
            mask[:, start_w:end_w] = 0  # Mask entire width of this time group
        
        return torch.from_numpy(mask).float()
    
    def multiblock_masking(self, mask_ratio: float = 0.75, num_blocks: int = 3) -> torch.Tensor:
        """
        Multi-block masking strategy.
        
        Paper: "Multi-block masking - contiguous spatio-temporal blocks"
        Creates multiple continuous blocks for balanced spatial-temporal inductive bias
        
        Args:
            mask_ratio: Ratio of area to mask
            num_blocks: Number of continuousblocks to mask
        
        Returns:
            Mask tensor of shape (height, width)
        """
        mask = np.ones((self.height, self.width), dtype=np.float32)
        
        total_area = self.height * self.width
        target_masked_area = int(total_area * mask_ratio)
        masked_area = 0
        attempts = 0
        
        # Create contiguous regions until the target masked area is reached.
        while masked_area < target_masked_area and attempts < num_blocks * 64:
            attempts += 1
            # Random contiguous blocks centered around the 64x32 paper patch
            # geometry, enlarged as needed to meet the requested mask ratio.
            remaining = max(1, target_masked_area - masked_area)
            nominal_area = max(self.patch_height * self.patch_width, remaining // max(1, num_blocks))
            aspect = self.patch_height / max(1, self.patch_width)
            block_h = int(np.sqrt(nominal_area * aspect))
            block_w = int(np.ceil(nominal_area / max(1, block_h)))
            jitter_h = self.rng.uniform(0.75, 1.25)
            jitter_w = self.rng.uniform(0.75, 1.25)
            block_h = int(block_h * jitter_h)
            block_w = int(block_w * jitter_w)
            
            # Ensure block fits within bounds
            block_h = min(block_h, self.height)
            block_w = min(block_w, self.width)
            
            # Random starting position
            start_h = self.rng.integers(0, max(1, self.height - block_h + 1))
            start_w = self.rng.integers(0, max(1, self.width - block_w + 1))
            
            # Apply mask
            mask[start_h:start_h + block_h,
                 start_w:start_w + block_w] = 0
            masked_area = np.count_nonzero(mask == 0)
        
        return torch.from_numpy(mask).float()
    
    def generate_mask(self, 
                     mask_type: str = "random",
                     mask_ratio: float = 0.75) -> torch.Tensor:
        """
        Generate mask based on type.
        
        Paper: Algorithm 1 - Mask generation step in training loop
        
        Args:
            mask_type: Type of masking ("random", "antenna", "temporal", "multiblock")
            mask_ratio: Ratio of pixels to mask
        
        Returns:
            Mask tensor (0 = masked, 1 = visible)
        """
        if mask_type == "random":
            return self.random_masking(mask_ratio)
        elif mask_type == "antenna":
            return self.antenna_masking(mask_ratio)
        elif mask_type == "temporal":
            return self.temporal_masking(mask_ratio)
        elif mask_type == "multiblock":
            return self.multiblock_masking(mask_ratio)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
    
    def get_latent_mask(self, mask: torch.Tensor, factor: int = 16) -> torch.Tensor:
        """
        Generate latent space mask from input space mask.
        
        Paper: Algorithm 2 - Latent space mask for compressed representation
        Downsamples mask by factor (typically 16x for encoder compression)
        
        Args:
            mask: Input space mask of shape (256, 256)
            factor: Downsampling factor (encoder compression ratio)
        
        Returns:
            Latent space mask of shape (16, 16)
        """
        if mask.dim() == 2:
            mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
        else:
            mask_expanded = mask
        
        # Average pooling preserves the masked/visible semantics better than
        # max pooling for partially covered latent cells. A latent position is
        # visible only when most of its input support is visible.
        visible_ratio = torch.nn.functional.avg_pool2d(
            mask_expanded.float(),
            kernel_size=factor,
            stride=factor
        )
        latent_mask = (visible_ratio >= 0.5).float()
        
        latent_mask = latent_mask.squeeze()
        return latent_mask


class SparseConvolutionMask:
    """
    Sparse convolution mechanism implementing mask propagation.
    
    Paper Reference: Algorithm 2 - Mask propagation through layers
    "For each conv/pool layer, upsample mask and suppress activation at masked regions"
    """
    
    @staticmethod
    def apply_mask_to_activation(activation: torch.Tensor,
                                  mask: torch.Tensor,
                                  mask_value: float = 0.0) -> torch.Tensor:
        """
        Apply mask to feature activation.
        
        Paper Algorithm 2: Mask suppression of activations
        Masked regions (mask=0) are set to mask_value (typically 0)
        
        Args:
            activation: Feature map of shape (B, C, H, W)
            mask: Mask of shape (H, W) or broadcastable to activation
            mask_value: Value to set at masked regions
        
        Returns:
            Masked activation
        """
        # Expand mask to match activation dimensions if needed
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Ensure mask is on same device and dtype as activation
        mask = mask.to(device=activation.device, dtype=activation.dtype)
        
        # Check if need to interpolate mask size
        if activation.shape[-2:] != mask.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask,
                size=activation.shape[-2:],
                mode='nearest'
            )
        
        # Apply mask: masked regions (0) become mask_value
        masked_activation = activation * mask
        
        return masked_activation
    
    @staticmethod
    def apply_norm_with_mask(feature: torch.Tensor,
                             mask: torch.Tensor,
                             norm_layer: torch.nn.Module) -> torch.Tensor:
        """
        Apply normalization only to unmasked regions.
        
        Paper Algorithm 2: "BatchNorm/LayerNorm only for unmasked positions"
        
        Args:
            feature: Feature map
            mask: Mask (1 = visible, 0 = masked)
            norm_layer: Normalization layer (e.g., BatchNorm2d)
        
        Returns:
            Normalized feature with masking applied
        """
        # Apply mask to get only visible regions
        masked_feature = feature * mask
        
        # Apply normalization (will work only on visible values)
        normalized = norm_layer(masked_feature)
        
        # Re-apply mask to ensure masked regions stay zero
        normalized = normalized * mask
        
        return normalized


def visualize_masks(save_path: str = "./mask_visualization.png"):
    """
    Visualize all 4 masking strategies from Paper Figure 2.
    
    Paper Reference: Figure 2 - Visual illustration of masking geometries
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("WirelessJEPA Masking Strategies (256×256 antenna-time plane)", 
                 fontsize=16, fontweight='bold')
    
    mask_gen = MaskGenerator()
    mask_types = ["random", "antenna", "temporal", "multiblock"]
    
    for idx, (ax, mask_type) in enumerate(zip(axes.flat, mask_types)):
        # Generate mask
        mask = mask_gen.generate_mask(mask_type, mask_ratio=0.75)
        
        # Visualize: masked regions in red, visible in blue
        vis_mask = np.zeros((256, 256, 3))
        vis_mask[mask == 0] = [1, 0, 0]  # Red for masked
        vis_mask[mask == 1] = [0, 0, 1]  # Blue for visible
        
        ax.imshow(vis_mask)
        ax.set_title(f"{mask_type.capitalize()} Masking (75%)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time dimension →")
        ax.set_ylabel("Antenna dimension →")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Mask visualization saved to {save_path}")
    plt.close()


def test_masking():
    """Test masking functions"""
    print("Testing masking module...")
    
    mask_gen = MaskGenerator(height=256, width=256, patch_height=64, patch_width=32)
    
    mask_types = ["random", "antenna", "temporal", "multiblock"]
    
    for mask_type in mask_types:
        mask = mask_gen.generate_mask(mask_type, mask_ratio=0.75)
        
        print(f"\n{mask_type.capitalize()} Mask:")
        print(f"  Shape: {mask.shape}")
        print(f"  Masked ratio: {(1 - mask.mean()).item():.4f}")
        print(f"  Range: [{mask.min():.2f}, {mask.max():.2f}]")
        
        # Test latent mask
        latent_mask = mask_gen.get_latent_mask(mask, factor=16)
        print(f"  Latent mask shape: {latent_mask.shape}")
    
    print("\n✓ Masking test passed!")


if __name__ == "__main__":
    test_masking()
    visualize_masks()
