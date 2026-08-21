"""
WirelessJEPA Core Model Architecture
Implements ShuffleNetV2-based encoder, sparse convolution, predictor, and teacher network
Based on: "WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"
Reference: Paper Section 3 - Architecture, Algorithm 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
import copy


def _adapt_mask(mask: Optional[torch.Tensor], tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Resize a visible=1/masked=0 mask to a feature tensor."""
    if mask is None:
        return None
    layer_mask = mask
    if layer_mask.dim() == 3:
        layer_mask = layer_mask.unsqueeze(1)
    layer_mask = layer_mask.to(device=tensor.device, dtype=tensor.dtype)
    if layer_mask.shape[-2:] != tensor.shape[-2:]:
        layer_mask = F.interpolate(layer_mask, size=tensor.shape[-2:], mode="nearest")
    return layer_mask


def _apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    layer_mask = _adapt_mask(mask, tensor)
    if layer_mask is None:
        return tensor
    return tensor * layer_mask


def _masked_batch_norm(x: torch.Tensor, mask: Optional[torch.Tensor],
                       bn: nn.BatchNorm2d) -> torch.Tensor:
    """BatchNorm over visible positions only, matching Algorithm 2."""
    layer_mask = _adapt_mask(mask, x)
    if layer_mask is None:
        return bn(x)

    if not bn.training:
        out = F.batch_norm(
            x,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=False,
            momentum=0.0,
            eps=bn.eps,
        )
        return out * layer_mask

    visible = layer_mask.expand_as(x)
    count = visible.sum(dim=(0, 2, 3)).clamp_min(1.0)
    mean = (x * visible).sum(dim=(0, 2, 3)) / count
    centered = (x - mean.view(1, -1, 1, 1)) * visible
    var = (centered ** 2).sum(dim=(0, 2, 3)) / count

    if bn.track_running_stats:
        with torch.no_grad():
            bn.running_mean.mul_(1.0 - bn.momentum).add_(bn.momentum * mean.detach())
            bn.running_var.mul_(1.0 - bn.momentum).add_(bn.momentum * var.detach())
            bn.num_batches_tracked.add_(1)

    out = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + bn.eps)
    if bn.weight is not None:
        out = out * bn.weight.view(1, -1, 1, 1)
    if bn.bias is not None:
        out = out + bn.bias.view(1, -1, 1, 1)
    return out * layer_mask


class ShuffleNetV2Block(nn.Module):
    """ShuffleNetV2 basic block with channel shuffle"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.stride = stride
        
        if stride == 1 and in_channels == out_channels:
            # Standard ShuffleNetV2 block for same channels
            hidden_channels = out_channels // 2
            
            # Depthwise → Pointwise → Depthwise → Pointwise path
            self.conv1 = nn.Conv2d(in_channels // 2, hidden_channels, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, 
                                  groups=hidden_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
            
            self.conv3 = nn.Conv2d(hidden_channels, out_channels // 2, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels // 2)
            
            self.use_split = True
        else:
            # Channel expansion block for stride=2 or channel increase
            hidden_channels = out_channels // 2
            
            # Left branch (shortcut)
            if stride == 2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels // 2),
                    nn.ReLU(inplace=True)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels // 2),
                    nn.ReLU(inplace=True)
                )
            
            # Right branch (convolution path)
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            
            self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, 
                                  groups=hidden_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
            
            self.conv3 = nn.Conv2d(hidden_channels, out_channels // 2, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels // 2)
            
            self.use_split = False
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with channel split, shuffle, and optional mask propagation."""
        def apply_mask(tensor: torch.Tensor) -> torch.Tensor:
            return _apply_mask(tensor, mask)

        def apply_bn(tensor: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
            return _masked_batch_norm(tensor, mask, bn)

        def forward_shortcut(tensor: torch.Tensor) -> torch.Tensor:
            for layer in self.shortcut:
                if isinstance(layer, nn.BatchNorm2d):
                    tensor = _masked_batch_norm(tensor, mask, layer)
                else:
                    tensor = layer(tensor)
                    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                        tensor = apply_mask(tensor)
            return apply_mask(tensor)

        if self.use_split:
            # Standard block with channel split
            # Split channels
            x1, x2 = x.chunk(2, 1)
            x2 = apply_mask(x2)
            
            # First path (branch 1)
            out = self.conv1(x1)
            out = apply_bn(out, self.bn1)
            out = F.relu(out, inplace=True)
            out = apply_mask(out)
            
            out = self.conv2(out)
            out = apply_bn(out, self.bn2)
            out = apply_mask(out)
            
            out = self.conv3(out)
            out = apply_bn(out, self.bn3)
            out = F.relu(out, inplace=True)
            out = apply_mask(out)
            
            # Concatenate and shuffle
            out = torch.cat([x2, out], 1)
            
            # Channel shuffle
            b, c, h, w = out.size()
            out = out.view(b, 2, c // 2, h, w)
            out = out.permute(0, 2, 1, 3, 4)
            out = out.contiguous().view(b, c, h, w)
            out = apply_mask(out)
            
            return out
        else:
            # Channel expansion block
            # Left branch
            out1 = forward_shortcut(x)
            
            # Right branch
            out2 = self.conv1(x)
            out2 = apply_bn(out2, self.bn1)
            out2 = F.relu(out2, inplace=True)
            out2 = apply_mask(out2)
            
            out2 = self.conv2(out2)
            out2 = apply_bn(out2, self.bn2)
            out2 = apply_mask(out2)
            
            out2 = self.conv3(out2)
            out2 = apply_bn(out2, self.bn3)
            out2 = F.relu(out2, inplace=True)
            out2 = apply_mask(out2)
            
            # Concatenate
            out = torch.cat([out1, out2], 1)
            
            # Channel shuffle
            b, c, h, w = out.size()
            out = out.view(b, 2, c // 2, h, w)
            out = out.permute(0, 2, 1, 3, 4)
            out = out.contiguous().view(b, c, h, w)
            out = apply_mask(out)
            
            return out


class ShuffleNetV2Encoder(nn.Module):
    """
    ShuffleNetV2-x0.5 encoder backbone for 2-channel IQ input.
    
    Paper Reference: Section 3.1 - "Context and teacher encoders use ShuffleNetV2-x0.5"
    Adapted for 2-channel (I/Q) wireless signal input instead of RGB
    """
    
    def __init__(self, in_channels: int = 2, depth_multiplier: float = 0.5):
        """
        Initialize ShuffleNetV2 encoder.
        
        Args:
            in_channels: Number of input channels (2 for I/Q)
            depth_multiplier: Channel multiplier (0.5 for x0.5 variant)
        """
        super().__init__()
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        
        # Base channels for x0.5 variant
        stage_channels = [24, 48, 96, 192, 1024]
        stage_channels = [int(c * depth_multiplier) for c in stage_channels]
        
        # Initial convolution for IQ input
        self.conv1 = nn.Conv2d(in_channels, stage_channels[0], 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(stage_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stages with ShuffleNetV2 blocks
        self.stage2 = self._make_stage(stage_channels[0], stage_channels[1], num_blocks=4, stride=2)
        self.stage3 = self._make_stage(stage_channels[1], stage_channels[2], num_blocks=8, stride=2)
        self.stage4 = self._make_stage(stage_channels[2], stage_channels[3], num_blocks=4, stride=2)
        
        # Final layers
        self.conv5 = nn.Conv2d(stage_channels[3], stage_channels[4], 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(stage_channels[4])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.output_dim = stage_channels[4]
        
        self._init_weights()
    
    def _make_stage(self, in_c: int, out_c: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a stage with multiple ShuffleNetV2 blocks"""
        layers = []
        
        # First block with stride
        layers.append(ShuffleNetV2Block(in_c, out_c, stride=stride))
        
        # Remaining blocks with stride 1
        for _ in range(num_blocks - 1):
            layers.append(ShuffleNetV2Block(out_c, out_c, stride=1))
        
        return nn.Sequential(*layers)

    def _forward_stage(self, stage: nn.Sequential, x: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward a ShuffleNet stage while propagating the sparse mask per block."""
        for block in stage:
            x = block(x, mask=mask)
        return x
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that preserves the spatial latent feature map.

        Returns:
            - latent_map: Spatial feature map used for JEPA prediction
            - features: Intermediate features from each stage [stage2, stage3, stage4]
        """
        def apply_layer_mask(tensor: torch.Tensor) -> torch.Tensor:
            return _apply_mask(tensor, mask)

        # Initial convolution
        x = self.conv1(x)
        x = _masked_batch_norm(x, mask, self.bn1)
        x = F.relu(x, inplace=True)
        x = apply_layer_mask(x)
        x = self.maxpool(x)
        x = apply_layer_mask(x)
        
        # Extract features from each stage
        x2 = self._forward_stage(self.stage2, x, mask=mask)
        x2 = apply_layer_mask(x2)
        x3 = self._forward_stage(self.stage3, x2, mask=mask)
        x3 = apply_layer_mask(x3)
        x4 = self._forward_stage(self.stage4, x3, mask=mask)
        x4 = apply_layer_mask(x4)
        
        # Stage-3 has 16x16 spatial resolution for 256x256 input, matching
        # the latent mask used by the JEPA objective in this implementation.
        latent_map = x3
        features = [x2, x3, x4]
        
        return latent_map, features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Dense encoder forward for downstream feature extraction.

        Returns:
            - output: Global average pooled feature (B, output_dim)
            - features: Intermediate features from each stage [stage2, stage3, stage4]
        """
        _, features = self.forward_features(x, mask=None)
        x4 = features[-1]

        x = self.conv5(x4)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        
        # Global average pooling
        output = self.gap(x)
        output = output.view(output.size(0), -1)
        
        return output, features


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 1, stride: int = 1):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                                   padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        return x


class LightweightPredictor(nn.Module):
    """
    Lightweight predictor using depthwise separable convolutions.
    
    Paper Reference: Section 3.2 - "3-layer lightweight predictor"
    Predicts latent representations at masked positions
    """
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 256, hidden_dim: int = 256, num_layers: int = 3):
        """
        Initialize predictor.
        
        Args:
            input_channels: Number of input channels (latent space channels)
            feature_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of depthwise separable conv layers
        """
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        
        # First layer: input_channels → hidden_dim
        layers.append(DepthwiseSeparableConv(input_channels, hidden_dim, 3, 1, 1))
        
        # Middle layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            layers.append(DepthwiseSeparableConv(hidden_dim, hidden_dim, 3, 1, 1))
        
        # Last layer: hidden_dim → feature_dim
        layers.append(DepthwiseSeparableConv(hidden_dim, feature_dim, 3, 1, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict masked latent representations.
        
        Paper Algorithm 1: Predictor forward for masked positions
        """
        return self.layers(x)


class ContextEncoder(nn.Module):
    """
    Context encoder with sparse convolution mechanism.
    
    Paper Reference: Algorithm 2 - "Context encoder with mask propagation"
    Applies mask suppression at each layer
    """
    
    def __init__(self, encoder: ShuffleNetV2Encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with optional mask propagation.
        
        Paper Algorithm 2: "For each conv/pool layer, upsample mask and suppress"
        
        Args:
            x: Input tensor (B, 2, 256, 256)
            mask: Optional mask tensor for sparse convolution
        
        Returns:
            - features: Context encoded features
            - intermediate_features: Features from intermediate layers
        """
        return self.encoder.forward_features(x, mask=mask)


class TeacherEncoder(nn.Module):
    """
    Momentum teacher encoder.
    
    Paper Reference: Section 3.3 - "Momentum teacher network"
    Parameters updated via exponential moving average (EMA)
    """
    
    def __init__(self, encoder: ShuffleNetV2Encoder):
        super().__init__()
        self.encoder = encoder
        
        # Copy encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass without gradient computation.
        
        Paper: Teacher network operates without gradients
        """
        features, intermediate = self.encoder(x)
        return features, intermediate
    
    @torch.no_grad()
    def ema_update(self, student_encoder: ShuffleNetV2Encoder, tau: float = 0.996):
        """
        Update teacher parameters via exponential moving average.
        
        Paper Algorithm 1: "EMA update: θ_t' ← τ·θ_t' + (1-τ)·θ_t"
        
        Args:
            student_encoder: Student encoder to update from
            tau: EMA momentum coefficient
        """
        for teacher_param, student_param in zip(
            self.encoder.parameters(), student_encoder.parameters()
        ):
            teacher_param.data = tau * teacher_param.data + (1.0 - tau) * student_param.data


class LearnableMaskToken(nn.Module):
    """
    Learnable mask token for sparse latent representation.
    
    Paper Reference: Section 3.2 - "Learnable masked latent tokens"
    Inserted at masked positions in latent space
    """
    
    def __init__(self, mask_token_dim: int = 256):
        super().__init__()
        self.mask_token_dim = mask_token_dim
        
        # Initialize learnable token
        self.mask_token = nn.Parameter(torch.randn(1, mask_token_dim, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Insert learnable mask tokens at masked positions.
        
        Paper Algorithm 2: "Insert mask tokens at masked positions"
        
        Args:
            features: Latent features (B, dim, H, W)
            mask: Mask tensor (B, 1, H, W) where 0 = masked, 1 = visible
        
        Returns:
            Features with mask tokens at masked regions
        """
        # Expand mask token to match feature dimensions
        mask_tokens = self.mask_token.expand(features.size(0), -1, features.size(2), features.size(3))
        
        # Insert mask tokens where mask is 0
        # Ensure mask has same number of channels as features
        if mask.dim() == features.dim() - 1:
            # mask is (B, H, W), features is (B, C, H, W), add channel dimension
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        masked_regions = (mask == 0).float()
        output = features * (1 - masked_regions) + mask_tokens * masked_regions
        
        return output


class WirelessJEPA(nn.Module):
    """
    Complete WirelessJEPA model combining all components.
    
    Paper Reference: Section 3 - Complete architecture
    Implements Algorithm 1 and Algorithm 2 from the paper
    """
    
    def __init__(self, config):
        """
        Initialize complete WirelessJEPA model.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        super().__init__()
        self.config = config
        
        # Create independent student and teacher encoders. The teacher must not
        # share Parameter objects with the context encoder because it is updated
        # only by EMA, not by gradient descent.
        base_encoder = ShuffleNetV2Encoder(
            in_channels=config.model.input_channels,
            depth_multiplier=0.5
        )
        teacher_encoder = copy.deepcopy(base_encoder)
        
        # Context and teacher encoders
        self.context_encoder = ContextEncoder(base_encoder)
        self.teacher_encoder = TeacherEncoder(teacher_encoder)

        self.latent_dim = config.model.feature_dim
        with torch.no_grad():
            was_training = base_encoder.training
            base_encoder.eval()
            dummy_input = torch.zeros(1, config.model.input_channels, 256, 256)
            dummy_latent, _ = base_encoder.forward_features(dummy_input)
            base_encoder.train(was_training)
            self.encoder_latent_channels = dummy_latent.shape[1]
        self.projection = nn.Conv2d(self.encoder_latent_channels, self.latent_dim, kernel_size=1)
        self.teacher_projection = copy.deepcopy(self.projection)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_projection.parameters():
            param.requires_grad = False
        
        # Predictor maps spatial latent features to teacher latent targets.
        self.predictor = LightweightPredictor(
            input_channels=self.latent_dim,
            feature_dim=self.latent_dim,
            hidden_dim=config.model.predictor_hidden_dim,
            num_layers=config.model.predictor_depth
        )
        
        # Learnable mask token
        if config.model.learnable_mask_token:
            self.mask_token = LearnableMaskToken(self.latent_dim)
        
        self.feature_dim = self.latent_dim
    
    def encode_context(self, x: torch.Tensor, 
                      mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Context encoder forward pass with mask-aware computation.
        
        Paper Algorithm 1: "Context encoder with mask"
        
        Args:
            x: Input tensor (B, 2, 256, 256)
            mask: Optional mask for sparse convolution
        
        Returns:
            - context_features: Context encoded features
            - intermediate_features: Intermediate layer features
        """
        context_features, intermediate_features = self.context_encoder(x, mask)
        context_features = self.projection(context_features)
        return context_features, intermediate_features
    
    @torch.no_grad()
    def encode_teacher(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Teacher encoder forward pass (no gradients).
        
        Paper Algorithm 1: "Teacher encoder forward (no mask)"
        """
        teacher_features, intermediate = self.teacher_encoder.encoder.forward_features(x, mask=None)
        teacher_features = self.teacher_projection(teacher_features)
        return teacher_features, intermediate

    @torch.no_grad()
    def update_teacher(self, tau: float = 0.996):
        """EMA update for the dense teacher encoder and latent projection."""
        self.teacher_encoder.ema_update(self.context_encoder.encoder, tau=tau)
        for teacher_param, student_param in zip(
            self.teacher_projection.parameters(), self.projection.parameters()
        ):
            teacher_param.data = tau * teacher_param.data + (1.0 - tau) * student_param.data

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled EMA teacher representations for downstream evaluation."""
        teacher_features, _ = self.encode_teacher(x)
        return F.adaptive_avg_pool2d(teacher_features, (1, 1)).flatten(1)
    
    def predict_masked(self, context_features: torch.Tensor, 
                      mask_latent: torch.Tensor) -> torch.Tensor:
        """
        Predict latent representations at masked positions.
        
        Paper Algorithm 2: "Predictor forward on masked positions"
        
        Args:
            context_features: Context encoded features
            mask_latent: Latent space mask
        
        Returns:
            Predicted features at masked positions
        """
        return self.predictor(context_features)
    
    def forward(self, x: torch.Tensor, x_teacher: torch.Tensor,
               mask: torch.Tensor, mask_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass following Paper Algorithm 1.
        
        Paper Algorithm 1: Full training forward
        "
        1. Input upsampling
        2. Generate context & target masks
        3. Encode context with masking
        4. Insert mask tokens in latent space
        5. Predict masked latents with predictor
        6. Encode target representation with teacher
        7. Compute JEPA loss
        "
        
        Args:
            x: Input tensor with mask applied (B, 2, 256, 256)
            x_teacher: Full input tensor for teacher (B, 2, 256, 256)
            mask: Input space mask
            mask_latent: Latent space mask
        
        Returns:
            Dictionary containing:
            - context_features: Context encoded features
            - target_features: Teacher encoded features
            - predicted_features: Predicted masked features
            - mask: Mask tensor
        """
        # Context encoding with mask
        context_features, _ = self.encode_context(x, mask)
        
        # Insert learnable mask tokens
        if hasattr(self, 'mask_token'):
            context_features = self.mask_token(context_features, mask_latent)
        
        # Predictor forward
        predicted_features = self.predictor(context_features)
        
        # Teacher encoding (no grad, no mask)
        with torch.no_grad():
            teacher_features, _ = self.encode_teacher(x_teacher)
        
        return {
            'context_features': context_features,
            'predicted_features': predicted_features,
            'target_features': teacher_features,
            'mask': mask_latent
        }


def create_model(config):
    """Factory function to create WirelessJEPA model"""
    return WirelessJEPA(config)


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing WirelessJEPA model...")
    
    from config import WirelessJEPAConfig
    
    config = WirelessJEPAConfig()
    model = create_model(config).cuda() if torch.cuda.is_available() else create_model(config)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 2, 256, 256)
    x_teacher = torch.randn(batch_size, 2, 256, 256)
    mask = torch.randint(0, 2, (batch_size, 256, 256)).float()
    mask_latent = torch.randint(0, 2, (batch_size, 16, 16)).float()
    
    if torch.cuda.is_available():
        x, x_teacher, mask, mask_latent = (x.cuda(), x_teacher.cuda(), 
                                           mask.cuda(), mask_latent.cuda())
        model = model.cuda()
    
    # Forward pass
    output = model(x, x_teacher, mask, mask_latent)
    
    print(f"✓ Model created successfully!")
    print(f"  Input shape: {x.shape}")
    print(f"  Context features shape: {output['context_features'].shape}")
    print(f"  Predicted features shape: {output['predicted_features'].shape}")
    print(f"  Target features shape: {output['target_features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
