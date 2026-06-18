"""
Neural Dynamic GI (NDGI) - PyTorch Implementation
=================================================
Based on the paper:
"Neural Dynamic GI: Random-Access Neural Compression for Temporal
Lightmaps in Dynamic Lighting Environments"

This code implements the core components of NDGI:
1. Hybrid Feature Maps (2D + 3D)
2. Lightweight MLP Decoder
3. BC Compression Simulation
4. Time Encoding
5. Training Pipeline

Author: AI Assistant (based on paper 2604.12625v2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import argparse
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # CJK fallback
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# 导入测试信号集
from test_signal import SIGNAL_REGISTRY, SIGNAL_DESCRIPTIONS


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NDGIConfig:
    """Configuration for NDGI model profiles."""
    # Feature map resolutions (Table 1 in paper)
    f2d_uv_resolution: Tuple[int, int] = (128, 128)   # F2D_uv
    f2d_ut_resolution: Tuple[int, int] = (64, 24)     # F2D_ut
    f2d_vt_resolution: Tuple[int, int] = (64, 24)     # F2D_vt
    f3d_uvt_resolution: Tuple[int, int, int] = (32, 32, 12)  # F3D_uvt
    
    # Channels
    f2d_uv_channels: int = 4
    f2d_ut_channels: int = 2
    f2d_vt_channels: int = 2
    f3d_uvt_channels: int = 4
    
    # MLP decoder
    hidden_size: int = 16  # Can be 16, 32, 64
    num_hidden_layers: int = 2
    
    # Time encoding
    time_freq_bands: int = 2  # sin/cos pairs: 2 -> 4 dims
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 2**12  # 4096 as per paper
    quantization_bits: int = 8
    alpha_noise: float = 1.0 / 256.0  # For 8-bit quantization simulation
    
    # BC compression
    bc_block_size: int = 4
    
    # Tile-based training
    tile_size: int = 128
    
    # Output
    output_channels: int = 3  # RGB lightmaps


# =============================================================================
# Time Encoding
# =============================================================================

class TimeEncoding(nn.Module):
    """
    Positional time encoding using sinusoidal functions.
    Equation (4) in paper: gamma(t) = [sin(2^0*pi*t), cos(2^0*pi*t), ...]
    """
    def __init__(self, freq_bands: int = 2):
        super().__init__()
        self.freq_bands = freq_bands
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, shape (B, 1), normalized to [0, 1]
        Returns:
            Encoded time, shape (B, 2 * freq_bands)
        """
        encodings = []
        for i in range(self.freq_bands):
            freq = 2 ** i * np.pi
            encodings.append(torch.sin(freq * t))
            encodings.append(torch.cos(freq * t))
        return torch.cat(encodings, dim=-1)


# =============================================================================
# Hybrid Feature Maps
# =============================================================================

class HybridFeatureMaps(nn.Module):
    """
    Hybrid feature map structure as described in Section 3.3.
    
    Feature maps:
    - F2D_uv: 2D spatial feature map (same size as target lightmap)
    - F2D_ut, F2D_vt: Tri-plane features for temporal variation
    - F3D_uvt: 3D feature map at lower resolution for temporal info
    """
    def __init__(self, config: NDGIConfig):
        super().__init__()
        self.config = config
        
        # 2D spatial feature map (same resolution as lightmap)
        self.f2d_uv = nn.Parameter(
            torch.randn(1, config.f2d_uv_channels, *config.f2d_uv_resolution) * 0.01
        )
        
        # Tri-plane features for temporal variation
        self.f2d_ut = nn.Parameter(
            torch.randn(1, config.f2d_ut_channels, *config.f2d_ut_resolution) * 0.01
        )
        self.f2d_vt = nn.Parameter(
            torch.randn(1, config.f2d_vt_channels, *config.f2d_vt_resolution) * 0.01
        )
        
        # 3D feature map at lower resolution
        self.f3d_uvt = nn.Parameter(
            torch.randn(1, config.f3d_uvt_channels, *config.f3d_uvt_resolution) * 0.01
        )
        
    def get_feature_params(self) -> Dict[str, nn.Parameter]:
        """Return all feature map parameters."""
        return {
            'f2d_uv': self.f2d_uv,
            'f2d_ut': self.f2d_ut,
            'f2d_vt': self.f2d_vt,
            'f3d_uvt': self.f3d_uvt
        }
        
    def get_feature_info(self) -> Dict[str, Tuple]:
        """Return feature map shapes and sizes."""
        info = {}
        for name, param in self.get_feature_params().items():
            info[name] = {
                'shape': tuple(param.shape),
                'numel': param.numel(),
                'size_mb': param.numel() * 4 / (1024 * 1024)  # float32
            }
        return info


# =============================================================================
# BC Compression Simulation
# =============================================================================

class BCSimulation(nn.Module):
    """
    Block Compression (BC) simulation strategy as described in Section 3.4.
    
    BC7 encodes each 4x4 texel block using:
    - Two endpoints e1, e2
    - Per-pixel interpolation weights w_p
    
    f_p = (1 - w_p) * e1 + w_p * e2
    
    During training, we directly optimize endpoints and weights instead of
    raw feature values to ensure compatibility with standard BC formats.
    """
    def __init__(self, height: int, width: int, channels: int, block_size: int = 4):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.block_size = block_size
        
        # Calculate number of blocks
        self.num_blocks_h = height // block_size
        self.num_blocks_w = width // block_size
        self.num_blocks = self.num_blocks_h * self.num_blocks_w
        
        # Endpoints: 2 per block (e1, e2)
        self.endpoints = nn.Parameter(
            torch.rand(self.num_blocks, 2, channels)  # [0, 1]
        )
        
        # Weights: 16 per block (for 4x4)
        self.weights = nn.Parameter(
            torch.rand(self.num_blocks, block_size * block_size)  # [0, 1]
        )
        
    def forward(self) -> torch.Tensor:
        """
        Reconstruct feature map from BC endpoints and weights.
        Returns: (1, C, H, W) feature map
        """
        # Reconstruct each block
        blocks = []
        for b in range(self.num_blocks):
            e1 = self.endpoints[b, 0]  # (C,)
            e2 = self.endpoints[b, 1]  # (C,)
            w = self.weights[b]  # (16,)
            
            # f_p = (1 - w_p) * e1 + w_p * e2 for each pixel
            # w: (16, 1), e1: (1, C), e2: (1, C)
            w = w.unsqueeze(-1)  # (16, 1)
            e1 = e1.unsqueeze(0)  # (1, C)
            e2 = e2.unsqueeze(0)  # (1, C)
            
            block_features = (1 - w) * e1 + w * e2  # (16, C)
            block_features = block_features.view(self.block_size, self.block_size, self.channels)
            block_features = block_features.permute(2, 0, 1)  # (C, 4, 4)
            blocks.append(block_features)
        
        # Arrange blocks into full feature map
        blocks = torch.stack(blocks)  # (num_blocks, C, 4, 4)
        
        # Reshape into grid
        blocks = blocks.view(
            self.num_blocks_h, self.num_blocks_w, 
            self.channels, self.block_size, self.block_size
        )
        
        # Use unfold-like arrangement
        feature_map = blocks.permute(2, 0, 3, 1, 4).contiguous()
        feature_map = feature_map.view(
            self.channels, 
            self.num_blocks_h * self.block_size, 
            self.num_blocks_w * self.block_size
        )
        
        return feature_map.unsqueeze(0)  # (1, C, H, W)
    
    def get_compressed_size(self, bits_per_weight: int = 8, bits_per_endpoint: int = 8) -> int:
        """Calculate compressed size in bits."""
        endpoint_bits = self.num_blocks * 2 * self.channels * bits_per_endpoint
        weight_bits = self.num_blocks * (self.block_size ** 2) * bits_per_weight
        return endpoint_bits + weight_bits


class BCFeatureMapCompressor(nn.Module):
    """
    Compresses feature maps using BC simulation for F3D and F2D_uv,
    and simple quantization for F2D_ut and F2D_vt.
    """
    def __init__(self, config: NDGIConfig):
        super().__init__()
        self.config = config
        
        # BC simulation for F3D_uvt (sliced along temporal dim)
        # For simplicity, we treat it as a 3D texture and slice
        h, w, t = config.f3d_uvt_resolution
        self.bc_f3d = BCSimulation(h, w * t, config.f3d_uvt_channels, config.bc_block_size)
        
        # BC simulation for F2D_uv (spatial feature map)
        h, w = config.f2d_uv_resolution
        self.bc_f2d_uv = BCSimulation(h, w, config.f2d_uv_channels, config.bc_block_size)
        
        # Simple quantization for F2D_ut and F2D_vt (smaller, no BC needed)
        self.register_buffer('quant_noise_ut', torch.zeros(1))
        self.register_buffer('quant_noise_vt', torch.zeros(1))
        
    def quantize(self, features: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """Apply quantization noise simulation during training."""
        if self.training:
            noise = torch.rand_like(features) - 0.5  # U(-0.5, 0.5)
            return features + noise * noise_scale
        else:
            # At inference, round to nearest quantization level
            levels = 2 ** self.config.quantization_bits
            return torch.round(features * (levels - 1)) / (levels - 1)


# =============================================================================
# MLP Decoder
# =============================================================================

class MLPDecoder(nn.Module):
    """
    Lightweight MLP decoder G_Phi as described in Section 3.3.
    
    Inputs:
    - V_uvt: sampled 3D feature vector
    - V_uv, V_ut, V_vt: sampled tri-plane feature vectors
    - gamma(t): time encoding
    
    Output: reconstructed lighting I(u, v, t)
    """
    def __init__(self, config: NDGIConfig):
        super().__init__()
        self.config = config
        
        # Calculate input dimension
        time_dim = 2 * config.time_freq_bands  # sin/cos pairs
        feature_dim = (
            config.f3d_uvt_channels + 
            config.f2d_uv_channels + 
            config.f2d_ut_channels + 
            config.f2d_vt_channels + 
            time_dim
        )
        
        # Build MLP layers
        layers = []
        in_dim = feature_dim
        
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_size))
            layers.append(nn.GELU())  # Paper uses GELU activation
            in_dim = config.hidden_size
        
        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, config.output_channels))
        
        self.network = nn.Sequential(*layers)
        
        # Time encoding
        self.time_encoding = TimeEncoding(config.time_freq_bands)
        
    def forward(
        self, 
        v_uvt: torch.Tensor,
        v_uv: torch.Tensor,
        v_ut: torch.Tensor,
        v_vt: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            v_uvt: (B, C3) 3D feature samples
            v_uv: (B, C2_uv) 2D spatial feature samples
            v_ut: (B, C2_ut) ut-plane feature samples
            v_vt: (B, C2_vt) vt-plane feature samples
            t: (B, 1) time values
        Returns:
            I(u, v, t): (B, 3) reconstructed lighting
        """
        # Encode time
        t_enc = self.time_encoding(t)  # (B, time_dim)
        
        # Concatenate all features
        features = torch.cat([v_uvt, v_uv, v_ut, v_vt, t_enc], dim=-1)
        
        # Decode
        return self.network(features)
    
    def count_parameters(self) -> int:
        """Count total number of parameters in the decoder."""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# NDGI Model
# =============================================================================

class NDGIModel(nn.Module):
    """
    Complete NDGI model as described in the paper.
    
    Compresses temporal lightmap set L into compact parameters Theta:
    L -> Theta = {F3D_uvt, F2D_uv, F2D_ut, F2D_vt, Phi}
    
    At inference:
    I(u, v, t) = G_Phi(V_uvt, V_uv, V_ut, V_vt, gamma(t))
    """
    def __init__(self, config: NDGIConfig):
        super().__init__()
        self.config = config
        
        # Feature maps
        self.features = HybridFeatureMaps(config)
        
        # Decoder
        self.decoder = MLPDecoder(config)
        
        # BC compressor (for training simulation)
        self.bc_compressor = BCFeatureMapCompressor(config)
        
    def sample_features(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample feature maps at given coordinates.
        
        Args:
            u, v: Spatial coordinates in [-1, 1], shape (B, 1)
            t: Time coordinate in [0, 1], shape (B, 1)
        Returns:
            V_uvt, V_uv, V_ut, V_vt: sampled feature vectors
        """
        B = u.shape[0]
        
        # Prepare grid for sampling: (B, 1, 1, 2 or 3)
        # For 2D: grid is (B, H_out, W_out, 2) with (x, y) = (u, v)
        # For 3D: grid is (B, D_out, H_out, W_out, 3) with (x, y, z) = (u, v, t)
        
        # Sample 2D spatial feature F2D_uv
        # Expand feature map from (1, C, H, W) to (B, C, H, W) to match grid batch size
        f_uv = self.features.f2d_uv.expand(B, -1, -1, -1)  # (B, C, H, W)
        grid_uv = torch.cat([u, v], dim=-1).view(B, 1, 1, 2)  # (B, 1, 1, 2)
        v_uv = F.grid_sample(
            f_uv, 
            grid_uv, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        ).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Sample 3D feature F3D_uvt
        # Expand feature map from (1, C, D, H, W) to (B, C, D, H, W)
        f_uvt = self.features.f3d_uvt.expand(B, -1, -1, -1, -1)  # (B, C, D, H, W)
        grid_uvt = torch.cat([u, v, t], dim=-1).view(B, 1, 1, 1, 3)
        v_uvt = F.grid_sample(
            f_uvt,
            grid_uvt,
            mode='bilinear',  # bilinear on 5D input = trilinear interpolation
            padding_mode='border',
            align_corners=False
        )
        # Squeeze spatial dims: (B, C, 1, 1, 1) → (B, C)
        v_uvt = v_uvt.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Sample ut-plane: coordinates (u, t)
        # Expand feature map from (1, C, H, W) to (B, C, H, W)
        f_ut = self.features.f2d_ut.expand(B, -1, -1, -1)  # (B, C, H, W)
        grid_ut = torch.cat([u, t], dim=-1).view(B, 1, 1, 2)
        v_ut = F.grid_sample(
            f_ut,
            grid_ut,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze(-1).squeeze(-1)  # (B, C)
        
        # Sample vt-plane: coordinates (v, t)
        # Expand feature map from (1, C, H, W) to (B, C, H, W)
        f_vt = self.features.f2d_vt.expand(B, -1, -1, -1)  # (B, C, H, W)
        grid_vt = torch.cat([v, t], dim=-1).view(B, 1, 1, 2)
        v_vt = F.grid_sample(
            f_vt,
            grid_vt,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze(-1).squeeze(-1)  # (B, C)
        
        return v_uvt, v_uv, v_ut, v_vt
        
    def forward(self, u: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct lighting at given coordinates.
        
        Args:
            u, v, t: Coordinates, each shape (B, 1)
        Returns:
            I(u, v, t): Reconstructed lighting, shape (B, 3)
        """
        # Sample features
        v_uvt, v_uv, v_ut, v_vt = self.sample_features(u, v, t)
        
        # Decode
        return self.decoder(v_uvt, v_uv, v_ut, v_vt, t)
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size breakdown in MB."""
        info = self.features.get_feature_info()
        decoder_params = self.decoder.count_parameters()
        
        return {
            'feature_maps_mb': sum(v['size_mb'] for v in info.values()),
            'decoder_params': decoder_params,
            'decoder_size_mb': decoder_params * 4 / (1024 * 1024),
            'total_size_mb': sum(v['size_mb'] for v in info.values()) + decoder_params * 4 / (1024 * 1024)
        }


# =============================================================================
# Quantization Noise Injection (Section 3.4)
# =============================================================================

class QuantizationNoise(nn.Module):
    """
    Simulates quantization noise during training for 8-bit representation.
    
    Equation (6): V' = V + U(-0.5, 0.5) * alpha
    where alpha = 1/256 for 8-bit.
    """
    def __init__(self, alpha: float = 1.0 / 256.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = (torch.rand_like(features) - 0.5) * self.alpha
            return features + noise
        return features


# =============================================================================
# Dataset
# =============================================================================

class TemporalLightmapDataset:
    """
    Dataset for temporal lightmaps.
    
    Loads multiple lightmap sets L = {L_i | i = 0, 1, ..., n-1}
    where each L_i is a lightmap of size (H, W, C) at time t_i.
    """
    def __init__(
        self, 
        lightmaps: torch.Tensor,  # (n, H, W, C)
        times: torch.Tensor,       # (n,)
        mask: Optional[torch.Tensor] = None  # (H, W) binary mask
    ):
        self.lightmaps = lightmaps
        self.times = times
        self.mask = mask
        self.n = len(times)
        self.H, self.W = lightmaps.shape[1:3]
        
    def sample_pixels(self, num_samples: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample pixel coordinates and their corresponding light values.
        
        Returns:
            coords: (num_samples, 3) -> (u, v, t)
            colors: (num_samples, 3) -> RGB values
        """
        # Sample random lightmap indices (time)
        time_idx = torch.randint(0, self.n, (num_samples,))
        
        # Sample random pixel coordinates
        if self.mask is not None:
            # Sample only valid pixels
            valid_coords = torch.argwhere(self.mask > 0)
            pixel_idx = torch.randint(0, len(valid_coords), (num_samples,))
            h_coords = valid_coords[pixel_idx, 0].float()
            w_coords = valid_coords[pixel_idx, 1].float()
        else:
            h_coords = torch.randint(0, self.H, (num_samples,)).float()
            w_coords = torch.randint(0, self.W, (num_samples,)).float()
        
        # Normalize to [-1, 1] for grid_sample
        u = (w_coords / (self.W - 1)) * 2 - 1
        v = (h_coords / (self.H - 1)) * 2 - 1
        t = self.times[time_idx]
        
        # Get colors
        colors = self.lightmaps[time_idx, h_coords.long(), w_coords.long()]
        
        # Move all to same device before stacking
        coords = torch.stack([u.to(device), v.to(device), t.to(device)], dim=-1)
        colors = colors.to(device)
        
        return coords, colors


# =============================================================================
# Training
# =============================================================================

class NDGITrainer:
    """
    Training pipeline for NDGI as described in Section 4 and 5.
    
    Key steps:
    1. Apply gamma correction to enhance dark region details
    2. Per-channel mean normalization at each time step
    3. Train with Adam optimizer, L2 loss
    4. Final stage: freeze feature maps, fine-tune MLP under quantization + BC
    """
    def __init__(
        self, 
        model: NDGIModel, 
        config: NDGIConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.quant_noise = QuantizationNoise(config.alpha_noise).to(device)
        
        # Optimizer: Adam with lr=1e-3 (paper)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    def apply_gamma_correction(self, colors: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
        """Apply gamma correction to enhance dark region details."""
        return torch.pow(colors, 1.0 / gamma)
    
    def normalize_per_channel(self, colors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-channel mean normalization.
        Returns normalized colors and mean values for later restoration.
        """
        mean = colors.mean(dim=0, keepdim=True)
        return colors - mean, mean
        
    def train_step(self, coords: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            coords: (B, 3) -> (u, v, t)
            targets: (B, 3) -> RGB light values
        Returns:
            loss value
        """
        self.optimizer.zero_grad()
        
        u = coords[:, 0:1]
        v = coords[:, 1:2]
        t = coords[:, 2:3]
        
        # Forward pass
        pred = self.model(u, v, t)
        
        # L2 loss (MSE)
        loss = F.mse_loss(pred, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_with_quantization_noise(
        self, 
        coords: torch.Tensor, 
        targets: torch.Tensor
    ) -> float:
        """
        Training step with quantization noise injection (Section 3.4).
        """
        self.optimizer.zero_grad()
        
        u = coords[:, 0:1]
        v = coords[:, 1:2]
        t = coords[:, 2:3]
        
        # Sample features and add noise
        v_uvt, v_uv, v_ut, v_vt = self.model.sample_features(u, v, t)
        
        # Add quantization noise to simulate 8-bit (Equation 6)
        v_ut = self.quant_noise(v_ut)
        v_vt = self.quant_noise(v_vt)
        
        # Decode
        pred = self.model.decoder(v_uvt, v_uv, v_ut, v_vt, t)
        
        loss = F.mse_loss(pred, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataset: TemporalLightmapDataset, num_batches: int = 100) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        for _ in range(num_batches):
            coords, targets = dataset.sample_pixels(self.config.batch_size, self.device)
            loss = self.train_step(coords, targets)
            total_loss += loss
        return total_loss / num_batches
    
    def freeze_features_finetune_mlp(self, dataset: TemporalLightmapDataset, epochs: int = 10):
        """
        Final training stage: freeze feature maps, fine-tune MLP under 
        simulated quantization and BC compression (Section 4).
        """
        # Freeze feature maps
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Only optimize decoder
        optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=self.config.learning_rate * 0.1)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for _ in range(100):
                coords, targets = dataset.sample_pixels(self.config.batch_size, self.device)
                
                optimizer.zero_grad()
                u, v, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
                
                # Add quantization noise during fine-tuning
                v_uvt, v_uv, v_ut, v_vt = self.model.sample_features(u, v, t)
                v_ut = self.quant_noise(v_ut)
                v_vt = self.quant_noise(v_vt)
                
                pred = self.model.decoder(v_uvt, v_uv, v_ut, v_vt, t)
                loss = F.mse_loss(pred, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Fine-tune Epoch {epoch+1}/{epochs}, Loss: {total_loss/100:.6f}")


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D circular-symmetric Gaussian kernel (standard SSIM: sigma=1.5, 11x11)."""
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g.unsqueeze(1) * g.unsqueeze(0)
    g = g / g.sum()
    return g.view(1, 1, size, size)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11, data_range: float = 1.0,
                 sigma: float = 1.5) -> float:
    """Standard SSIM with Gaussian kernel."""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    kernel = _gaussian_kernel(window_size, sigma, img1.device)
    pad = window_size // 2

    # (H, W, C) -> (C, 1, H, W)
    if img1.dim() == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(1)
        img2 = img2.permute(2, 0, 1).unsqueeze(1)

    vals = []
    for c in range(img1.shape[0]):
        ch1, ch2 = img1[c:c + 1], img2[c:c + 1]
        mu1  = F.conv2d(ch1,       kernel, padding=pad)
        mu2  = F.conv2d(ch2,       kernel, padding=pad)
        mu11 = F.conv2d(ch1 * ch1, kernel, padding=pad)
        mu22 = F.conv2d(ch2 * ch2, kernel, padding=pad)
        mu12 = F.conv2d(ch1 * ch2, kernel, padding=pad)

        sigma1  = mu11 - mu1 ** 2
        sigma2  = mu22 - mu2 ** 2
        sigma12 = mu12 - mu1 * mu2

        s = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
            ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2) + 1e-8)
        vals.append(s.mean().item())
    return float(np.mean(vals))


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute PSNR, MSE, and MAE."""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    psnr = compute_psnr(pred, target)
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr
    }


# =============================================================================
# Virtual Texturing Integration (Section 3.5)
# =============================================================================

class TileBasedNDGI:
    """
    Tile-based NDGI for virtual texturing integration.
    
    Each lightmap is partitioned into fixed-size tiles (e.g., 128x128),
    with a dedicated NDGI model per tile. At runtime, only visible tiles
    are decompressed on-demand.
    """
    def __init__(self, tile_size: int = 128, config: NDGIConfig = None):
        self.tile_size = tile_size
        self.config = config or NDGIConfig()
        self.tile_models: Dict[Tuple[int, int], NDGIModel] = {}
        
    def create_tile_model(self, tile_x: int, tile_y: int) -> NDGIModel:
        """Create a new NDGI model for a specific tile."""
        model = NDGIModel(self.config)
        self.tile_models[(tile_x, tile_y)] = model
        return model
    
    def get_tile_model(self, tile_x: int, tile_y: int) -> NDGIModel:
        """Get or create model for a tile."""
        if (tile_x, tile_y) not in self.tile_models:
            return self.create_tile_model(tile_x, tile_y)
        return self.tile_models[(tile_x, tile_y)]
    
    def reconstruct_tile(
        self, 
        tile_x: int, 
        tile_y: int, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct lighting for a specific tile."""
        model = self.get_tile_model(tile_x, tile_y)
        return model(u, v, t)


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(lightmaps_raw: torch.Tensor, pred_images: List[torch.Tensor],
                      loss_hist: List[float], psnr_list: List[float], ssim_list: List[float],
                      model: NDGIModel, config: NDGIConfig, device: str):
    """
    Generate comprehensive visualization of NDGI results.
    
    Layout (2x3 grid):
      Top-left  : GT vs Pred lightmap comparison
      Top-right : Training loss curve
      Bottom-left:  PSNR bar chart per time step
      Bottom-mid :  SSIM bar chart per time step
      Bottom-right: Model info & compression ratio
    """
    T = lightmaps_raw.shape[0]
    H = lightmaps_raw.shape[1]
    W = lightmaps_raw.shape[2]
    
    # ── Compute compression ratio ──
    # Raw lightmap data size (float32 storage)
    raw_lightmap_bytes = T * H * W * 3 * 4  # T frames, HxW pixels, 3 channels, 4 bytes/float
    raw_lightmap_elems = T * H * W * 3
    
    # NDGI model: feature map elements (float32) + decoder (float32)
    feature_info = model.features.get_feature_info()
    total_params = sum(p.numel() for p in model.parameters())
    decoder_params = model.decoder.count_parameters()
    feature_params = total_params - decoder_params
    
    # BC compressed size (bits → bytes)
    bc_uv_bits = model.bc_compressor.bc_f2d_uv.get_compressed_size()
    bc_uvt_bits = model.bc_compressor.bc_f3d.get_compressed_size()
    bc_uv_bytes = bc_uv_bits / 8
    bc_uvt_bytes = bc_uvt_bits / 8
    
    # 8-bit quantized feature maps
    quant_ut_bytes = model.features.f2d_ut.numel()  # 1 byte per element
    quant_vt_bytes = model.features.f2d_vt.numel()   # 1 byte per element
    
    # Decoder (float32)
    decoder_bytes = decoder_params * 4
    
    # Total NDGI compressed size
    ndgi_bc_bytes = bc_uv_bytes + bc_uvt_bytes + quant_ut_bytes + quant_vt_bytes + decoder_bytes
    ndgi_raw_bytes = feature_params * 4 + decoder_bytes  # Without BC compression
    
    # Ratios
    ratio_bc = raw_lightmap_bytes / ndgi_bc_bytes
    ratio_raw = raw_lightmap_bytes / ndgi_raw_bytes
    ratio_feature_bc = (feature_params * 4) / (bc_uv_bytes + bc_uvt_bytes + quant_ut_bytes + quant_vt_bytes)
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1, 1],
                  hspace=0.45, wspace=0.35)

    # ---- Top-left: GT vs Pred ----
    gsl = GridSpecFromSubplotSpec(2, T, subplot_spec=gs[0, :2],
                                   wspace=0.1, hspace=0.2)
    for ti in range(T):
        for row, (label, data) in enumerate([('GT', lightmaps_raw), ('Pred', pred_images)]):
            ax = fig.add_subplot(gsl[row, ti])
            ax.imshow(np.clip(data[ti].cpu().numpy(), 0, 1))
            ax.set_title(f'{label} t={ti}', fontsize=9)
            ax.axis('off')

    # ---- Top-right: Loss curve ----
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(loss_hist, color='#2c3e50', linewidth=0.8)
    ax.set_title('Training Loss', fontweight='bold')
    ax.set_xlabel('Batch'); ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)

    # ---- Bottom-left: PSNR ----
    ax = fig.add_subplot(gs[1, 0])
    c_psnr = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    bars = ax.bar(range(T), psnr_list, color=c_psnr[:T])
    ax.set_title('PSNR per Time Step', fontweight='bold')
    ax.set_xticks(range(T)); ax.set_ylabel('dB')
    for b, v in zip(bars, psnr_list):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f'{v:.1f}', ha='center', fontsize=9)

    # ---- Bottom-mid: SSIM ----
    ax = fig.add_subplot(gs[1, 1])
    c_ssim = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(range(T), ssim_list, color=c_ssim[:T])
    ax.set_title('SSIM per Time Step', fontweight='bold')
    ax.set_xticks(range(T)); ax.set_ylim(0, 1)
    for b, v in zip(bars, ssim_list):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f'{v:.3f}', ha='center', fontsize=9)

    # ---- Bottom-right: Model info + Compression ratio ----
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    
    # Format sizes nicely
    def fmt_bytes(b):
        if b >= 1024 * 1024:
            return f"{b / (1024*1024):.2f} MB"
        elif b >= 1024:
            return f"{b / 1024:.1f} KB"
        else:
            return f"{b} B"
    
    txt = (
        "NDGI Model Info\n"
        "================\n"
        f"Feature Maps (float32):\n"
    )
    for name, info in feature_info.items():
        txt += f"  {name:8s}: {info['shape']}  {info['numel']:6d}\n"
    txt += (
        f"\nDecoder (MLP+PE):\n"
        f"  Hidden:    {config.hidden_size}\n"
        f"  Layers:    {config.num_hidden_layers}\n"
        f"  PE freq:   {config.time_freq_bands} -> {2*config.time_freq_bands} dims\n"
        f"  Params:    {decoder_params}\n\n"
        f"Compression Ratio\n"
        "----------------\n"
        f"Raw Lightmap:  {fmt_bytes(raw_lightmap_bytes)}\n"
        f"  ({T}x{H}x{W}x3, float32)\n\n"
        f"NDGI (no BC):  {fmt_bytes(ndgi_raw_bytes)}\n"
        f"NDGI (BC+8b):  {fmt_bytes(ndgi_bc_bytes)}\n"
        f"  BC F_uv:   {fmt_bytes(bc_uv_bytes)}\n"
        f"  BC F_uvt:  {fmt_bytes(bc_uvt_bytes)}\n"
        f"  Quant ut:  {fmt_bytes(quant_ut_bytes)}\n"
        f"  Quant vt:  {fmt_bytes(quant_vt_bytes)}\n"
        f"  Decoder:   {fmt_bytes(decoder_bytes)}\n\n"
        f"Ratio (no BC):  {ratio_raw:.2f} : 1\n"
        f"Ratio (BC+8b):  {ratio_bc:.2f} : 1\n"
    )
    ax.text(0.5, 0.5, txt, transform=ax.transAxes,
            fontsize=8.5, va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    fig.suptitle('NDGI \u2014 Results Summary', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig('ndgi_implementation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nVisualization saved -> ndgi_implementation_results.png")
    print(f"\nCompression Ratio Summary:")
    print(f"  Raw lightmap:      {fmt_bytes(raw_lightmap_bytes)} ({T}x{H}x{W}x3 float32)")
    print(f"  NDGI (no BC):      {fmt_bytes(ndgi_raw_bytes)}   Ratio: {ratio_raw:.2f}:1")
    print(f"  NDGI (BC+8bit):    {fmt_bytes(ndgi_bc_bytes)}   Ratio: {ratio_bc:.2f}:1")


# =============================================================================
# Main / Demo
# =============================================================================

def generate_demo_lightmaps(num_frames: int = 4, H: int = 128, W: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate temporally-varying lightmaps: moving Gaussian spot + checkerboard.
    Returns: lightmaps (T, H, W, 3) and times (T,) in [0, 1]
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    
    for i, t_val in enumerate(times):
        # Checkerboard background
        bg = ((torch.arange(H).unsqueeze(1) + torch.arange(W)) % 2).float() * 0.2
        # Moving Gaussian spot
        center_u = 0.3 + 0.4 * t_val
        center_v = 0.5
        u = torch.linspace(0, 1, W).view(1, -1).expand(H, W)
        v = torch.linspace(0, 1, H).view(-1, 1).expand(H, W)
        spot = torch.exp(-((u - center_u) ** 2 + (v - center_v) ** 2) / 0.02)
        img = bg + spot * 0.8
        lightmaps[i] = img.unsqueeze(-1).repeat(1, 1, 3)
    
    return lightmaps, times


def demo(signal_name: Optional[str] = 'multi_light'):
    """Demonstration of NDGI model with training, evaluation, and visualization.
    
    Args:
        signal_name: 测试信号名称, 可选值见 SIGNAL_REGISTRY.keys()
                     None 则使用原始 generate_demo_lightmaps
    """
    print("=" * 60)
    print("Neural Dynamic GI (NDGI) - PyTorch Implementation")
    print("=" * 60)
    
    # ── Configuration ──
    cfg = {
        'data': {
            'signal_name': signal_name,
        },
        'feature': {
            'f3d_uvt_resolution': (32, 32, 12),
            'f2d_uv_resolution': (128, 128),
            'f2d_ut_resolution': (64, 24),
            'f2d_vt_resolution': (64, 24),
            'f2d_uv_channels': 4,
            'f2d_ut_channels': 2,
            'f2d_vt_channels': 2,
            'f3d_uvt_channels': 4,
        },
        'decoder': {
            'hidden_size': 16,
            'num_hidden_layers': 2,
            'time_freq_bands': 2,
        },
        'train': {
            'learning_rate': 1e-3,
            'batch_size': 2048,
            'quantization_bits': 8,
            'alpha_noise': 1.0 / 256.0,
            'bc_block_size': 4,
            'tile_size': 128,
            'epochs_phase1': 2000,
            'epochs_phase2': 500,
            'output_channels': 3,
        },
    }
    
    config = NDGIConfig(
        f3d_uvt_resolution=cfg['feature']['f3d_uvt_resolution'],
        f2d_uv_resolution=cfg['feature']['f2d_uv_resolution'],
        f2d_ut_resolution=cfg['feature']['f2d_ut_resolution'],
        f2d_vt_resolution=cfg['feature']['f2d_vt_resolution'],
        f2d_uv_channels=cfg['feature']['f2d_uv_channels'],
        f2d_ut_channels=cfg['feature']['f2d_ut_channels'],
        f2d_vt_channels=cfg['feature']['f2d_vt_channels'],
        f3d_uvt_channels=cfg['feature']['f3d_uvt_channels'],
        hidden_size=cfg['decoder']['hidden_size'],
        num_hidden_layers=cfg['decoder']['num_hidden_layers'],
        time_freq_bands=cfg['decoder']['time_freq_bands'],
        batch_size=cfg['train']['batch_size'],
        tile_size=cfg['train']['tile_size'],
        output_channels=cfg['train']['output_channels'],
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    model = NDGIModel(config).to(device)
    
    # ── Model info ──
    print("\n--- Model Info ---")
    size_info = model.get_model_size()
    for key, value in size_info.items():
        print(f"  {key}: {value:.4f} MB")
    print(f"  Decoder params: {model.decoder.count_parameters()}")
    print(f"  Total params:   {sum(p.numel() for p in model.parameters())}")
    
    # ── Generate data ──
    print("\n--- Generating Lightmaps ---")
    T_frames = config.f3d_uvt_resolution[2]  # Use temporal dim as frame count
    H, W = config.f2d_uv_resolution
    
    signal_name = cfg['data']['signal_name']
    if signal_name is not None and signal_name in SIGNAL_REGISTRY:
        # 使用测试信号集
        desc = SIGNAL_DESCRIPTIONS[signal_name]
        print(f"  Signal: {signal_name} — {desc}")
        lightmaps, times = SIGNAL_REGISTRY[signal_name](num_frames=T_frames, H=H, W=W)
    else:
        if signal_name is not None:
            print(f"  Warning: unknown signal '{signal_name}', using default")
            print(f"  Available: {list(SIGNAL_REGISTRY.keys())}")
        # 使用原始简单信号
        lightmaps, times = generate_demo_lightmaps(num_frames=T_frames, H=H, W=W)
    lightmaps = lightmaps.to(device)
    times = times.to(device)
    print(f"  Lightmaps: {list(lightmaps.shape)}, Times: {list(times.shape)}")
    
    # ── Train: Phase 1 (全参数) + Phase 2 (冻结特征图, fine-tune MLP) ──
    print("\n--- Training ---")
    dataset = TemporalLightmapDataset(lightmaps, times)
    trainer = NDGITrainer(model, config, device)
    loss_hist = []
    
    epochs_phase1 = cfg['train']['epochs_phase1']
    epochs_phase2 = cfg['train']['epochs_phase2']
    total_epochs = epochs_phase1 + epochs_phase2
    
    # Phase 1 scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=total_epochs)
    
    for epoch in range(total_epochs):
        # ── Phase 2 transition: freeze feature maps, fine-tune MLP ──
        if epoch == epochs_phase1:
            print("\n>>> Phase 2: freeze feature maps, fine-tune MLP decoder")
            for n, p in model.named_parameters():
                if 'decoder' not in n:  # freeze everything except MLP decoder
                    p.requires_grad = False
            # Re-create optimizer for decoder only
            trainer.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate * 0.1
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=epochs_phase2)
        
        coords, targets = dataset.sample_pixels(config.batch_size, device)
        loss = trainer.train_step(coords, targets)
        loss_hist.append(loss)
        
        if (epoch + 1) % 200 == 0:
            ph = "P1" if epoch < epochs_phase1 else "P2"
            lr_val = scheduler.get_last_lr()[0]
            print(f"  [{ph}] Epoch {epoch+1:4d} | Loss = {loss:.6f} | LR = {lr_val:.2e}")
        
        scheduler.step()
    
    # ── Evaluate: full-frame rendering ──
    print("\n--- Evaluation ---")
    model.eval()
    
    # Build (u, v) grid for full image
    u_lin = torch.linspace(-1, 1, W, device=device)
    v_lin = torch.linspace(-1, 1, H, device=device)
    uu, vv = torch.meshgrid(u_lin, v_lin, indexing='xy')
    u_flat = uu.flatten().unsqueeze(-1)  # (N, 1) in [-1, 1]
    v_flat = vv.flatten().unsqueeze(-1)  # (N, 1) in [-1, 1]
    N = H * W
    
    pred_images = []
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for ti in range(T_frames):
            t_val = times[ti]
            t_tensor = torch.full((N, 1), t_val.item(), device=device)
            pred = model(u_flat, v_flat, t_tensor)  # (N, 3)
            pred_img = pred.view(H, W, 3)
            pred_clipped = torch.clamp(pred_img, 0, 1)
            pred_images.append(pred_clipped)
            
            gt = lightmaps[ti]  # (H, W, 3)
            psnr_val = compute_psnr(pred_clipped, gt)
            ssim_val = compute_ssim(pred_clipped, gt, window_size=min(11, min(H, W) // 8 * 2 + 1))
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            print(f"  t={ti}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")
    
    # ── Visualize ──
    visualize_results(lightmaps, pred_images, loss_hist, psnr_list, ssim_list,
                      model, config, device)
    
    print(f"\nAvg PSNR: {np.mean(psnr_list):.2f} dB, Avg SSIM: {np.mean(ssim_list):.4f}")
    
    return model, trainer, {'psnr': psnr_list, 'ssim': ssim_list, 'loss_hist': loss_hist}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDGI Demo — 神经动态全局光照")
    parser.add_argument(
        '--signal', type=str, default='checkerboard_shadow',
        help=f"测试信号名称, 可选: {list(SIGNAL_REGISTRY.keys())}. "
             f"默认使用 moving_point_light。"
    )
    parser.add_argument(
        '--list-signals', action='store_true',
        help="列出所有可用的测试信号"
    )
    args = parser.parse_args()
    
    if args.list_signals:
        print("可用测试信号:")
        for name, desc in SIGNAL_DESCRIPTIONS.items():
            print(f"  {name:20s} — {desc}")
        exit(0)
    
    model, trainer, metrics = demo(signal_name=args.signal)
