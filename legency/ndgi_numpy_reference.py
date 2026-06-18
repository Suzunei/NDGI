"""
Neural Dynamic GI (NDGI) - NumPy Reference Implementation
=============================================================
Pure NumPy implementation of NDGI core concepts for educational purposes.
This version demonstrates the algorithm without PyTorch dependencies.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# =============================================================================
# 1. Time Encoding (Equation 4 in paper)
# =============================================================================

def time_encoding(t: float, freq_bands: int = 2) -> np.ndarray:
    """
    Positional encoding for time using sinusoidal functions.
    
    gamma(t) = [sin(2^0*pi*t), cos(2^0*pi*t), sin(2^1*pi*t), cos(2^1*pi*t), ...]
    
    Args:
        t: Time value in [0, 1]
        freq_bands: Number of frequency bands
    Returns:
        Encoded time vector of shape (2 * freq_bands,)
    """
    encodings = []
    for i in range(freq_bands):
        freq = 2 ** i * np.pi
        encodings.append(np.sin(freq * t))
        encodings.append(np.cos(freq * t))
    return np.array(encodings)


# =============================================================================
# 2. Hybrid Feature Map Sampling
# =============================================================================

def sample_2d_feature(feature_map: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    Sample a 2D feature map at normalized coordinates (u, v) in [-1, 1].
    
    Args:
        feature_map: (H, W, C) array
        u, v: Coordinates in [-1, 1]
    Returns:
        Feature vector of shape (C,)
    """
    H, W, C = feature_map.shape
    # Convert from [-1, 1] to pixel coordinates
    x = (u + 1) / 2 * (W - 1)
    y = (v + 1) / 2 * (H - 1)
    
    # Bilinear interpolation
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    
    dx, dy = x - x0, y - y0
    
    # Interpolate
    val = (1 - dx) * (1 - dy) * feature_map[y0, x0] + \
          dx * (1 - dy) * feature_map[y0, x1] + \
          (1 - dx) * dy * feature_map[y1, x0] + \
          dx * dy * feature_map[y1, x1]
    
    return val


def sample_3d_feature(feature_map: np.ndarray, u: float, v: float, t: float) -> np.ndarray:
    """
    Sample a 3D feature map at normalized coordinates (u, v, t) in [-1, 1] and [0, 1].
    
    Args:
        feature_map: (D, H, W, C) array where D is temporal dimension
        u, v: Spatial coordinates in [-1, 1]
        t: Time coordinate in [0, 1]
    Returns:
        Feature vector of shape (C,)
    """
    D, H, W, C = feature_map.shape
    # Convert to voxel coordinates
    x = (u + 1) / 2 * (W - 1)
    y = (v + 1) / 2 * (H - 1)
    z = t * (D - 1)
    
    # Trilinear interpolation (simplified)
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    z1 = min(z0 + 1, D - 1)
    
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # Interpolate
    c000 = feature_map[z0, y0, x0]
    c100 = feature_map[z0, y0, x1]
    c010 = feature_map[z0, y1, x0]
    c110 = feature_map[z0, y1, x1]
    c001 = feature_map[z1, y0, x0]
    c101 = feature_map[z1, y0, x1]
    c011 = feature_map[z1, y1, x0]
    c111 = feature_map[z1, y1, x1]
    
    val = (1 - dx) * (1 - dy) * (1 - dz) * c000 + \
          dx * (1 - dy) * (1 - dz) * c100 + \
          (1 - dx) * dy * (1 - dz) * c010 + \
          dx * dy * (1 - dz) * c110 + \
          (1 - dx) * (1 - dy) * dz * c001 + \
          dx * (1 - dy) * dz * c101 + \
          (1 - dx) * dy * dz * c011 + \
          dx * dy * dz * c111
    
    return val


# =============================================================================
# 3. Lightweight MLP Decoder
# =============================================================================

class MLPDecoder:
    """
    Simple MLP decoder with GELU activation.
    
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Initialize weights with small random values
        self.weights = []
        self.biases = []
        
        dims = [input_dim] + [hidden_size] * num_layers + [output_dim]
        for i in range(len(dims) - 1):
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * 0.01)
            self.biases.append(np.zeros(dims[i+1]))
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input vector of shape (input_dim,)
        Returns:
            Output vector of shape (output_dim,)
        """
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W + b
            if i < len(self.weights) - 1:  # Apply activation to hidden layers
                x = self.gelu(x)
        return x


# =============================================================================
# 4. BC Compression Simulation (Section 3.4)
# =============================================================================

class BCSimulation:
    """
    Block Compression simulation using endpoints and weights.
    
    BC7: Each 4x4 block stores two endpoints e1, e2 and per-pixel weights.
    f_p = (1 - w_p) * e1 + w_p * e2
    """
    def __init__(self, height: int, width: int, channels: int, block_size: int = 4):
        self.height = height
        self.width = width
        self.channels = channels
        self.block_size = block_size
        
        self.num_blocks_h = height // block_size
        self.num_blocks_w = width // block_size
        self.num_blocks = self.num_blocks_h * self.num_blocks_w
        
        # Initialize endpoints and weights
        self.endpoints = np.random.rand(self.num_blocks, 2, channels)
        self.weights = np.random.rand(self.num_blocks, block_size * block_size)
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct feature map from BC endpoints and weights."""
        feature_map = np.zeros((self.height, self.width, self.channels))
        
        for b in range(self.num_blocks):
            bh = b // self.num_blocks_w
            bw = b % self.num_blocks_w
            
            e1 = self.endpoints[b, 0]
            e2 = self.endpoints[b, 1]
            
            for p in range(self.block_size * self.block_size):
                ph = p // self.block_size
                pw = p % self.block_size
                
                w = self.weights[b, p]
                f = (1 - w) * e1 + w * e2
                
                y = bh * self.block_size + ph
                x = bw * self.block_size + pw
                feature_map[y, x] = f
        
        return feature_map
    
    def compressed_size_bits(self, bits_per_endpoint: int = 8, bits_per_weight: int = 8) -> int:
        """Calculate compressed size in bits."""
        endpoint_bits = self.num_blocks * 2 * self.channels * bits_per_endpoint
        weight_bits = self.num_blocks * (self.block_size ** 2) * bits_per_weight
        return endpoint_bits + weight_bits


# =============================================================================
# 5. Quantization Noise Injection (Equation 6)
# =============================================================================

def add_quantization_noise(features: np.ndarray, alpha: float = 1.0 / 256.0) -> np.ndarray:
    """
    Simulate quantization noise for 8-bit representation.
    
    V' = V + U(-0.5, 0.5) * alpha
    where alpha = 1/256 for 8-bit.
    """
    noise = (np.random.rand(*features.shape) - 0.5) * alpha
    return features + noise


# =============================================================================
# 6. Complete NDGI Inference Pipeline
# =============================================================================

class NDGIInference:
    """
    Complete NDGI inference pipeline for reconstructing temporal lightmaps.
    """
    def __init__(
        self,
        f2d_uv: np.ndarray,      # (H, W, C_uv)
        f2d_ut: np.ndarray,       # (H_ut, T_ut, C_ut)
        f2d_vt: np.ndarray,       # (H_vt, T_vt, C_vt)
        f3d_uvt: np.ndarray,      # (D, H, W, C_3d)
        decoder: MLPDecoder,
        time_freq_bands: int = 2
    ):
        self.f2d_uv = f2d_uv
        self.f2d_ut = f2d_ut
        self.f2d_vt = f2d_vt
        self.f3d_uvt = f3d_uvt
        self.decoder = decoder
        self.time_freq_bands = time_freq_bands
    
    def reconstruct(self, u: float, v: float, t: float) -> np.ndarray:
        """
        Reconstruct lighting at position (u, v) and time t.
        
        I(u, v, t) = G_Phi(V_uvt, V_uv, V_ut, V_vt, gamma(t))
        
        Args:
            u, v: Spatial coordinates in [-1, 1]
            t: Time in [0, 1]
        Returns:
            RGB lighting vector of shape (3,)
        """
        # Sample all feature maps
        V_uv = sample_2d_feature(self.f2d_uv, u, v)
        V_ut = sample_2d_feature(self.f2d_ut, u, t)
        V_vt = sample_2d_feature(self.f2d_vt, v, t)
        V_uvt = sample_3d_feature(self.f3d_uvt, u, v, t)
        
        # Encode time
        gamma_t = time_encoding(t, self.time_freq_bands)
        
        # Concatenate all inputs
        features = np.concatenate([V_uvt, V_uv, V_ut, V_vt, gamma_t])
        
        # Decode
        return self.decoder.forward(features)
    
    def reconstruct_lightmap(self, H: int, W: int, t: float) -> np.ndarray:
        """
        Reconstruct entire lightmap at time t.
        
        Args:
            H, W: Resolution
            t: Time value
        Returns:
            Lightmap of shape (H, W, 3)
        """
        lightmap = np.zeros((H, W, 3))
        for y in range(H):
            for x in range(W):
                u = x / (W - 1) * 2 - 1
                v = y / (H - 1) * 2 - 1
                lightmap[y, x] = self.reconstruct(u, v, t)
        return lightmap


# =============================================================================
# 7. Training Simulation (Simplified)
# =============================================================================

def train_ndgi_simplified(
    lightmaps: np.ndarray,  # (n, H, W, 3)
    times: np.ndarray,      # (n,)
    num_iterations: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 256
) -> Dict:
    """
    Simplified training of NDGI parameters.
    
    This is a conceptual demonstration - in practice you would use
    PyTorch with automatic differentiation.
    """
    n, H, W, C = lightmaps.shape
    
    # Initialize feature maps (small random values)
    f2d_uv = np.random.randn(H, W, 4) * 0.01
    f2d_ut = np.random.randn(64, n, 2) * 0.01
    f2d_vt = np.random.randn(64, n, 2) * 0.01
    f3d_uvt = np.random.randn(12, 32, 32, 4) * 0.01
    
    # Initialize decoder
    feature_dim = 4 + 4 + 2 + 2 + 4  # V_uvt + V_uv + V_ut + V_vt + time_enc
    decoder = MLPDecoder(feature_dim, hidden_size=16, output_dim=3, num_layers=2)
    
    # Create inference object
    ndgi = NDGIInference(f2d_uv, f2d_ut, f2d_vt, f3d_uvt, decoder)
    
    losses = []
    for iteration in range(num_iterations):
        # Sample random pixels
        idx = np.random.randint(0, n, batch_size)
        y = np.random.randint(0, H, batch_size)
        x = np.random.randint(0, W, batch_size)
        
        batch_loss = 0.0
        for i in range(batch_size):
            t = times[idx[i]]
            u = x[i] / (W - 1) * 2 - 1
            v = y[i] / (H - 1) * 2 - 1
            
            pred = ndgi.reconstruct(u, v, t)
            target = lightmaps[idx[i], y[i], x[i]]
            
            loss = np.mean((pred - target) ** 2)
            batch_loss += loss
        
        batch_loss /= batch_size
        losses.append(batch_loss)
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {batch_loss:.6f}")
    
    return {
        'ndgi': ndgi,
        'losses': losses,
        'final_loss': losses[-1] if losses else None
    }


# =============================================================================
# 8. Evaluation Metrics
# =============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Simplified SSIM computation."""
    # Simplified version - full SSIM requires Gaussian filtering
    mu_pred = pred.mean()
    mu_target = target.mean()
    
    sigma_pred = pred.std()
    sigma_target = target.std()
    
    cov = np.mean((pred - mu_pred) * (target - mu_target))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = (2 * mu_pred * mu_target + c1) * (2 * cov + c2) / \
           ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2))
    
    return ssim


# =============================================================================
# 9. Tile-Based Virtual Texturing (Section 3.5)
# =============================================================================

class TileBasedNDGI:
    """
    Tile-based NDGI for virtual texturing.
    Each tile has its own NDGI model.
    """
    def __init__(self, tile_size: int = 128):
        self.tile_size = tile_size
        self.tile_models = {}  # (tile_x, tile_y) -> NDGIInference
    
    def add_tile(self, tile_x: int, tile_y: int, ndgi: NDGIInference):
        self.tile_models[(tile_x, tile_y)] = ndgi
    
    def get_tile(self, tile_x: int, tile_y: int) -> NDGIInference:
        return self.tile_models.get((tile_x, tile_y))
    
    def reconstruct(self, global_x: int, global_y: int, t: float) -> np.ndarray:
        """
        Reconstruct at global coordinates by routing to appropriate tile.
        """
        tile_x = global_x // self.tile_size
        tile_y = global_y // self.tile_size
        local_x = global_x % self.tile_size
        local_y = global_y % self.tile_size
        
        ndgi = self.get_tile(tile_x, tile_y)
        if ndgi is None:
            return np.zeros(3)
        
        u = local_x / (self.tile_size - 1) * 2 - 1
        v = local_y / (self.tile_size - 1) * 2 - 1
        return ndgi.reconstruct(u, v, t)


# =============================================================================
# 10. Demo
# =============================================================================

def demo():
    """Demonstrate NDGI core concepts with synthetic data."""
    print("=" * 60)
    print("NDGI - NumPy Reference Implementation Demo")
    print("=" * 60)
    
    # 1. Time Encoding
    print("\n[1] Time Encoding")
    t = 0.5
    t_enc = time_encoding(t, freq_bands=2)
    print(f"  Time t={t} -> encoded: {t_enc}")
    print(f"  Encoding dimension: {len(t_enc)}")
    
    # 2. Feature Maps
    print("\n[2] Feature Maps")
    H, W = 128, 128
    f2d_uv = np.random.randn(H, W, 4) * 0.01
    f2d_ut = np.random.randn(64, 10, 2) * 0.01
    f2d_vt = np.random.randn(64, 10, 2) * 0.01
    f3d_uvt = np.random.randn(12, 32, 32, 4) * 0.01
    
    print(f"  F2D_uv: {f2d_uv.shape} (spatial)")
    print(f"  F2D_ut: {f2d_ut.shape} (u-t plane)")
    print(f"  F2D_vt: {f2d_vt.shape} (v-t plane)")
    print(f"  F3D_uvt: {f3d_uvt.shape} (3D spatio-temporal)")
    
    total_params = sum(np.prod(s) for s in [f2d_uv.shape, f2d_ut.shape, f2d_vt.shape, f3d_uvt.shape])
    print(f"  Total feature parameters: {total_params:,} ({total_params * 4 / 1024:.2f} KB)")
    
    # 3. MLP Decoder
    print("\n[3] MLP Decoder")
    feature_dim = 4 + 4 + 2 + 2 + 4  # 16
    decoder = MLPDecoder(feature_dim, hidden_size=16, output_dim=3, num_layers=2)
    print(f"  Input dim: {feature_dim}")
    print(f"  Hidden size: {decoder.hidden_size}")
    print(f"  Layers: {decoder.num_layers}")
    print(f"  Output dim: {decoder.output_dim}")
    
    # Test forward pass
    test_input = np.random.randn(feature_dim) * 0.1
    test_output = decoder.forward(test_input)
    print(f"  Test output: {test_output}")
    
    # 4. BC Compression
    print("\n[4] BC Compression Simulation")
    bc = BCSimulation(32, 32, channels=4, block_size=4)
    reconstructed = bc.reconstruct()
    print(f"  Original size: {32 * 32 * 4 * 32} bits = {32 * 32 * 4 * 4 / 1024:.2f} KB (float32)")
    print(f"  Compressed size: {bc.compressed_size_bits()} bits = {bc.compressed_size_bits() / 8 / 1024:.2f} KB")
    print(f"  Compression ratio: {32 * 32 * 4 * 32 / bc.compressed_size_bits():.2f}x")
    
    # 5. Full NDGI Inference
    print("\n[5] NDGI Inference")
    ndgi = NDGIInference(f2d_uv, f2d_ut, f2d_vt, f3d_uvt, decoder)
    u, v, t = 0.0, 0.0, 0.5
    lighting = ndgi.reconstruct(u, v, t)
    print(f"  I(u={u}, v={v}, t={t}) = {lighting}")
    
    # 6. Reconstruct full lightmap
    print("\n[6] Full Lightmap Reconstruction")
    lightmap = ndgi.reconstruct_lightmap(64, 64, t=0.5)
    print(f"  Reconstructed lightmap: {lightmap.shape}")
    print(f"  Value range: [{lightmap.min():.4f}, {lightmap.max():.4f}]")
    
    # 7. Synthetic Training Demo
    print("\n[7] Simplified Training Demo")
    n = 5  # 5 time steps
    H, W = 32, 32
    
    # Create synthetic temporal lightmaps (gradual color change)
    lightmaps = np.zeros((n, H, W, 3))
    times = np.linspace(0, 1, n)
    for i, t_val in enumerate(times):
        lightmaps[i, :, :, 0] = t_val  # R increases with time
        lightmaps[i, :, :, 2] = 1 - t_val  # B decreases with time
        lightmaps[i, :, :, 1] = 0.3  # G constant
    
    result = train_ndgi_simplified(lightmaps, times, num_iterations=500, batch_size=64)
    print(f"  Final training loss: {result['final_loss']:.6f}")
    
    # 8. Evaluate reconstruction
    print("\n[8] Evaluation")
    reconstructed = result['ndgi'].reconstruct_lightmap(H, W, t=0.5)
    target = lightmaps[2]  # middle time step
    
    psnr = compute_psnr(reconstructed, target)
    ssim = compute_ssim(reconstructed, target)
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # 9. Tile-Based System
    print("\n[9] Tile-Based Virtual Texturing")
    tile_system = TileBasedNDGI(tile_size=128)
    tile_system.add_tile(0, 0, ndgi)
    print(f"  Tile (0,0) added")
    print(f"  Reconstructed at (50, 50): {tile_system.reconstruct(50, 50, 0.5)}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    return ndgi, result


if __name__ == "__main__":
    ndgi, result = demo()
