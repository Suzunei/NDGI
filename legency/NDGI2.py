"""
NDGI2 — Neural Dynamic Global Illumination (v2 Demo)
=====================================================
Based on NDGI2.md specification:

  I(u,v,t) = MLP_Φ(V_uvt, V_uv, V_ut, V_vt, γ(t))

  1. Hybrid feature maps (tri-plane + 3D voxel)   — Section 4
  2. BC compression for F_uv_2d & F_uvt_3d        — Section 7
  3. 8-bit quantization for F_ut_2d & F_vt_2d      — Section 8
  4. Positional Encoding γ(t) = [sin,cos]          — Section 5
  5. MLP decoder with PE                           — Section 6
  6. Gamma correction + normalization              — Section 9
  7. Two-phase training                            — Section 9

      (u,v,t) ──→ [F_uv, F_uvt, F_ut, F_vt, γ(t)] ──→ MLP+PE ──→ RGB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import Tuple, List


# ═══════════════════════════════════════════════════════════════
#  Utility: PSNR / SSIM
# ═══════════════════════════════════════════════════════════════

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR = 10·log10(MAX² / MSE)"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * log10(data_range) - 10 * log10(mse)


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D circular-symmetric Gaussian kernel (standard SSIM: σ=1.5, 11×11)."""
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


# ═══════════════════════════════════════════════════════════════
#  1. Synthetic Lightmap Dataset (with gamma + normalization)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_lightmaps(num_frames: int = 4, h: int = 16, w: int = 16,
                                 gamma: float = 2.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate temporally-varying lightmaps: moving Gaussian spot + checkerboard.

    Preprocessing:
      (a) gamma correction:    L' = L_raw ^ (1/gamma)
      (b) per-channel mean:    L'' = L' / (2 * mean[L'])

    Returns: lightmaps_raw, lightmaps_norm, norm_stats
    """
    frames = []
    for t in range(num_frames):
        bg = ((torch.arange(h).unsqueeze(1) + torch.arange(w)) % 2).float() * 0.2
        center_u = 0.3 + 0.4 * (t / max(num_frames - 1, 1))
        center_v = 0.5
        u = torch.linspace(0, 1, w).view(1, -1).expand(h, w)
        v = torch.linspace(0, 1, h).view(-1, 1).expand(h, w)
        spot = torch.exp(-((u - center_u) ** 2 + (v - center_v) ** 2) / 0.02)
        img = (bg + spot * 0.8).unsqueeze(-1).repeat(1, 1, 3)
        frames.append(img)

    lightmaps_raw = torch.stack(frames, dim=0)  # (T, H, W, 3)

    # (a) Gamma correction
    lightmaps_gamma = lightmaps_raw ** (1.0 / gamma)

    # (b) Per-channel mean normalization
    mean_spatial = lightmaps_gamma.mean(dim=(1, 2), keepdim=True)  # (T, 1, 1, 3)
    lightmaps_norm = lightmaps_gamma / (2.0 * mean_spatial + 1e-8)
    norm_stats = mean_spatial.squeeze(1).squeeze(1)  # (T, 3)

    return lightmaps_raw, lightmaps_norm, norm_stats


def denormalize(pred_norm: torch.Tensor, t_idx: torch.Tensor,
                norm_stats: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Reverse normalization → reverse gamma → clip to [0, 1]."""
    mean = norm_stats[t_idx]
    pred_gamma = pred_norm * (2.0 * mean + 1e-8)
    pred_raw = torch.clamp(pred_gamma, min=0.0) ** gamma
    return torch.clamp(pred_raw, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
#  2. Positional Encoding  (NDGI2.md Section 5)
# ═══════════════════════════════════════════════════════════════

def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """
    Fourier positional encoding γ(x) per NDGI2.md Section 5:

      γ(x) = [sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), …,
              sin(2^{F-1}πx), cos(2^{F-1}πx)]

    Args:
        x: (batch,) scalar ∈ [0, 1]
        num_freqs: number of frequency bands F
    Returns:
        (batch, 2 * num_freqs)
    """
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device)) * np.pi
    phases = x.unsqueeze(-1) * freqs.unsqueeze(0)  # (batch, F)
    return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


# ═══════════════════════════════════════════════════════════════
#  3. MLP Decoder with PE  (NDGI2.md Section 6)
# ═══════════════════════════════════════════════════════════════

class MLPWithPE(nn.Module):
    """
    Lightweight MLP decoder that explicitly combines feature vectors with PE.

    Per NDGI2.md Section 6:
      I(u,v,t) = MLP_Φ(V_uvt, V_uv, V_ut, V_vt, γ(t))

    Architecture:
      [V_uv | V_uvt | V_ut | V_vt | γ(t)] → Linear → GELU → Linear → GELU → Linear → 3 (RGB)

    The PE γ(t) is concatenated with sampled feature vectors as MLP input,
    giving the decoder explicit time-frequency information alongside features.
    """

    def __init__(self, ch_uv: int, ch_uvt: int, ch_ut: int, ch_vt: int,
                 num_time_freqs: int, hidden_size: int = 32):
        super().__init__()
        # PE dimension: γ(t) → 2 * num_time_freqs
        pe_dim = 2 * num_time_freqs
        in_dim = ch_uv + ch_uvt + ch_ut + ch_vt + pe_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 3),  # RGB output
        )
        self.num_time_freqs = num_time_freqs
        self.in_dim = in_dim

    def forward(self, V_uv: torch.Tensor, V_uvt: torch.Tensor,
                V_ut: torch.Tensor, V_vt: torch.Tensor,
                gamma_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V_uv:   (batch, ch_uv)   — sampled from F_uv_2d
            V_uvt:  (batch, ch_uvt)  — sampled from F_uvt_3d
            V_ut:   (batch, ch_ut)   — sampled from F_ut_2d
            V_vt:   (batch, ch_vt)   — sampled from F_vt_2d
            gamma_t: (batch, 2*F_t)  — positional encoding γ(t)
        Returns:
            (batch, 3) RGB
        """
        x = torch.cat([V_uv, V_uvt, V_ut, V_vt, gamma_t], dim=-1)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
#  4. NDGI2 Model  (NDGI2.md Sections 3-6, 7-8)
# ═══════════════════════════════════════════════════════════════

class NDGI2(nn.Module):
    """
    NDGI2 Model per NDGI2.md:

    Θ = {F_uvt³D, F_uv²D, F_ut²D, F_vt²D, Φ}

    Feature sampling (Section 4):
      V_uvt = F_uvt³D(u,v,t)
      V_uv  = F_uv²D(u,v)
      V_ut  = F_ut²D(u,t)
      V_vt  = F_vt²D(v,t)

    BC compression (Section 7):
      f_p = (1 - w_p)·e₁ + w_p·e₂

    Quantization (Section 8):
      Ṽ = V + U(-0.5, 0.5) / 256

    Decoder (Section 6):
      I(u,v,t) = MLP_Φ(V_uvt, V_uv, V_ut, V_vt, γ(t))
    """

    def __init__(self, H: int = 16, W: int = 16, T: int = 4,
                 ch_uv: int = 4, ch_uvt: int = 4,
                 ch_ut: int = 2, ch_vt: int = 2,
                 hidden_size: int = 32,
                 num_time_freqs: int = 3,
                 block_size: int = 4, uvt_scale: int = 2):
        super().__init__()
        self.H, self.W, self.T = H, W, T
        self.block_size = block_size
        self.uvt_scale = uvt_scale
        H3, W3 = H // uvt_scale, W // uvt_scale

        # ---------- Trainable feature maps ----------
        self.F_uv_2d  = nn.Parameter(torch.randn(ch_uv,  H, W)     * 0.1)
        self.F_uvt_3d = nn.Parameter(torch.randn(ch_uvt, T, H3, W3) * 0.1)
        self.F_ut_2d  = nn.Parameter(torch.randn(ch_ut,  H, T)     * 0.1)
        self.F_vt_2d  = nn.Parameter(torch.randn(ch_vt,  W, T)     * 0.1)

        # ---------- BC simulation: F_uv_2d ----------
        hb, wb = H // block_size, W // block_size
        self.bc_e_uv = nn.Parameter(torch.randn(hb * wb, 2, ch_uv) * 0.1)
        self.bc_w_uv = nn.Parameter(torch.randn(hb * wb, block_size ** 2) * 0.1)

        # ---------- BC simulation: F_uvt_3d (per time slice) ----------
        hb3, wb3 = H3 // block_size, W3 // block_size
        self.bc_e_uvt = nn.Parameter(torch.randn(T, hb3 * wb3, 2, ch_uvt) * 0.1)
        self.bc_w_uvt = nn.Parameter(torch.randn(T, hb3 * wb3, block_size ** 2) * 0.1)

        # ---------- Quantization noise scale (8-bit LSB) ----------
        self.quant_alpha = 1.0 / 256.0  # paper Section 8

        # ---------- MLP decoder WITH PE ----------
        self.decoder = MLPWithPE(
            ch_uv=ch_uv, ch_uvt=ch_uvt,
            ch_ut=ch_ut, ch_vt=ch_vt,
            num_time_freqs=num_time_freqs,
            hidden_size=hidden_size,
        )
        self.num_time_freqs = num_time_freqs

    # ------------------------------------------------------------------
    #  BC reconstruction  (Section 7)
    # ------------------------------------------------------------------
    @staticmethod
    def _bc_reconstruct(H: int, W: int, bc_e: torch.Tensor,
                        bc_w: torch.Tensor, blk: int) -> torch.Tensor:
        """f_p = (1 − w_p)·e₁ + w_p·e₂"""
        C = bc_e.shape[-1]
        bh, bw = H // blk, W // blk

        w = torch.sigmoid(bc_w)              # (N, blk²) ∈ [0,1]
        e1, e2 = bc_e[:, 0, :], bc_e[:, 1, :]  # (N, C)

        texels = (1 - w.unsqueeze(-1)) * e1.unsqueeze(1) + \
                 w.unsqueeze(-1) * e2.unsqueeze(1)  # (N, blk², C)
        texels = texels.view(bh, bw, blk, blk, C)
        texels = texels.permute(4, 0, 2, 1, 3).contiguous()
        return texels.reshape(C, H, W)

    # ------------------------------------------------------------------
    #  Sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_2d(feat: torch.Tensor, u: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:
        """feat: (C, H, W) → (batch, C)"""
        B = u.shape[0]
        if feat.dim() == 3:
            feat = feat.unsqueeze(0).expand(B, -1, -1, -1)
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).view(B, 1, 1, 2)
        return F.grid_sample(feat, grid, mode='bilinear',
                             align_corners=False).reshape(B, -1)

    @staticmethod
    def _sample_3d(feat: torch.Tensor, u: torch.Tensor,
                   v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """feat: (C, D, H, W) → (batch, C)"""
        B = u.shape[0]
        if feat.dim() == 4:
            feat = feat.unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid = torch.stack([u * 2 - 1, v * 2 - 1, t * 2 - 1],
                           dim=-1).view(B, 1, 1, 1, 3)
        return F.grid_sample(feat, grid, mode='bilinear',
                             align_corners=False).reshape(B, -1)

    # ------------------------------------------------------------------
    #  Forward  (NDGI2.md Section 6)
    # ------------------------------------------------------------------
    def forward(self, u: torch.Tensor, v: torch.Tensor, t: torch.Tensor,
                use_bc_sim: bool = True, add_quant_noise: bool = True) -> torch.Tensor:
        """
        u, v, t ∈ [0, 1],  shape (batch,)
        →  (batch, 3) RGB

        I(u,v,t) = MLP_Φ(V_uvt, V_uv, V_ut, V_vt, γ(t))
        """
        # --- ① F_uv_2d (BC simulation) ---
        recon_uv = self._bc_reconstruct(self.H, self.W,
                                        self.bc_e_uv, self.bc_w_uv,
                                        self.block_size) if use_bc_sim else self.F_uv_2d
        V_uv = self._sample_2d(recon_uv, u, v)  # (B, ch_uv)

        # --- ② F_uvt_3d (BC simulation, per time slice) ---
        if use_bc_sim:
            H3, W3 = self.H // self.uvt_scale, self.W // self.uvt_scale
            slices = [self._bc_reconstruct(H3, W3,
                                           self.bc_e_uvt[ti], self.bc_w_uvt[ti],
                                           self.block_size)
                      for ti in range(self.T)]
            recon_uvt = torch.stack(slices, dim=1)  # (C, T, H3, W3)
        else:
            recon_uvt = self.F_uvt_3d
        V_uvt = self._sample_3d(recon_uvt, u, v, t)  # (B, ch_uvt)

        # --- ③ F_ut_2d (quantization noise) ---
        V_ut = self._sample_2d(self.F_ut_2d, t, u)
        if add_quant_noise:
            V_ut = V_ut + (torch.rand_like(V_ut) - 0.5) * self.quant_alpha

        # --- ④ F_vt_2d (quantization noise) ---
        V_vt = self._sample_2d(self.F_vt_2d, t, v)
        if add_quant_noise:
            V_vt = V_vt + (torch.rand_like(V_vt) - 0.5) * self.quant_alpha

        # --- ⑤ γ(t) Positional Encoding ---
        gamma_t = positional_encoding(t, self.num_time_freqs)  # (B, 2·F_t)

        # --- ⑥ MLP + PE → RGB ---
        #   I(u,v,t) = MLP_Φ(V_uvt, V_uv, V_ut, V_vt, γ(t))
        return self.decoder(V_uv, V_uvt, V_ut, V_vt, gamma_t)

    # ------------------------------------------------------------------
    #  Parameter counting
    # ------------------------------------------------------------------
    def count_params(self) -> dict:
        ch_uv, ch_uvt = self.F_uv_2d.shape[0], self.F_uvt_3d.shape[0]
        ch_ut, ch_vt   = self.F_ut_2d.shape[0], self.F_vt_2d.shape[0]
        H3, W3, blk = self.H // self.uvt_scale, self.W // self.uvt_scale, self.block_size

        raw = {
            'uv':  ch_uv  * self.H * self.W,
            'uvt': ch_uvt * self.T * H3 * W3,
            'ut':  ch_ut  * self.H * self.T,
            'vt':  ch_vt  * self.W * self.T,
        }
        bc = {
            'uv':  (self.H // blk) * (self.W // blk) * (2 * ch_uv + blk * blk),
            'uvt': self.T * (H3 // blk) * (W3 // blk) * (2 * ch_uvt + blk * blk),
        }
        dec = sum(p.numel() for p in self.decoder.parameters())
        total_raw = raw['uv'] + raw['uvt'] + raw['ut'] + raw['vt'] + dec
        total_bc  = bc['uv'] + bc['uvt'] + raw['ut'] + raw['vt'] + dec

        return {'raw': raw, 'bc': bc, 'decoder': dec,
                'total_raw': total_raw, 'total_bc': total_bc}


# ═══════════════════════════════════════════════════════════════
#  5. Training  (Two-phase)
# ═══════════════════════════════════════════════════════════════

def train(model: NDGI2,
          lightmaps_norm: torch.Tensor,
          epochs_phase1: int = 2000,
          epochs_phase2: int = 500,
          batch_size: int = 2048,
          lr: float = 1e-3,
          device: str = 'cuda') -> List[float]:
    """
    Phase 1 — train all parameters with BC sim + quant noise.
    Phase 2 — freeze feature maps, fine-tune MLP decoder only.
    """
    T, H, W = lightmaps_norm.shape[0], model.H, model.W
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_phase1 + epochs_phase2)
    loss_fn = nn.MSELoss()
    history = []

    lightmaps_4d = lightmaps_norm.permute(0, 3, 1, 2)  # (T, 3, H, W)

    for epoch in range(epochs_phase1 + epochs_phase2):
        # ── Phase transition ──
        if epoch == epochs_phase1:
            print("\n>>> Phase 2: freeze feature maps, fine-tune MLP+PE decoder")
            for n, p in model.named_parameters():
                if 'decoder' not in n:
                    p.requires_grad = False
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                          model.parameters()), lr=lr * 0.1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_phase2)

        # ── Random batch ──
        u = torch.rand(batch_size, device=device)
        v = torch.rand(batch_size, device=device)
        t = torch.rand(batch_size, device=device)

        # Bilinear GT sampling
        t_scaled = t * (T - 1)
        t0 = t_scaled.floor().long().clamp(0, T - 1)
        t1 = (t0 + 1).clamp(0, T - 1)
        wt = (t_scaled - t0.float()).view(-1, 1, 1, 1)

        lm0 = lightmaps_4d[t0]
        lm1 = lightmaps_4d[t1]

        grid_uv = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).view(batch_size, 1, 1, 2)
        gt0 = F.grid_sample(lm0, grid_uv, mode='bilinear', align_corners=False)
        gt1 = F.grid_sample(lm1, grid_uv, mode='bilinear', align_corners=False)
        gt = (gt0 * (1 - wt) + gt1 * wt).reshape(batch_size, 3)

        pred = model(u, v, t, use_bc_sim=True, add_quant_noise=True)
        loss = loss_fn(pred, gt)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())

        if (epoch + 1) % 200 == 0:
            ph = "P1" if epoch < epochs_phase1 else "P2"
            print(f"  [{ph}] Epoch {epoch + 1:4d}  |  Loss = {loss.item():.6f}  |  LR = {sched.get_last_lr()[0]:.2e}")

    return history


# ═══════════════════════════════════════════════════════════════
#  6. Visualization
# ═══════════════════════════════════════════════════════════════

def visualize(lightmaps_raw: torch.Tensor, pred_raw: List[torch.Tensor],
              loss_hist: List[float], psnr: List[float], ssim: List[float],
              params: dict, num_time_freqs: int):
    T = lightmaps_raw.shape[0]
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1, 1],
                  hspace=0.45, wspace=0.35)

    # ---- Top-left: GT vs Pred ----
    gsl = GridSpecFromSubplotSpec(2, T, subplot_spec=gs[0, :2],
                                   wspace=0.1, hspace=0.2)
    for ti in range(T):
        for row, (label, data) in enumerate([('GT', lightmaps_raw), ('Pred', pred_raw)]):
            ax = fig.add_subplot(gsl[row, ti])
            ax.imshow(np.clip(data[ti].cpu().numpy(), 0, 1))
            ax.set_title(f'{label} t={ti}', fontsize=9)
            ax.axis('off')

    # ---- Top-right: Loss curve ----
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(loss_hist, color='#2c3e50', linewidth=0.8)
    ax.set_title('Training Loss (MLP+PE)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)

    # ---- Bottom-left: PSNR ----
    ax = fig.add_subplot(gs[1, 0])
    c = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(range(T), psnr, color=c[:T])
    ax.set_title('PSNR', fontweight='bold')
    ax.set_xticks(range(T)); ax.set_ylabel('dB')
    for b, v in zip(bars, psnr):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f'{v:.1f}', ha='center', fontsize=9)

    # ---- Bottom-mid: SSIM ----
    ax = fig.add_subplot(gs[1, 1])
    c2 = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    bars = ax.bar(range(T), ssim, color=c2[:T])
    ax.set_title('SSIM', fontweight='bold')
    ax.set_xticks(range(T)); ax.set_ylim(0, 1)
    for b, v in zip(bars, ssim):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f'{v:.3f}', ha='center', fontsize=9)

    # ---- Bottom-right: Compression + PE info ----
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    r, b = params['raw'], params['bc']
    txt = (
        "NDGI2 Model Info\n(MLP + Positional Encoding)\n\n"
        f"γ(t) freq bands: {num_time_freqs}  → dim {2*num_time_freqs}\n\n"
        f"F_uv_2d   raw {r['uv']:5d}  → BC {b['uv']:5d}   {r['uv']/b['uv']:.2f}:1\n"
        f"F_uvt_3d  raw {r['uvt']:5d}  → BC {b['uvt']:5d}   {r['uvt']/b['uvt']:.2f}:1\n"
        f"F_ut_2d        {r['ut']:5d}  (8-bit quant)\n"
        f"F_vt_2d        {r['vt']:5d}  (8-bit quant)\n"
        f"Decoder+PE     {params['decoder']:5d}\n\n"
        f"Total raw  {params['total_raw']:5d}\n"
        f"Total BC   {params['total_bc']:5d}\n"
        f"Overall    {params['total_raw']/params['total_bc']:.2f}:1"
    )
    ax.text(0.5, 0.5, txt, transform=ax.transAxes,
            fontsize=9.5, va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    fig.suptitle('NDGI2 — MLP+PE Results Summary', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig('NDGI2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved → NDGI2_results.png")


# ═══════════════════════════════════════════════════════════════
#  7. Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ── Hyperparameter config ──
    cfg = {
        'data': {
            'T': 4, 'H': 128, 'W': 128, 'gamma': 2.2,
        },
        'feature': {
            'ch_uv':  2, 'ch_uvt': 2,
            'ch_ut':  2, 'ch_vt':  2,
            'block_size': 8, 'uvt_scale': 2,
        },
        'decoder': {
            'hidden_size':    32,
            'num_time_freqs': 3,   # γ(t) freq bands → PE dim = 2·3 = 6
        },
        'train': {
            'epochs_phase1': 2000,
            'epochs_phase2': 500,
            'batch_size':    2048,
            'lr':            1e-3,
        },
    }

    d = cfg['data']
    f = cfg['feature']
    c = cfg['decoder']
    t_cfg = cfg['train']
    T, H, W = d['T'], d['H'], d['W']

    # ---- Data ----
    print("Generating synthetic lightmaps …")
    raw, norm, stats = generate_synthetic_lightmaps(T, H, W, gamma=d['gamma'])
    raw, norm, stats = raw.to(device), norm.to(device), stats.to(device)
    print(f"  raw {list(raw.shape)}  norm {list(norm.shape)}  stats {list(stats.shape)}\n")

    # ---- Model ----
    model = NDGI2(
        H=H, W=W, T=T,
        ch_uv=f['ch_uv'], ch_uvt=f['ch_uvt'],
        ch_ut=f['ch_ut'], ch_vt=f['ch_vt'],
        hidden_size=c['hidden_size'],
        num_time_freqs=c['num_time_freqs'],
        block_size=f['block_size'],
        uvt_scale=f['uvt_scale'],
    ).to(device)

    params_dict = model.count_params()
    r, b = params_dict['raw'], params_dict['bc']
    pe_dim = 2 * c['num_time_freqs']
    print("Parameter breakdown:")
    print(f"  γ(t) PE: {c['num_time_freqs']} freq bands → {pe_dim} dim")
    print(f"  F_uv_2d   raw {r['uv']:5d}  →  BC {b['uv']:5d}   ({r['uv']/b['uv']:.2f}:1)")
    print(f"  F_uvt_3d  raw {r['uvt']:5d}  →  BC {b['uvt']:5d}   ({r['uvt']/b['uvt']:.2f}:1)")
    print(f"  F_ut_2d        {r['ut']:5d}   (8-bit quant)")
    print(f"  F_vt_2d        {r['vt']:5d}   (8-bit quant)")
    print(f"  Decoder+PE     {params_dict['decoder']:5d}")
    print(f"  Total    raw {params_dict['total_raw']:5d}  →  BC {params_dict['total_bc']:5d}   ({params_dict['total_raw']/params_dict['total_bc']:.2f}:1)\n")

    # ---- Train ----
    print(f"Training (P1: {t_cfg['epochs_phase1']} + P2: {t_cfg['epochs_phase2']} epochs) …")
    loss_hist = train(model, norm,
                      epochs_phase1=t_cfg['epochs_phase1'],
                      epochs_phase2=t_cfg['epochs_phase2'],
                      batch_size=t_cfg['batch_size'],
                      lr=t_cfg['lr'],
                      device=device)

    # ---- Evaluate ----
    print("\nEvaluating …")
    model.eval()
    ul = torch.linspace(0, 1, W, device=device)
    vl = torch.linspace(0, 1, H, device=device)
    uu, vv = torch.meshgrid(ul, vl, indexing='xy')
    uf, vf = uu.flatten(), vv.flatten()
    N = H * W

    pred_raw, psnr_list, ssim_list = [], [], []
    with torch.no_grad():
        for ti in range(T):
            tv = ti / (T - 1) if T > 1 else 0.0
            tt = torch.full((N,), tv, device=device)
            pred_norm = model(uf, vf, tt, use_bc_sim=True, add_quant_noise=False)
            idx = torch.full((N,), ti, dtype=torch.long, device=device)
            pr = denormalize(pred_norm, idx, stats, d['gamma']).view(H, W, 3)
            pred_raw.append(pr)
            psnr_list.append(compute_psnr(pr, raw[ti]))
            ssim_list.append(compute_ssim(pr, raw[ti]))
            print(f"  t={ti}:  PSNR = {psnr_list[-1]:.2f} dB   SSIM = {ssim_list[-1]:.4f}")

    # ---- Visualize ----
    visualize(raw, pred_raw, loss_hist, psnr_list, ssim_list,
              params_dict, c['num_time_freqs'])


if __name__ == '__main__':
    main()
