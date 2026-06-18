import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import Tuple

# -------------------- SSIM / PSNR 指标函数 --------------------
def compute_psnr(img1, img2, data_range=1.0):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * log10(data_range) - 10 * log10(mse)


def compute_ssim(img1, img2, window_size=4, data_range=1.0):
    """计算 SSIM (Structural Similarity Index)，适合 16×16 小图"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = torch.ones(1, 1, window_size, window_size) / (window_size * window_size)
    kernel = kernel.to(img1.device)

    # (H, W, C) → (C, 1, H, W)
    if img1.dim() == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(1)
        img2 = img2.permute(2, 0, 1).unsqueeze(1)

    ssim_vals = []
    for c in range(img1.shape[0]):
        ch1 = img1[c:c+1]
        ch2 = img2[c:c+1]

        mu1 = F.conv2d(ch1, kernel, padding=window_size // 2)
        mu2 = F.conv2d(ch2, kernel, padding=window_size // 2)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(ch1 * ch1, kernel, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(ch2 * ch2, kernel, padding=window_size // 2) - mu2_sq
        sigma12  = F.conv2d(ch1 * ch2, kernel, padding=window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        ssim_vals.append(ssim_map.mean().item())

    return float(np.mean(ssim_vals))

# -------------------- 1. 模拟时序 Lightmap 数据集 --------------------
def generate_synthetic_lightmaps(num_frames=4, h=16, w=16):
    """
    生成简单的时间变化光照图：一个移动的光斑 + 静态背景。
    返回形状 (num_frames, h, w, 3) 的 HDR 光照值（0~1 范围）。
    """
    lightmaps = []
    for t in range(num_frames):
        # 静态背景：棋盘格
        bg = ((torch.arange(h).unsqueeze(1) + torch.arange(w)) % 2).float() * 0.2
        # 移动光斑：高斯状
        center_u = 0.3 + 0.4 * (t / num_frames)  # 水平移动
        center_v = 0.5
        u = torch.linspace(0, 1, w).view(1, -1).expand(h, w)
        v = torch.linspace(0, 1, h).view(-1, 1).expand(h, w)
        spot = torch.exp(-((u - center_u)**2 + (v - center_v)**2) / 0.02)
        # 组合
        img = bg + spot * 0.8
        # 变成 3 通道（RGB 相同，简化）
        img = img.unsqueeze(-1).repeat(1, 1, 3)
        lightmaps.append(img)
    return torch.stack(lightmaps, dim=0)  # (T, H, W, 3)

# -------------------- 2. 位置编码 --------------------
def positional_encoding_time(t: torch.Tensor, num_freqs=2) -> torch.Tensor:
    """将归一化时间 t ∈ [0,1] 编码为 2*num_freqs 维向量"""
    freqs = 2 ** torch.arange(num_freqs, device=t.device) * np.pi
    phases = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (batch, num_freqs)
    enc = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
    return enc  # (batch, 2*num_freqs)

# -------------------- 3. NDGI 模型 --------------------
class NDGI(nn.Module):
    def __init__(self, H=16, W=16, T=4,
                 ch_uv=4, ch_uvt=4, ch_ut=2, ch_vt=2,
                 hidden_size=16):
        super().__init__()
        self.H, self.W, self.T = H, W, T
        # ---------- 可训练的特征图 ----------
        # F_uv_2d: (ch_uv, H, W)
        self.F_uv_2d = nn.Parameter(torch.randn(ch_uv, H, W) * 0.1)
        # F_uvt_3d: (ch_uvt, T, H//2, W//2)  降低分辨率
        self.F_uvt_3d = nn.Parameter(torch.randn(ch_uvt, T, H//2, W//2) * 0.1)
        # F_ut_2d: (ch_ut, H, T)
        self.F_ut_2d = nn.Parameter(torch.randn(ch_ut, H, T) * 0.1)
        # F_vt_2d: (ch_vt, W, T)
        self.F_vt_2d = nn.Parameter(torch.randn(ch_vt, W, T) * 0.1)

        # ---------- 用于 BC 仿真的端点-权重参数（仅用于 F_uv_2d 和 F_uvt_3d）----------
        # 论文中将特征图划分为 4x4 块，每块用两个端点+权重表示。
        # 这里简化：对整个特征图进行块划分（块大小 4x4）
        self.block_size = 4
        self._init_bc_params()

        # ---------- 轻量 MLP 解码器 ----------
        input_dim = ch_uv + ch_uvt + ch_ut + ch_vt + 4  # 4 = γ(t) 维度（2 freq）
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),  # RGB
        )

    def _init_bc_params(self):
        """初始化 BC 仿真所需的端点 (e1,e2) 和权重 (w) 参数"""
        # 对 F_uv_2d: 分成 (H/4)*(W/4) 个块，每块 4x4 texels
        h_blocks = self.H // self.block_size
        w_blocks = self.W // self.block_size
        # 端点形状: (num_blocks, 2, channels)
        self.bc_e_uv = nn.Parameter(torch.randn(h_blocks * w_blocks, 2, self.F_uv_2d.shape[0]) * 0.1)
        # 权重形状: (num_blocks, 16)  -- 16个texel的权重
        self.bc_w_uv = nn.Parameter(torch.randn(h_blocks * w_blocks, 16) * 0.1)

        # 对 F_uvt_3d: 沿时间轴切片，每片 2D 单独处理
        # 这里假设时间维度 T，每片大小为 H/2 x W/2
        ht = self.H // 2
        wt = self.W // 2
        h_blocks_t = ht // self.block_size
        w_blocks_t = wt // self.block_size
        # 每个时间片独立参数，但共享同一组端点-权重？论文中是每片独立压缩，这里简化：所有时间片共用一套块参数？
        # 更准确：每个时间片有自己的端点权重。为了节省参数，我们只演示对第一片做 BC 仿真。
        # 但为了符合论文，我们为所有时间片都创建参数（但会导致参数量增大）。
        # 这里简化：只对第一片做 BC 仿真，其余直接优化。或者，为每个时间片独立创建参数。
        # 为演示简洁，我们只对 F_uv_2d 做 BC 仿真，对 F_uvt_3d 不做（直接优化）。
        # 实际上论文中对两者都做了 BC 仿真。我们可以在代码中注释说明。
        # 因此跳过 F_uvt_3d 的 BC 参数初始化。

    def bc_reconstruct(self, feat_param, bc_e, bc_w, block_size=4):
        """
        根据端点-权重重建特征图。
        输入:
            feat_param: 原始特征图参数 (C, H, W) 仅用于获取形状
            bc_e: (num_blocks, 2, C)
            bc_w: (num_blocks, 16)
        返回: (C, H, W) 重建后的特征图
        """
        C, H, W = feat_param.shape
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size
        assert num_blocks_h * num_blocks_w == bc_e.shape[0]
        # 将 bc_w 限制在 [0,1]
        w = torch.sigmoid(bc_w)  # (num_blocks, 16)
        # 线性插值: f_p = (1-w)*e1 + w*e2
        e1 = bc_e[:, 0, :]  # (num_blocks, C)
        e2 = bc_e[:, 1, :]  # (num_blocks, C)
        # 重建每个块的 16 个 texel
        # 输出形状: (num_blocks, 16, C)
        block_texels = (1 - w.unsqueeze(-1)) * e1.unsqueeze(1) + w.unsqueeze(-1) * e2.unsqueeze(1)
        # 将块排列成图像
        # 先将 block_texels reshape 为 (num_blocks_h, num_blocks_w, block_size, block_size, C)
        block_texels = block_texels.view(num_blocks_h, num_blocks_w, block_size, block_size, C)
        # 交换维度以合并为 (C, H, W)
        block_texels = block_texels.permute(4, 0, 2, 1, 3).contiguous()  # (C, num_blocks_h, block_size, num_blocks_w, block_size)
        recon = block_texels.view(C, H, W)
        return recon

    def forward(self, u, v, t, use_bc_sim=True):
        """
        u,v: 归一化纹理坐标 [0,1]，形状 (batch,)
        t: 归一化时间 [0,1]，形状 (batch,)
        use_bc_sim: 是否使用 BC 仿真重建 F_uv_2d
        """
        batch = u.shape[0]
        device = u.device

        # ---- 采样特征图 ----
        # 将 uv 映射到像素坐标（考虑边界 clamp）
        def sample_grid(feat, u, v):
            """feat: (C, H, W), 返回 (batch, C)"""
            C, H, W = feat.shape
            # 构造采样网格 (batch, 1, 1, 2)
            grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).view(batch, 1, 1, 2)
            # expand 使 input 与 grid 的 batch 维度一致
            feat_4d = feat.unsqueeze(0).expand(batch, -1, -1, -1)  # (batch, C, H, W)
            sampled = F.grid_sample(feat_4d, grid, mode='bilinear', align_corners=False)  # (batch, C, 1, 1)
            return sampled.reshape(batch, C)  # (batch, C)

        # F_uv_2d
        if use_bc_sim:
            # 使用 BC 仿真重建
            recon_uv = self.bc_reconstruct(self.F_uv_2d, self.bc_e_uv, self.bc_w_uv)
        else:
            recon_uv = self.F_uv_2d
        V_uv = sample_grid(recon_uv, u, v)  # (batch, ch_uv)

        # F_uvt_3d: 3D 采样 (u,v,t) — 使用 5D grid_sample
        _, T, H3, W3 = self.F_uvt_3d.shape
        feat_3d = self.F_uvt_3d.unsqueeze(0).expand(batch, -1, -1, -1, -1)  # (batch, C, T, H3, W3)
        # 构造 3D 采样网格 (batch, 1, 1, 1, 3), 坐标顺序 (x, y, z) = (W, H, D)
        grid_3d = torch.stack([
            u * 2 - 1,  # x (对应 W)
            v * 2 - 1,  # y (对应 H)
            t * 2 - 1   # z (对应 T)
        ], dim=-1).view(batch, 1, 1, 1, 3)
        sampled_3d = F.grid_sample(feat_3d, grid_3d, mode='bilinear', align_corners=False)  # (batch, C, 1, 1, 1)
        V_uvt = sampled_3d.reshape(batch, -1)  # (batch, C)

        # F_ut_2d: 采样 (u,t)
        # 形状 (ch_ut, H, T): H 是空间高度，T 是时间
        C_ut, H_ut, T_ut = self.F_ut_2d.shape
        feat_ut = self.F_ut_2d.unsqueeze(0).expand(batch, -1, -1, -1)  # (batch, C, H, T)
        grid_ut = torch.stack([
            t * 2 - 1,  # x (对应 T)
            u * 2 - 1   # y (对应 H)
        ], dim=-1).view(batch, 1, 1, 2)
        sampled_ut = F.grid_sample(feat_ut, grid_ut, mode='bilinear', align_corners=False)  # (batch, C_ut, 1, 1)
        V_ut = sampled_ut.reshape(batch, -1)  # (batch, C_ut)

        # F_vt_2d: 采样 (v,t)
        C_vt, W_vt, T_vt = self.F_vt_2d.shape
        feat_vt = self.F_vt_2d.unsqueeze(0).expand(batch, -1, -1, -1)  # (batch, C, W, T)
        grid_vt = torch.stack([
            t * 2 - 1,  # x (对应 T)
            v * 2 - 1   # y (对应 W)
        ], dim=-1).view(batch, 1, 1, 2)
        sampled_vt = F.grid_sample(feat_vt, grid_vt, mode='bilinear', align_corners=False)  # (batch, C_vt, 1, 1)
        V_vt = sampled_vt.reshape(batch, -1)  # (batch, C_vt)

        # ---- 时间位置编码 ----
        gamma_t = positional_encoding_time(t, num_freqs=2)  # (batch, 4)

        # ---- 拼接并送入 MLP ----
        x = torch.cat([V_uv, V_uvt, V_ut, V_vt, gamma_t], dim=-1)  # (batch, total_dim)
        rgb = self.decoder(x)  # (batch, 3)
        return rgb

# -------------------- 4. 训练准备 --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 生成数据
T = 4
H, W = 16, 16
lightmaps = generate_synthetic_lightmaps(num_frames=T, h=H, w=W).to(device)  # (T, H, W, 3)
print(f"Lightmap shape: {lightmaps.shape}")

# 实例化模型
model = NDGI(H=H, W=W, T=T).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 准备训练样本：随机采样 (u,v,t) 坐标
def sample_random_batch(batch_size=256):
    u = torch.rand(batch_size, device=device)
    v = torch.rand(batch_size, device=device)
    t = torch.rand(batch_size, device=device)  # 连续时间
    # 获取真实值：需要从 lightmaps 中插值
    # 将 t 映射到最近的帧索引（或双线性插值）
    # 简化：取最近邻帧
    idx = (t * (T-1)).long().clamp(0, T-1)
    # 从 lightmaps 中采样 (u,v) 位置
    # 将 u,v 映射到像素坐标
    u_idx = (u * (W-1)).long().clamp(0, W-1)
    v_idx = (v * (H-1)).long().clamp(0, H-1)
    gt = lightmaps[idx, v_idx, u_idx, :]  # (batch, 3)
    return u, v, t, gt

# -------------------- 5. 训练循环 --------------------
num_epochs = 1000
batch_size = 2048
loss_history = []
for epoch in range(num_epochs):
    u, v, t, gt = sample_random_batch(batch_size)
    # 前向传播（使用 BC 仿真）
    pred = model(u, v, t, use_bc_sim=True)
    loss = loss_fn(pred, gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# -------------------- 6. 测试推理 --------------------
# 随机采样一些点
with torch.no_grad():
    test_u = torch.tensor([0.25, 0.75, 0.5], device=device)
    test_v = torch.tensor([0.5, 0.25, 0.8], device=device)
    test_t = torch.tensor([0.0, 0.5, 1.0], device=device)
    pred_rgb = model(test_u, test_v, test_t, use_bc_sim=True)
    print("Predicted RGB:")
    print(pred_rgb.cpu().numpy())

# -------------------- 7. 可视化（合并图表）--------------------
model.eval()
with torch.no_grad():
    # 生成密集的 (u,v) 坐标网格
    u_lin = torch.linspace(0, 1, W, device=device)
    v_lin = torch.linspace(0, 1, H, device=device)
    uu, vv = torch.meshgrid(u_lin, v_lin, indexing='xy')  # (H, W)
    u_flat = uu.flatten()
    v_flat = vv.flatten()
    num_pixels = H * W

    # 预测所有帧
    pred_images = []
    for t_idx in range(T):
        t_val = t_idx / (T - 1) if T > 1 else 0.0
        t_tensor = torch.full((num_pixels,), t_val, device=device)
        pred = model(u_flat, v_flat, t_tensor, use_bc_sim=True)
        pred_images.append(pred.view(H, W, 3))

    # ====== 计算 PSNR / SSIM ======
    psnr_list, ssim_list = [], []
    for t_idx in range(T):
        psnr_list.append(compute_psnr(pred_images[t_idx], lightmaps[t_idx]))
        ssim_list.append(compute_ssim(pred_images[t_idx], lightmaps[t_idx], window_size=4))
        print(f"  t={t_idx}: PSNR={psnr_list[-1]:.2f} dB, SSIM={ssim_list[-1]:.4f}")

    # ====== 计算压缩率 ======
    ch_uv, ch_uvt, ch_ut, ch_vt = (model.F_uv_2d.shape[0], model.F_uvt_3d.shape[0],
                                    model.F_ut_2d.shape[0], model.F_vt_2d.shape[0])
    raw_uv_params = ch_uv * H * W
    bc_uv_params = (H // 4) * (W // 4) * (2 * ch_uv + 16)
    compression_uv = raw_uv_params / bc_uv_params

    total_raw = (raw_uv_params + ch_uvt * T * (H // 2) * (W // 2)
                 + ch_ut * H * T + ch_vt * W * T
                 + sum(p.numel() for p in model.decoder.parameters()))
    total_bc  = (bc_uv_params + ch_uvt * T * (H // 2) * (W // 2)
                 + ch_ut * H * T + ch_vt * W * T
                 + sum(p.numel() for p in model.decoder.parameters()))
    compression_total = total_raw / total_bc

    print(f"\n压缩率分析:")
    print(f"  F_uv_2d: {raw_uv_params} -> {bc_uv_params} params, 压缩比 {compression_uv:.2f}x")
    print(f"  模型整体: {total_raw} -> {total_bc} params, 压缩比 {compression_total:.2f}x")

    # ====== 单一合并图表 ======
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 1, 1],
                  hspace=0.45, wspace=0.35)

    # ----- 左上 (span 2 cols): Lightmap 对比 -----
    gs_light = GridSpecFromSubplotSpec(2, T, subplot_spec=gs[0, :2],
                                        wspace=0.1, hspace=0.2)
    for t_idx in range(T):
        # GT 行
        ax_gt = fig.add_subplot(gs_light[0, t_idx])
        ax_gt.imshow(np.clip(lightmaps[t_idx].cpu().numpy(), 0, 1))
        ax_gt.set_title(f'GT t={t_idx}', fontsize=9)
        ax_gt.axis('off')
        # Pred 行
        ax_pred = fig.add_subplot(gs_light[1, t_idx])
        ax_pred.imshow(np.clip(pred_images[t_idx].cpu().numpy(), 0, 1))
        ax_pred.set_title(f'Pred t={t_idx}', fontsize=9)
        ax_pred.axis('off')

    # ----- 右上: Loss 曲线 -----
    ax_loss = fig.add_subplot(gs[0, 2])
    ax_loss.plot(loss_history, color='#2c3e50', linewidth=0.8)
    ax_loss.set_title('Training Loss', fontweight='bold', fontsize=11)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('MSE Loss')
    ax_loss.grid(True, alpha=0.3)

    # ----- 左下: PSNR 柱状图 -----
    ax_psnr = fig.add_subplot(gs[1, 0])
    colors_psnr = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars_psnr = ax_psnr.bar(range(T), psnr_list, color=colors_psnr[:T])
    ax_psnr.set_title('PSNR per Time Step', fontweight='bold', fontsize=11)
    ax_psnr.set_xlabel('Time step')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.set_xticks(range(T))
    for bar, v in zip(bars_psnr, psnr_list):
        ax_psnr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # ----- 中下: SSIM 柱状图 -----
    ax_ssim = fig.add_subplot(gs[1, 1])
    colors_ssim = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    bars_ssim = ax_ssim.bar(range(T), ssim_list, color=colors_ssim[:T])
    ax_ssim.set_title('SSIM per Time Step', fontweight='bold', fontsize=11)
    ax_ssim.set_xlabel('Time step')
    ax_ssim.set_ylabel('SSIM')
    ax_ssim.set_xticks(range(T))
    ax_ssim.set_ylim(0, 1)
    for bar, v in zip(bars_ssim, ssim_list):
        ax_ssim.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # ----- 右下: 压缩率面板 -----
    ax_comp = fig.add_subplot(gs[1, 2])
    ax_comp.axis('off')
    comp_text = (
        f"Compression Rate\n(BC Simulation)\n\n"
        f"F_uv_2d  raw:  {raw_uv_params:4d}  params\n"
        f"F_uv_2d  BC:   {bc_uv_params:4d}  params\n"
        f"UV ratio:          {compression_uv:.2f}x\n\n"
        f"Model (no BC):  {total_raw:5d}  params\n"
        f"Model (BC):     {total_bc:5d}  params\n"
        f"Overall ratio:    {compression_total:.2f}x"
    )
    ax_comp.text(0.5, 0.5, comp_text, transform=ax_comp.transAxes,
                 fontsize=10, va='center', ha='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9))

    fig.suptitle('NDGI: Results Summary', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig('NDGI_results.png', dpi=150)
    plt.show()
    print("\n可视化已保存至 NDGI_results.png")