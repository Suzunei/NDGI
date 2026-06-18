"""
NDGI 测试信号集 — 模拟真实 Lightmap 的时序光照数据
=====================================================
为 NDGI_demo.py 提供匹配 TemporalLightmapDataset 接口的测试信号。

真实 Lightmap 特征:
- 平滑渐变（间接光照扩散）
- 阴影边界（直接光产生的阴影，半影区柔和过渡）
- 色彩渗透（彩色表面反射带色间接光）
- 时间变化（动态光源移动/闪烁/开关）
- 非均匀空间分布（光照集中在光源附近）
- HDR 特征（某些区域亮度较高）

每个生成器返回 (lightmaps, times):
  lightmaps: (T, H, W, 3)  RGB ∈ [0, 1]
  times:     (T,)           归一化时间 ∈ [0, 1]

可直接传入 TemporalLightmapDataset(lightmaps, times)
"""

import torch
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # CJK fallback
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import Dict, Tuple, Optional
from math import pi


# =============================================================================
# 基础工具函数
# =============================================================================

def _make_uv_grid(H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建归一化 UV 坐标网格, u∈[0,1] 横轴, v∈[0,1] 纵轴"""
    u = torch.linspace(0, 1, W).view(1, -1).expand(H, W)
    v = torch.linspace(0, 1, H).view(-1, 1).expand(H, W)
    return u, v


def _gaussian_spot(u, v, center_u, center_v, sigma, amplitude=1.0):
    """
    2D 高斯光斑 — 模拟点光源在表面上的照射
    Args:
        u, v:       归一化坐标网格 (H, W)
        center_u/v: 光斑中心
        sigma:      光斑扩散半径 (越大越柔和)
        amplitude:  亮度峰值
    """
    return amplitude * torch.exp(-((u - center_u)**2 + (v - center_v)**2) / (2 * sigma**2))


def _soft_shadow(u, v, center_u, center_v, shadow_dir_u, shadow_dir_v,
                 penumbra_width=0.05, shadow_length=0.3):
    """
    模拟柔和阴影 — 从光源中心沿 shadow_dir 方向投射
    penumbra_width 控制半影过渡宽度（真实lightmap阴影边缘不是硬切换）
    """
    # 阴影投影距离: 沿 shadow_dir 方向的距离
    du = u - center_u
    dv = v - center_v
    proj = du * shadow_dir_u + dv * shadow_dir_v  # 沿阴影方向投影

    # 垂直于阴影方向的距离（决定是否在阴影带内）
    perp = torch.abs(-du * shadow_dir_v + dv * shadow_dir_u)

    # 阴影区域: proj > 0 (在光源对面), perp < shadow_width
    shadow_width = 0.08 + penumbra_width
    shadow_factor = torch.where(
        proj > 0,
        torch.where(
            perp < shadow_width,
            1.0 - torch.clamp((perp - (shadow_width - penumbra_width)) / penumbra_width, 0, 1),
            torch.tensor(0.0)
        ),
        torch.tensor(0.0)
    )
    # 只在有意义的范围内有阴影
    shadow_mask = torch.clamp(proj / shadow_length, 0, 1) * shadow_factor
    return shadow_mask


def _area_light(u, v, center_u, center_v, width, height, softness=0.03, amplitude=1.0):
    """
    面光源 — 模拟矩形区域光源 (如窗户、面板灯)
    softness 控制边缘柔和度
    """
    dist_u = (u - center_u) / (width / 2 + softness)
    dist_v = (v - center_v) / (height / 2 + softness)
    # Sigmoid 边缘过渡，模拟面光源柔和边界
    falloff_u = torch.sigmoid(-(torch.abs(dist_u) - 1) * 20)
    falloff_v = torch.sigmoid(-(torch.abs(dist_v) - 1) * 20)
    return amplitude * falloff_u * falloff_v


def _ambient_base(H: int, W: int, intensity=0.05, color=(0.8, 0.85, 0.9)):
    """
    环境光基底 — 模拟最低限度的间接光照
    真实 lightmap 中即使是阴影区域也不是完全黑的
    """
    base = torch.ones(H, W, 3) * intensity
    base[:, :, 0] *= color[0]
    base[:, :, 1] *= color[1]
    base[:, :, 2] *= color[2]
    return base


def _color_bleeding(u, v, wall_pos, wall_color, bleed_sigma=0.15, bleed_strength=0.15):
    """
    色彩渗透 — 模拟彩色墙面反射的间接光照
    wall_pos: 墙面位置 ('left', 'right', 'top', 'bottom')
    wall_color: (R, G, B) 墙面颜色
    """
    if wall_pos == 'left':
        dist = u  # 从左墙向右扩散
    elif wall_pos == 'right':
        dist = 1 - u
    elif wall_pos == 'top':
        dist = v
    elif wall_pos == 'bottom':
        dist = 1 - v
    else:
        dist = torch.zeros_like(u)

    bleed = bleed_strength * torch.exp(-(dist**2) / (2 * bleed_sigma**2))
    color_tensor = torch.tensor(wall_color)
    result = bleed.unsqueeze(-1) * color_tensor
    return result


def _checkerboard(u, v, scale=8, offset_u=0.0, offset_v=0.0):
    """
    棋盘格高频图案 — 模拟地面瓷砖/地砖反射
    scale: 每方向格子数 (越大频率越高)
    offset_u/v: 图案偏移 (模拟光源移动时阴影随动)
    """
    cu = ((u + offset_u) * scale).floor() % 2
    cv = ((v + offset_v) * scale).floor() % 2
    return ((cu + cv) % 2).float()


def _venetian_blind(u, v, num_slats=12, slat_width=0.04, gap=0.02,
                    angle=0.0, offset_v=0.0):
    """
    百叶窗/格栅阴影 — 窄条交替明暗, 高频空间切变
    num_slats:  百叶窗叶片数
    slat_width: 每叶片宽度 (暗区)
    gap:        间隙宽度 (亮区)
    angle:      倾斜角度 (模拟叶片翻转)
    offset_v:   时间偏移 (模拟叶片随时间翻转)
    """
    period = slat_width + gap
    # 旋转坐标
    u_rot = u * torch.cos(torch.tensor(angle)) - (v + offset_v) * torch.sin(torch.tensor(angle))
    v_rot = u * torch.sin(torch.tensor(angle)) + (v + offset_v) * torch.cos(torch.tensor(angle))
    # 条纹周期
    phase = (v_rot / period).floor()
    pos_in_period = v_rot - phase * period
    # 间隙(亮区) vs 叶片(暗区)
    bright = torch.where(pos_in_period < gap, torch.tensor(1.0), torch.tensor(0.0))
    return bright


def _caustic_pattern(u, v, time_phase, scale=4.0, intensity=0.6):
    """
    水面焦散图案 — 多频率正弦叠加, 模拟水面折射聚焦光
    time_phase: 时间相位 (焦散随时间流动)
    scale:      空间频率
    intensity:  焦散峰值强度
    """
    # 多频率叠加产生类焦散纹理
    c1 = torch.sin((u + time_phase * 0.3) * scale * 2 * pi) * torch.cos((v + time_phase * 0.2) * scale * 2 * pi)
    c2 = torch.sin((u * 1.7 - time_phase * 0.15) * scale * 1.5 * 2 * pi) * torch.cos((v * 1.3 + time_phase * 0.25) * scale * 1.5 * 2 * pi)
    c3 = torch.cos((u * 0.9 + v * 1.1 + time_phase * 0.1) * scale * 2 * 2 * pi)
    # 平方后亮暗对比更锐利 (焦散特征: 高亮细线 + 暗背景)
    caustic = (c1 + c2 + c3) ** 2
    caustic = caustic / caustic.max()  # 归一化到 [0,1]
    return caustic * intensity


def _stained_glass(u, v, time_phase, cell_size=0.15, 
                   colors=None, edge_width=0.01):
    """
    彩色玻璃/花窗图案 — Voronoi 式色块 + 细黑边框, 高频空间切变
    time_phase:  时间偏移 (模拟光源位置变化导致色块亮度变化)
    cell_size:   单元大小
    colors:      色块颜色列表 [(R,G,B), ...]
    edge_width:  边框宽度 (细黑线, 高频特征)
    """
    if colors is None:
        colors = [(0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.1, 0.9, 0.1),
                  (0.9, 0.9, 0.1), (0.9, 0.1, 0.9), (0.1, 0.9, 0.9),
                  (0.95, 0.6, 0.1), (0.6, 0.95, 0.3)]
    # 简化 Voronoi: 用网格中心 + 小偏移模拟不规则色块
    cu = ((u / cell_size).floor()).long()
    cv = ((v / cell_size).floor()).long()
    # 用 (cu + cv * 7) % N 来分配颜色, 模拟伪随机分布
    num_colors = len(colors)
    idx = (cu + cv * 7 + int(time_phase * 2)) % num_colors
    # 离中心距离 → 边框效果
    center_u = (cu.float() + 0.5) * cell_size
    center_v = (cv.float() + 0.5) * cell_size
    dist = torch.max(torch.abs(u - center_u), torch.abs(v - center_v))
    edge_mask = (dist > cell_size / 2 - edge_width).float()
    # 构建RGB图案
    pattern = torch.zeros(u.shape[0], u.shape[1], 3)
    for ci, col in enumerate(colors):
        mask = (idx == ci).float()
        pattern += mask.unsqueeze(-1) * torch.tensor(col).unsqueeze(0).unsqueeze(0)
    # 边框: 黑色细线
    pattern *= (1 - edge_mask.unsqueeze(-1))
    return pattern


def _smooth_noise(H: int, W: int, scale=0.02, seed=None):
    """
    低频平滑噪声 — 模拟光照细微不均匀（真实lightmap并非完全平滑）
    """
    if seed is not None:
        torch.manual_seed(seed)
    # 生成低分辨率噪声再上采样，得到平滑结果
    low_h, low_w = max(H // 8, 2), max(W // 8, 2)
    noise_low = torch.randn(low_h, low_w) * scale
    # 双线性上采样
    noise_up = torch.nn.functional.interpolate(
        noise_low.unsqueeze(0).unsqueeze(0), size=(H, W),
        mode='bilinear', align_corners=False
    ).squeeze()
    return noise_up


# =============================================================================
# 测试信号生成器
# =============================================================================

def generate_moving_point_light(
    num_frames: int = 12, H: int = 128, W: int = 128,
    light_color=(1.0, 0.95, 0.8),  # 暖白光
    ambient_color=(0.05, 0.05, 0.07),  # 冷调环境光
    shadow_enabled=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 1: 移动点光源
    — 模拟一个人持灯在房间中移动
    — 光斑位置随时间变化，带柔和阴影和色彩渗透
    — 最接近真实动态GI场景
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    # 环境基底
    ambient = _ambient_base(H, W, intensity=0.05, color=ambient_color)

    for i, t_val in enumerate(times):
        # 点光源沿椭圆轨迹移动（模拟人在房间走动）
        center_u = 0.5 + 0.35 * torch.cos(t_val * 2 * pi)
        center_v = 0.5 + 0.25 * torch.sin(t_val * 2 * pi)

        # 光斑强度随距离衰减 (点光源特征)
        sigma = 0.08 + 0.02 * torch.sin(t_val * 4 * pi)  # 微微变化
        spot = _gaussian_spot(u, v, center_u, center_v, sigma, amplitude=0.9)

        # 间接光照扩散（GI bounce — 更大范围、更柔和）
        bounce = _gaussian_spot(u, v, center_u, center_v, sigma=0.2, amplitude=0.2)

        # 柔和阴影投射
        if shadow_enabled:
            shadow_dir_u = center_u - 0.5
            shadow_dir_v = center_v - 0.5
            norm = torch.sqrt(shadow_dir_u**2 + shadow_dir_v**2 + 1e-8)
            shadow_dir_u /= norm
            shadow_dir_v /= norm
            shadow = _soft_shadow(u, v, center_u, center_v,
                                  shadow_dir_u, shadow_dir_v,
                                  penumbra_width=0.04, shadow_length=0.25)
            # 阴影降低间接光bounce
            bounce = bounce * (1 - shadow * 0.5)

        # 色彩渗透 — 墙面反射暖色光
        bleed = _color_bleeding(u, v, 'left', (0.9, 0.6, 0.3),
                                bleed_sigma=0.12, bleed_strength=0.08)

        # 低频噪声 (真实lightmap微不均匀)
        noise = _smooth_noise(H, W, scale=0.008, seed=i * 42)

        # 组合
        img = ambient.clone()
        color_tensor = torch.tensor(light_color)
        # 直接光
        img += spot.unsqueeze(-1) * color_tensor
        # 间接光bounce（颜色偏冷 — 多次反射后颜色趋于环境色）
        img += bounce.unsqueeze(-1) * torch.tensor([0.7, 0.75, 0.85])
        # 色彩渗透
        img += bleed
        # 噪声
        img += noise.unsqueeze(-1).repeat(1, 1, 3) * 0.5

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_flickering_light(
    num_frames: int = 12, H: int = 128, W: int = 128,
    light_color=(1.0, 0.75, 0.4),  # 橙黄色火焰光
    ambient_color=(0.03, 0.03, 0.05),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 2: 闪烁光源（烛光/火焰）
    — 光源位置固定但强度剧烈波动
    — 火焰色温偏暖 (橙黄)
    — 闪烁模式模拟真实火焰的低频+高频波动
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    ambient = _ambient_base(H, W, intensity=0.03, color=ambient_color)

    # 固定光源位置
    center_u, center_v = 0.5, 0.45

    for i, t_val in enumerate(times):
        # 真实火焰闪烁: 低频缓慢变化 + 高频快速抖动
        low_freq = 0.7 + 0.3 * torch.sin(t_val * 2 * pi)  # 缓慢明暗
        high_freq = 0.85 + 0.15 * torch.sin(t_val * 12 * pi)  # 快速抖动
        flicker = low_freq * high_freq
        # 随机扰动（每帧微偏移模拟真实不稳定）
        torch.manual_seed(i * 7 + 13)
        flicker *= (0.9 + 0.2 * torch.rand(1).item())

        # 光斑大小也随闪烁变化
        sigma = 0.06 + 0.015 * flicker

        spot = _gaussian_spot(u, v, center_u, center_v, sigma, amplitude=flicker * 0.85)
        bounce = _gaussian_spot(u, v, center_u, center_v, sigma=0.18, amplitude=flicker * 0.15)

        # 色彩随闪烁强度变化 — 火焰弱时更红，强时偏黄白
        r = min(1.0, light_color[0] * (0.8 + 0.2 * flicker))
        g = min(1.0, light_color[1] * (0.5 + 0.5 * flicker))
        b = min(1.0, light_color[2] * (0.3 + 0.7 * flicker))
        dynamic_color = torch.tensor([r, g, b])

        img = ambient.clone()
        img += spot.unsqueeze(-1) * dynamic_color
        img += bounce.unsqueeze(-1) * torch.tensor([r * 0.5, g * 0.5, b * 0.3])
        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_day_night_cycle(
    num_frames: int = 12, H: int = 128, W: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 3: 日夜循环
    — 模拟室外场景一天中光照变化
    — 日光: 强暖色 (从上方照射), 夜光: 弱冷蓝色
    — 中间过渡: 黄昏橙色渐变
    — 阴影方向随太阳角度旋转
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    for i, t_val in enumerate(times):
        # 日光强度曲线: 早上渐亮 -> 中午最强 -> 下午渐暗 -> 夜间微光
        day_factor = torch.sin(t_val * pi) ** 0.8  # 非对称，更自然
        night_factor = 1 - day_factor

        # 色温变化
        # 日光: 暖白 (1.0, 0.95, 0.85)
        # 黄昏: 橙红 (1.0, 0.6, 0.2)  — 在 t=0.3 和 t=0.7 附近增强
        sunset_boost = torch.exp(-((t_val - 0.3)**2) / 0.01) + torch.exp(-((t_val - 0.7)**2) / 0.01)

        r = day_factor * (0.85 + 0.15 * sunset_boost) + night_factor * 0.05
        g = day_factor * (0.8 + 0.05 * sunset_boost - 0.3 * sunset_boost) + night_factor * 0.05
        b = day_factor * (0.75 - 0.5 * sunset_boost) + night_factor * 0.12

        # 太阳位置随时间变化 (东升西落)
        sun_u = 0.1 + 0.8 * t_val  # 从左到右
        sun_v = 0.7 - 0.4 * torch.sin(t_val * pi)  # 弧形轨迹

        # 日光: 大面积柔和照射 (太阳是远距离大光源)
        sun_light = _gaussian_spot(u, v, sun_u, sun_v, sigma=0.3, amplitude=day_factor * 0.6)
        # 更大范围的间接散射
        sun_bounce = _gaussian_spot(u, v, sun_u, sun_v, sigma=0.5, amplitude=day_factor * 0.15)

        # 夜光: 弱蓝色从上方 (月光)
        moon_u, moon_v = 0.6, 0.2
        moon_light = _gaussian_spot(u, v, moon_u, moon_v, sigma=0.25, amplitude=night_factor * 0.08)

        # 阴影 (白天有明显的定向阴影)
        if day_factor > 0.1:
            shadow_dir_u = sun_u - 0.5
            shadow_dir_v = sun_v - 0.5
            norm = torch.sqrt(shadow_dir_u**2 + shadow_dir_v**2 + 1e-8)
            shadow_dir_u /= norm
            shadow_dir_v /= norm
            shadow = _soft_shadow(u, v, sun_u, sun_v,
                                  shadow_dir_u, shadow_dir_v,
                                  penumbra_width=0.06, shadow_length=0.35)
            sun_bounce = sun_bounce * (1 - shadow * 0.4)

        # 组合
        img = torch.zeros(H, W, 3)
        img += sun_light.unsqueeze(-1) * torch.tensor([r, g, b])
        img += sun_bounce.unsqueeze(-1) * torch.tensor([r * 0.6, g * 0.6, b * 0.7])
        img += moon_light.unsqueeze(-1) * torch.tensor([0.15, 0.18, 0.35])
        # 底部环境光
        img += torch.tensor([0.02 * r.item() + 0.01, 0.02 * g.item() + 0.01, 0.02 * b.item() + 0.02])

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_color_bleeding_scene(
    num_frames: int = 12, H: int = 128, W: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 4: 色彩渗透（Color Bleeding）
    — 重点展示彩色墙面反射间接光的经典GI现象
    — 左墙红色, 右墙蓝色, 中间白光照射
    — 反射光随时间微微变化（光源缓慢移动）
    — 这是GI最显著的特征之一
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    ambient = _ambient_base(H, W, intensity=0.02, color=(0.5, 0.5, 0.5))

    for i, t_val in enumerate(times):
        # 中间白色光源缓慢左右移动
        center_u = 0.45 + 0.1 * torch.sin(t_val * 2 * pi)
        center_v = 0.5

        # 白色直接光
        spot = _gaussian_spot(u, v, center_u, center_v, sigma=0.1, amplitude=0.7)

        # 红色墙色彩渗透 (左墙)
        red_bleed = _color_bleeding(u, v, 'left', (1.0, 0.2, 0.05),
                                    bleed_sigma=0.15, bleed_strength=0.25)
        # 距光源越近渗透越强
        red_bleed *= (1 + _gaussian_spot(u, v, 0.0, center_v, sigma=0.3, amplitude=0.3)).unsqueeze(-1)

        # 蓝色墙色彩渗透 (右墙)
        blue_bleed = _color_bleeding(u, v, 'right', (0.05, 0.2, 1.0),
                                     bleed_sigma=0.15, bleed_strength=0.25)
        blue_bleed *= (1 + _gaussian_spot(u, v, 1.0, center_v, sigma=0.3, amplitude=0.3)).unsqueeze(-1)

        # 绿色地面色彩渗透 (底部)
        green_bleed = _color_bleeding(u, v, 'bottom', (0.1, 0.5, 0.15),
                                      bleed_sigma=0.12, bleed_strength=0.12)

        # 间接bounce (整体柔和)
        bounce = _gaussian_spot(u, v, center_u, center_v, sigma=0.25, amplitude=0.15)

        img = ambient.clone()
        img += spot.unsqueeze(-1) * torch.tensor([0.95, 0.95, 0.9])
        img += bounce.unsqueeze(-1) * torch.tensor([0.6, 0.6, 0.65])
        img += red_bleed
        img += blue_bleed
        img += green_bleed

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_area_light_shadow(
    num_frames: int = 12, H: int = 128, W: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 5: 面光源与阴影
    — 模拟天花板面板灯 / 窗户光照
    — 面光源产生柔和阴影（半影区宽）
    — 光源位置/大小随时间微变
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    ambient = _ambient_base(H, W, intensity=0.04, color=(0.8, 0.85, 0.9))

    for i, t_val in enumerate(times):
        # 面光源中心缓慢移动
        center_u = 0.5 + 0.08 * torch.sin(t_val * pi)
        center_v = 0.35 + 0.05 * torch.cos(t_val * pi)

        # 面光源宽度随时间微变
        width = 0.25 + 0.05 * torch.sin(t_val * 2 * pi)
        height = 0.12 + 0.03 * torch.cos(t_val * 2 * pi)

        # 面光源照射
        area = _area_light(u, v, center_u, center_v, width, height,
                          softness=0.04, amplitude=0.85)

        # 面光源的间接bounce（更大范围扩散）
        bounce = _gaussian_spot(u, v, center_u, center_v,
                               sigma=0.3, amplitude=0.2)

        # 面光源柔和阴影 — 模拟物体在面光源下投射的宽半影阴影
        # 面光源阴影特点: 半影区很宽，过渡非常柔和
        shadow_center_u = center_u + 0.15
        shadow_center_v = center_v + 0.2
        du = u - shadow_center_u
        dv = v - shadow_center_v
        # 面光源阴影比点光源阴影更柔和
        shadow = torch.exp(-(du**2 + dv**2) / (2 * 0.06**2)) * 0.4
        shadow *= torch.where(dv > 0, torch.tensor(1.0), torch.tensor(0.0))  # 只在下方
        # 柔和过渡
        shadow = shadow * torch.sigmoid(dv * 30)

        # 组合
        img = ambient.clone()
        # 面光源色温偏冷白 (面板灯特征)
        img += area.unsqueeze(-1) * torch.tensor([0.95, 0.97, 1.0])
        img += bounce.unsqueeze(-1) * torch.tensor([0.7, 0.75, 0.85])
        # 阴影降低bounce
        img -= shadow.unsqueeze(-1) * torch.tensor([0.15, 0.12, 0.08])

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_multi_light_scene(
    num_frames: int = 12, H: int = 128, W: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 6: 多光源叠加
    — 模拟房间中多个灯光 (如走廊、大厅)
    — 3个光源: 暖色主灯 + 冷色辅灯 + 红色装饰灯
    — 各光源独立变化 (开关、移动、闪烁)
    — 真实场景中常见多光源交互
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    ambient = _ambient_base(H, W, intensity=0.02, color=(0.5, 0.5, 0.55))

    for i, t_val in enumerate(times):
        # 主灯: 暖白色, 固定位置, 缓慢变亮
        main_intensity = 0.5 + 0.3 * torch.sin(t_val * pi)
        main_spot = _gaussian_spot(u, v, 0.3, 0.4, sigma=0.12, amplitude=main_intensity)
        main_bounce = _gaussian_spot(u, v, 0.3, 0.4, sigma=0.25, amplitude=main_intensity * 0.15)

        # 辅灯: 冷白色, 从右侧移动
        aux_u = 0.7 + 0.1 * torch.sin(t_val * 2 * pi)
        aux_spot = _gaussian_spot(u, v, aux_u, 0.6, sigma=0.08, amplitude=0.4)
        aux_bounce = _gaussian_spot(u, v, aux_u, 0.6, sigma=0.2, amplitude=0.08)

        # 装饰灯: 红色, 闪烁开关 (周期性开关)
        decor_on = (torch.sin(t_val * 4 * pi) > 0).float()
        decor_spot = _gaussian_spot(u, v, 0.5, 0.8, sigma=0.06, amplitude=0.35 * decor_on)

        # 组合 — 注意多光源叠加后的色彩混合
        img = ambient.clone()
        img += main_spot.unsqueeze(-1) * torch.tensor([1.0, 0.92, 0.8])
        img += main_bounce.unsqueeze(-1) * torch.tensor([0.7, 0.65, 0.55])
        img += aux_spot.unsqueeze(-1) * torch.tensor([0.85, 0.9, 1.0])
        img += aux_bounce.unsqueeze(-1) * torch.tensor([0.5, 0.55, 0.7])
        img += decor_spot.unsqueeze(-1) * torch.tensor([1.0, 0.15, 0.05])

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_switching_light(
    num_frames: int = 12, H: int = 128, W: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 7: 开关灯（离散状态切换）
    — 模拟房间灯突然开关
    — 开灯后间接光逐渐填充（GI需要几帧才能"建立"）
    — 关灯后间接光逐渐消退（GI残光/衰退）
    — 这是动态GI最典型的测试场景
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    ambient = _ambient_base(H, W, intensity=0.01, color=(0.2, 0.2, 0.3))

    for i, t_val in enumerate(times):
        # 灯状态: 前1/3时间开, 中间1/3关, 后1/3开
        if t_val < 0.33:
            # 开灯阶段 — 逐渐增亮
            ramp = torch.sigmoid((t_val - 0.05) * 30)  # 快速开启
            gi_buildup = torch.sigmoid((t_val - 0.1) * 10)  # GI慢于直接光
        elif t_val < 0.66:
            # 关灯阶段 — GI残光衰退
            ramp = torch.sigmoid((0.33 - t_val) * 30)  # 快速关闭
            # GI衰退比直接光慢 (间接光有余辉)
            gi_decay = torch.exp(-((t_val - 0.33) * 5)) * 0.8
        else:
            # 再次开灯
            ramp = torch.sigmoid((t_val - 0.68) * 30)
            gi_buildup = torch.sigmoid((t_val - 0.72) * 10)

        # 直接光 (中心上方)
        center_u, center_v = 0.5, 0.35
        direct = _gaussian_spot(u, v, center_u, center_v, sigma=0.1, amplitude=ramp * 0.8)

        # 间接光 (GI) — 建立和衰退有不同速度
        if t_val < 0.33 or t_val >= 0.66:
            gi_factor = gi_buildup
        else:
            gi_factor = gi_decay

        gi_bounce = _gaussian_spot(u, v, center_u, center_v, sigma=0.3, amplitude=gi_factor * 0.2)
        # 色彩渗透也需要时间建立
        bleed = _color_bleeding(u, v, 'left', (0.9, 0.5, 0.2),
                                bleed_sigma=0.15, bleed_strength=gi_factor * 0.1)

        img = ambient.clone()
        img += direct.unsqueeze(-1) * torch.tensor([1.0, 0.95, 0.85])
        img += gi_bounce.unsqueeze(-1) * torch.tensor([0.7, 0.7, 0.75])
        img += bleed

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


def generate_sunlight_through_window(
    num_frames: int = 12, H: int = 128, W: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    场景 8: 窗户光照
    — 模拟阳光透过窗户照进室内
    — 窗户是矩形光区, 光束角度随时间变化 (太阳移动)
    — 窗框投射矩形阴影
    — 室内只有窗户方向有光, 其他区域靠间接bounce
    — 这是最经典的室内lightmap场景
    """
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    u, v = _make_uv_grid(H, W)

    ambient = _ambient_base(H, W, intensity=0.015, color=(0.4, 0.45, 0.55))

    for i, t_val in enumerate(times):
        # 太阳强度 (中午最强)
        sun_intensity = 0.5 + 0.4 * torch.sin(t_val * pi)

        # 窗户位置 (左墙上方)
        win_center_u = 0.12
        win_center_v = 0.3

        # 窗户宽度随太阳角度变化 (太阳越高, 光束越窄)
        win_width = 0.06 + 0.02 * torch.sin(t_val * pi)
        win_height = 0.15

        # 窗户直射光
        window_light = _area_light(u, v, win_center_u, win_center_v,
                                   win_width, win_height,
                                   softness=0.02, amplitude=sun_intensity * 0.9)

        # 光束投射到地面 — 随太阳角度移动
        # 太阳低时光束远, 太阳高时光束近窗户
        beam_u = win_center_u + 0.3 * (1 - torch.sin(t_val * pi))
        beam_v = 0.6
        beam = _area_light(u, v, beam_u, beam_v,
                          width=0.15 + 0.1 * torch.cos(t_val * pi),
                          height=0.1,
                          softness=0.05, amplitude=sun_intensity * 0.6)

        # 间接bounce (室内整体微亮)
        bounce = _gaussian_spot(u, v, 0.3, 0.5, sigma=0.4, amplitude=sun_intensity * 0.08)

        # 窗框阴影条纹 (竖条纹)
        shadow_strip1 = torch.where(
            (u > win_center_u + 0.015) & (u < win_center_u + 0.025),
            torch.tensor(0.7), torch.tensor(0.0)
        ) * sun_intensity
        shadow_strip2 = torch.where(
            (u > win_center_u + 0.04) & (u < win_center_u + 0.05),
            torch.tensor(0.5), torch.tensor(0.0)
        ) * sun_intensity

        # 色温: 日光偏暖, 但窗户光比室内更冷白
        img = ambient.clone()
        img += window_light.unsqueeze(-1) * torch.tensor([1.0, 0.95, 0.88])
        img += beam.unsqueeze(-1) * torch.tensor([0.95, 0.9, 0.8])
        img += bounce.unsqueeze(-1) * torch.tensor([0.65, 0.65, 0.75])
        # 减去窗框阴影
        img -= shadow_strip1.unsqueeze(-1) * torch.tensor([0.3, 0.3, 0.25])
        img -= shadow_strip2.unsqueeze(-1) * torch.tensor([0.2, 0.2, 0.15])

        lightmaps[i] = torch.clamp(img, 0, 1)

    return lightmaps, times


# =============================================================================
# 高频信号生成器 — 空间/时间高频, 挑战 NDGI 压缩与重建
# =============================================================================

def generate_checkerboard_shadow(num_frames=12, H=128, W=128):
    """
    棋盘格地面阴影 — 地面瓷砖反射 + 移动点光源
    高频特征: 棋盘格交替明暗 (像素级切变)
    时间变化: 光源移动导致阴影偏移, 棋盘格亮度随时间变化
    """
    u, v = _make_uv_grid(H, W)
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    
    for i, t_val in enumerate(times):
        ambient = _ambient_base(H, W, intensity=0.03, color=(0.7, 0.75, 0.8))
        
        # 移动点光源 (椭圆轨迹)
        center_u = 0.3 + 0.35 * torch.cos(t_val * 2 * pi)
        center_v = 0.5 + 0.2 * torch.sin(t_val * 2 * pi)
        light = _gaussian_spot(u, v, center_u.item(), center_v.item(), sigma=0.12, amplitude=0.8)
        
        # 柔和阴影
        shadow_dir_u = 0.3
        shadow_dir_v = 0.6
        shadow = _soft_shadow(u, v, center_u.item(), center_v.item(),
                              shadow_dir_u, shadow_dir_v,
                              penumbra_width=0.03, shadow_length=0.25)
        
        # 棋盘格地面纹理 (高频空间切变)
        checker = _checkerboard(u, v, scale=10) * 0.3  # 高频纹理
        # 棋盘格亮度受光照影响
        checker_bright = checker * (1 + light * 0.5)
        
        # 组合: 光照 × (1 - 阴影) + 棋盘格纹理 + 环境光
        diffuse = light * (1 - shadow * 0.7)
        img = ambient.clone()
        img += diffuse.unsqueeze(-1) * torch.tensor([1.0, 0.95, 0.88])
        img += checker_bright.unsqueeze(-1) * torch.tensor([0.6, 0.65, 0.7])
        # 阴影区域棋盘格更暗
        shadow_zone = (shadow > 0.1).float()
        img -= shadow_zone.unsqueeze(-1) * checker.unsqueeze(-1) * torch.tensor([0.15, 0.15, 0.12])
        
        lightmaps[i] = torch.clamp(img, 0, 1)
    
    return lightmaps, times


def generate_venetian_blind(num_frames=12, H=128, W=128):
    """
    百叶窗光影 — 百叶窗叶片产生的窄条纹交替明暗
    高频特征: 等间距窄条明暗切变 (空间高频)
    时间变化: 叶片随时间缓慢翻转, 条纹角度/宽度变化
    """
    u, v = _make_uv_grid(H, W)
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    
    for i, t_val in enumerate(times):
        ambient = _ambient_base(H, W, intensity=0.04, color=(0.8, 0.85, 0.9))
        
        # 百叶窗叶片随时间翻转: 角度从 0 → 30° → 0
        angle = 0.5 * torch.sin(t_val * 2 * pi)  # 最大约 0.5 rad ≈ 28°
        # 间隙随角度变化 (叶片翻转时透光量变化)
        gap = 0.025 + 0.015 * torch.cos(t_val * 2 * pi)  # 2.5% ~ 4%
        slat_width = 0.06
        
        blind = _venetian_blind(u, v, num_slats=14, slat_width=slat_width,
                                 gap=gap.item(), angle=angle.item())
        
        # 窗户光源 (上方)
        window = _area_light(u, v, 0.5, 0.15, 0.8, 0.08, softness=0.02, amplitude=0.6)
        
        # 百叶窗条纹 × 窗户光 = 条纹光斑
        striped_light = blind * window
        # 散布的间接光 (百叶窗透过的光在地面散射)
        bounce = _gaussian_spot(u, v, 0.5, 0.7, sigma=0.3, amplitude=0.12) * blind
        
        # 色温: 窗户光偏冷白, 间接光偏暖
        img = ambient.clone()
        img += striped_light.unsqueeze(-1) * torch.tensor([0.95, 0.97, 1.0])
        img += bounce.unsqueeze(-1) * torch.tensor([0.85, 0.8, 0.75])
        
        lightmaps[i] = torch.clamp(img, 0, 1)
    
    return lightmaps, times


def generate_stained_glass(num_frames=12, H=128, W=128):
    """
    彩色花窗光 — Voronoi 式色块 + 细黑边框, 模拟教堂/装饰花窗
    高频特征: 色块边界细黑线 (空间极高频), 多色快速切变
    时间变化: 光源移动导致色块亮度随时间变化, 焦点移动
    """
    u, v = _make_uv_grid(H, W)
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    
    for i, t_val in enumerate(times):
        # 花窗图案 (高频空间切变 + 细边框)
        glass = _stained_glass(u, v, time_phase=t_val.item(),
                               cell_size=0.12, edge_width=0.008)
        
        # 窗外光源强度随时间变化 (云层遮挡)
        sun_intensity = 0.7 + 0.3 * torch.sin(t_val * 3 * pi + 0.5)
        
        # 透过花窗的光在地面的投影 (柔和)
        center_u = (0.5 + 0.1 * torch.sin(t_val * 2 * pi)).item()
        center_v = 0.55
        focus = _gaussian_spot(u, v, center_u, center_v, sigma=0.2, amplitude=0.3)
        
        # 组合: 花窗图案 × 光照强度 + 地面散射 + 环境光
        ambient = _ambient_base(H, W, intensity=0.03, color=(0.85, 0.88, 0.9))
        img = ambient.clone()
        img += glass * sun_intensity.item()
        img += focus.unsqueeze(-1) * glass * 0.4  # 色块聚焦光斑
        
        lightmaps[i] = torch.clamp(img, 0, 1)
    
    return lightmaps, times


def generate_caustics_light(num_frames=12, H=128, W=128):
    """
    水面焦散光 — 模拟水面折射在地面产生的聚焦光纹
    高频特征: 多频率正弦叠加产生锐利明暗线纹 (空间高频)
    时间变化: 焦散纹样随时间流动变形, 水波持续运动
    """
    u, v = _make_uv_grid(H, W)
    times = torch.linspace(0, 1, num_frames)
    lightmaps = torch.zeros(num_frames, H, W, 3)
    
    for i, t_val in enumerate(times):
        ambient = _ambient_base(H, W, intensity=0.04, color=(0.75, 0.82, 0.88))
        
        # 焦散图案 (高频空间纹样, 随时间流动)
        caustic = _caustic_pattern(u, v, time_phase=t_val.item(), scale=5.0, intensity=0.5)
        
        # 水面整体反射 (蓝绿色调, 柔和底色)
        water_reflect = 0.08 * torch.ones(H, W, 3)
        water_reflect[:, :, 0] *= 0.6  # 偏蓝绿
        water_reflect[:, :, 1] *= 0.9
        water_reflect[:, :, 2] *= 1.0
        
        # 焦散聚焦光斑 (局部更亮, 随时间移动)
        focus_u = 0.4 + 0.15 * torch.sin(t_val * 2 * pi)
        focus_v = 0.5 + 0.1 * torch.cos(t_val * 1.5 * pi)
        focus = _gaussian_spot(u, v, focus_u.item(), focus_v.item(), sigma=0.25, amplitude=0.15)
        
        # 组合: 焦散纹样(暖白) + 水面底色(蓝绿) + 聚焦区域 + 环境光
        img = ambient.clone()
        img += water_reflect
        img += caustic.unsqueeze(-1) * torch.tensor([1.0, 0.95, 0.85])  # 焦散偏暖白
        img += focus.unsqueeze(-1) * torch.tensor([0.9, 0.85, 0.75]) * (1 + caustic.unsqueeze(-1))
        
        lightmaps[i] = torch.clamp(img, 0, 1)
    
    return lightmaps, times

SIGNAL_REGISTRY: Dict[str, callable] = {
    'moving_point_light': generate_moving_point_light,
    'flickering_light':   generate_flickering_light,
    'day_night_cycle':    generate_day_night_cycle,
    'color_bleeding':     generate_color_bleeding_scene,
    'area_light_shadow':  generate_area_light_shadow,
    'multi_light':        generate_multi_light_scene,
    'switching_light':    generate_switching_light,
    'sunlight_window':    generate_sunlight_through_window,
    'checkerboard_shadow': generate_checkerboard_shadow,
    'venetian_blind':     generate_venetian_blind,
    'stained_glass':      generate_stained_glass,
    'caustics_light':     generate_caustics_light,
}

SIGNAL_DESCRIPTIONS: Dict[str, str] = {
    'moving_point_light': '移动点光源 — 模拟持灯走动, 椭圆轨迹+阴影+色彩渗透',
    'flickering_light':   '闪烁烛光 — 火焰低频+高频闪烁, 橙黄色温',
    'day_night_cycle':    '日夜循环 — 日光暖色→黄昏→月光冷蓝, 阴影旋转',
    'color_bleeding':     '色彩渗透 — 红/蓝墙反射, GI经典特征',
    'area_light_shadow':  '面光源阴影 — 面板灯柔和半影, 矩形光源',
    'multi_light':        '多光源叠加 — 暖主灯+冷辅灯+红装饰灯, 独立变化',
    'switching_light':    '开关灯切换 — GI建立/衰退时序, 动态GI核心场景',
    'sunlight_window':    '窗户日光 — 矩形光束+窗框阴影, 室内经典场景',
    'checkerboard_shadow': '棋盘格阴影 — 地面瓷砖反射+移动光源, 空间高频切变',
    'venetian_blind':     '百叶窗光影 — 窄条明暗交替+叶片翻转, 空间高频',
    'stained_glass':      '彩色花窗光 — Voronoi色块+细黑边框, 极高频边界',
    'caustics_light':     '水面焦散光 — 多频正弦聚焦纹样, 时间流动高频',
}


def generate_all_signals(
    num_frames: int = 12, H: int = 128, W: int = 128
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """批量生成所有测试信号"""
    results = {}
    for name, func in SIGNAL_REGISTRY.items():
        print(f"  Generating: {name} ...")
        lightmaps, times = func(num_frames=num_frames, H=H, W=W)
        results[name] = (lightmaps, times)
    return results


# =============================================================================
# 可视化
# =============================================================================

def visualize_signal_set(
    signals: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    max_show: int = 6,
    save_path: str = 'test_signal_preview.png'
):
    """
    预览所有测试信号 — 每种场景一行, 选取关键帧展示
    """
    names = list(signals.keys())[:max_show]
    T_sample = signals[names[0]][0].shape[0]
    # 选取4个关键帧 (开头, 1/3, 2/3, 结尾)
    frame_indices = [0, T_sample // 3, 2 * T_sample // 3, T_sample - 1]

    num_rows = len(names)
    num_cols = len(frame_indices)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for row, name in enumerate(names):
        lightmaps, times = signals[name]
        desc = SIGNAL_DESCRIPTIONS.get(name, name)
        for col, fi in enumerate(frame_indices):
            ax = axes[row, col]
            img = lightmaps[fi].numpy()
            ax.imshow(np.clip(img, 0, 1))
            t_val = times[fi].item()
            ax.set_title(f't={t_val:.2f}', fontsize=8)
            ax.axis('off')
        # 行标题
        axes[row, 0].set_ylabel(desc, fontsize=8, rotation=0, labelpad=120,
                                 va='center', fontweight='bold')

    fig.suptitle('NDGI 测试信号集预览 — 真实 Lightmap 模拟', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n预览保存至: {save_path}")


def visualize_single_signal(
    lightmaps: torch.Tensor, times: torch.Tensor,
    name: str = '', save_path: Optional[str] = None
):
    """详细可视化单个信号: 全帧展示 + RGB通道统计"""
    T, H, W, C = lightmaps.shape
    desc = SIGNAL_DESCRIPTIONS.get(name, name)

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(2, max(T, 1) + 1, figure=fig,
                  width_ratios=[1] * max(T, 1) + [0.6],
                  hspace=0.35, wspace=0.15)

    # 上行: RGB全帧
    for ti in range(T):
        ax = fig.add_subplot(gs[0, ti])
        ax.imshow(np.clip(lightmaps[ti].numpy(), 0, 1))
        ax.set_title(f't={times[ti].item():.2f}', fontsize=8)
        ax.axis('off')

    # 下行: 亮度图 (Luminance)
    for ti in range(T):
        ax = fig.add_subplot(gs[1, ti])
        lum = 0.2126 * lightmaps[ti, :, :, 0] + \
              0.7152 * lightmaps[ti, :, :, 1] + \
              0.0722 * lightmaps[ti, :, :, 2]
        ax.imshow(lum.numpy(), cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Lum', fontsize=8)
        ax.axis('off')

    # 右侧: RGB统计
    ax_stats = fig.add_subplot(gs[:, -1])
    ax_stats.axis('off')
    r_mean = lightmaps[:, :, :, 0].mean().item()
    g_mean = lightmaps[:, :, :, 1].mean().item()
    b_mean = lightmaps[:, :, :, 2].mean().item()
    r_max = lightmaps[:, :, :, 0].max().item()
    g_max = lightmaps[:, :, :, 1].max().item()
    b_max = lightmaps[:, :, :, 2].max().item()
    r_min = lightmaps[:, :, :, 0].min().item()
    g_min = lightmaps[:, :, :, 1].min().item()
    b_min = lightmaps[:, :, :, 2].min().item()
    txt = (
        f"{desc}\n\n"
        f"Shape: ({T}, {H}, {W}, 3)\n\n"
        f"R: min={r_min:.3f} mean={r_mean:.3f} max={r_max:.3f}\n"
        f"G: min={g_min:.3f} mean={g_mean:.3f} max={g_max:.3f}\n"
        f"B: min={b_min:.3f} mean={b_mean:.3f} max={b_max:.3f}\n\n"
        f"Times: [{times[0].item():.2f} .. {times[-1].item():.2f}]"
    )
    ax_stats.text(0.5, 0.5, txt, transform=ax_stats.transAxes,
                  fontsize=9, va='center', ha='center', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                            edgecolor='gray', alpha=0.9))

    fig.suptitle(f'NDGI 测试信号: {name}', fontsize=12, fontweight='bold')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存至: {save_path}")
    plt.show()


# =============================================================================
# Main: 生成并预览
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NDGI 测试信号集 — 真实 Lightmap 模拟")
    print("=" * 60)

    # 生成参数 (匹配 NDGI_demo.py 的默认配置)
    NUM_FRAMES = 12   # 与 f3d_uvt_resolution 的 temporal dim 匹配
    H, W = 128, 128   # 与 f2d_uv_resolution 匹配

    print(f"\n参数: T={NUM_FRAMES}, H={H}, W={W}")
    print(f"信号类型: {len(SIGNAL_REGISTRY)} 种\n")

    # 批量生成
    signals = generate_all_signals(num_frames=NUM_FRAMES, H=H, W=W)

    # 打印统计信息
    print("\n--- 信号统计 ---")
    for name, (lm, ts) in signals.items():
        desc = SIGNAL_DESCRIPTIONS[name]
        r_range = (lm[:, :, :, 0].min().item(), lm[:, :, :, 0].max().item())
        g_range = (lm[:, :, :, 1].min().item(), lm[:, :, :, 1].max().item())
        b_range = (lm[:, :, :, 2].min().item(), lm[:, :, :, 2].max().item())
        print(f"  {name:20s}: R[{r_range[0]:.2f},{r_range[1]:.2f}] "
              f"G[{g_range[0]:.2f},{g_range[1]:.2f}] "
              f"B[{b_range[0]:.2f},{b_range[1]:.2f}]")
        print(f"    {desc}")

    # 预览
    print("\n--- 生成预览 ---")
    visualize_signal_set(signals, save_path='test_signal_preview.png')

    # 详细展示每种信号 (可选)
    # for name, (lm, ts) in signals.items():
    #     visualize_single_signal(lm, ts, name)

    print("\n完成! 所有信号已生成并保存预览图。")
    print("使用方式: 将 signals[name][0] 作为 lightmaps, signals[name][1] 作为 times")
    print("          传入 TemporalLightmapDataset(lightmaps, times)")
