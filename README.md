# NDGI — Neural Dynamic GI

> PyTorch implementation of **Neural Dynamic GI: Random-Access Neural Compression for Temporal Lightmaps in Dynamic Lighting Environments**

NDGI 是一种面向动态光照环境的神经压缩方法，将多套时序 Lightmap 压缩为紧凑的神经隐式表示，支持随机访问解码，最终码率仅为原始数据的不到 1%。

## 核心思想

传统动态 GI 为不同光照条件预烘焙多套 Lightmap，运行时在两套之间插值。存储和内存随光照采样点数量线性爆炸。

NDGI 的替代方案：不直接存多套图，而是将整个时序 Lightmap 集合压缩为一个参数集 $\Theta$，让光照值变为可查询函数：

$$
I(u, v, t) = G_\Phi\big(V_{uvt}, V_{uv}, V_{ut}, V_{vt}, \gamma(t)\big)
$$

## 模型架构

### 混合特征图（Hybrid Feature Maps）

| 特征图 | 维度 | 通道 | 作用 |
|--------|------|------|------|
| $F_{uv}^{2D}$ | 128×128 | 4 | 捕获空间静态细节（几何遮挡、间接光分布） |
| $F_{ut}^{2D}$ | 64×24 | 2 | 沿 u-t 平面的相关性 |
| $F_{vt}^{2D}$ | 64×24 | 2 | 沿 v-t 平面的相关性 |
| $F_{uvt}^{3D}$ | 32×32×12 | 4 | 捕获时空联合亮度变化（体素化时序变化） |

参数集：$\Theta = \{F_{uvt}^{3D}, F_{uv}^{2D}, F_{ut}^{2D}, F_{vt}^{2D}, \Phi\}$

### 时间位置编码

$$
\gamma(t) = [\sin(2^0 \pi t), \cos(2^0 \pi t), \sin(2^1 \pi t), \cos(2^1 \pi t)]
$$

2 阶傅里叶特征将标量时间 $t \in [0, 1]$ 映射到 4 维，使 MLP 更容易拟合时间域变化。

### MLP 解码器

极轻量 MLP：2 个隐藏层（hidden size 16~64），GELU 激活，输出 float RGB。

### BC 压缩仿真

对大块特征 $F_{uv}^{2D}$ 和 $F_{uvt}^{3D}$ 采用 BC7 压缩仿真训练策略——不直接优化像素值，而是优化端点 $E$ 和权重 $W$：

$$
f_p = (1 - w_p) e_1 + w_p e_2
$$

对 $F_{ut}^{2D}$ 和 $F_{vt}^{2D}$ 采用 8-bit 量化噪声模拟：

$$
\tilde{V} = V + \mathcal{U}(-0.5, 0.5) \cdot \frac{1}{256}
$$

### 压缩效果

| 阶段 | BPP | 相对原始 |
|------|-----|----------|
| 原始（未压） | 4.62 | 4.8% |
| + 8-bit 量化 | 2.32 | 2.4% |
| + 8-bit 量化 & BC7 | **0.68** | **0.7%** |

## 项目结构

```
NDGI/
├── NDGI_demo.py           # 主入口：完整 NDGI 模型 + 训练 + 评估 + 可视化
├── test_signal.py         # 测试信号集：12 种模拟真实 Lightmap 的时序光照数据
├── legency/
│   ├── NDGI.py            # 初版实现（BC 仿真 + 量化噪声 + Gamma 校正）
│   ├── NDGI2.py           # 改进版（MLP+PE 显式时间编码）
│   ├── NDGI_demo.py       # 初版演示脚本
│   ├── reference.md       # 论文方法总结与数学原理详解
│   ├── NDGI2.md           # NDGI2 方法简述
│   └── ndgi_numpy_reference.py  # NumPy 参考实现
│   └── *.png              # 各版本运行结果图
```

## 快速开始

### 安装依赖

```bash
pip install torch numpy matplotlib
```

### 运行演示

```bash
# 使用默认信号（棋盘格阴影）
python NDGI_demo.py

# 指定测试信号
python NDGI_demo.py --signal moving_point_light

# 查看所有可用信号
python NDGI_demo.py --list-signals
```

### 可用测试信号

| 信号名 | 描述 |
|--------|------|
| `moving_point_light` | 移动点光源 — 椭圆轨迹+阴影+色彩渗透 |
| `flickering_light` | 闪烁烛光 — 火焰低频+高频闪烁 |
| `day_night_cycle` | 日夜循环 — 日光→黄昏→月光 |
| `color_bleeding` | 色彩渗透 — 红/蓝墙反射，GI经典特征 |
| `area_light_shadow` | 面光源阴影 — 面板灯柔和半影 |
| `multi_light` | 多光源叠加 — 暖主灯+冷辅灯+红装饰灯 |
| `switching_light` | 开关灯切换 — GI建立/衰退时序 |
| `sunlight_window` | 窗户日光 — 矩形光束+窗框阴影 |
| `checkerboard_shadow` | 棋盘格阴影 — 高频空间切变 |
| `venetian_blind` | 百叶窗光影 — 窄条明暗交替 |
| `stained_glass` | 彩色花窗光 — Voronoi色块+细黑边框 |
| `caustics_light` | 水面焦散光 — 多频正弦聚焦纹样 |

## 训练策略

采用两阶段训练：

- **Phase 1**：全参数训练，使用 BC 仿真 + 量化噪声，Adam lr=1e-3，CosineAnnealingLR
- **Phase 2**：冻结特征图，仅 fine-tune MLP 解码器，lr 降低 10×，在量化+BC仿真条件下微调

预处理：Gamma 校正（增强暗部细节）+ per-channel mean normalization。

## 运行时解码

NDGI 与 Virtual Texturing（VT）管线集成：

> 每帧可见区域 → VT 反馈 → 按需选 tile → compute shader 跑 $G_\Phi$ → 写 physical texture → 正常采样

实测：解一个 1024×1024 Lightmap，NDGI Medium 仅 **0.203 ms** on RTX 4060。

## 参考文献

> **Neural Dynamic GI: Random-Access Neural Compression for Temporal Lightmaps in Dynamic Lighting Environments**
>
> Paper: [arxiv.org/abs/2604.12625](https://arxiv.org/abs/2604.12625v2)
