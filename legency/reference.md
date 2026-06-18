# Neural Dynamic GI (NDGI) — 方法总结与数学原理详解

## 一、核心要解决的问题

传统动态 GI 的做法是：为不同光照条件（time-of-day、光源开关等）预烘焙多套 lightmap，运行时在两套之间插值：

\[
\mathcal{L} = \{L_i \mid i = 0, 1, \dots, n-1\}, \quad L_i \in \mathbb{R}^{H \times W \times C}
\]

在渲染时，给定纹理坐标 \((u, v)\) 和时间 \(t \in (t_{i-1}, t_i)\)：

\[
I(u, v, t) = \text{Interp}_{[t_{i-1}, t_i]}\big[L_{i-1}(u, v), L_i(u, v)\big](t) \tag{2}
\]

这套方案的致命问题是：**存储和内存随光照采样点数量 \(n\) 线性爆炸**，大规模场景根本吃不消。

---

## 二、核心思想：从"存多套图" → "学一个紧凑模型"

NDGI 的根本替代方案是：不直接存 \(\mathcal{L}\)，而是把整个时序 lightmap 集合压缩成一个参数集 \(\Theta\)，让光照值变为一个可查询函数：

\[
I(u, v, t) = \mathcal{H}_\Theta(u, v, t) \tag{3}
\]

其中 \(\Theta\) 不再包含原始 lightmap 像素，而是由**少量特征图 + 一个轻量 MLP** 构成——本质上是一个 \((u, v, t) \mapsto \text{RGB}\) 的神经隐式表示，但做了大量工程化改造以适配实时渲染的随机访问和带宽约束。

---

## 三、Hybrid Feature Map 表征（Section 3.3）——最核心的设计

### 3.1 为什么单张 2D feature map 不够？

直觉：lightmap 沿空间维度 \((u, v)\) 有精细的静态几何遮挡细节（低频/中频空间信号），而沿时间 \(t\) 的变化来自太阳角度/天光/光源的高频变化。如果用一张 \(F_{uv}^{2D}\) 同时编码空间和时间的共性，MLP decoder 就得自己"记住"所有时间变化 → 要么特征图做得很大，要么质量崩。

### 3.2 混合特征结构

作者将特征分解为四个分量，形成 **tri-plane + 3D 体素的混合**：

| 特征图 | 维度 | 通道 | 作用 |
|--------|------|------|------|
| \(F_{uv}^{2D}\) | 128×128 | 4 ch | 捕获空间静态细节（几何遮挡、间接光分布） |
| \(F_{ut}^{2D}\) | 64×24 | 2 ch | 沿 \(u\)-\(t\) 平面的相关性（u 方向的空间 × 时间） |
| \(F_{vt}^{2D}\) | 64×24 | 2 ch | 沿 \(v\)-\(t\) 平面的相关性 |
| \(F_{uvt}^{3D}\) | 32×32×12 | 4 ch | 捕获时空联合的亮度变化（体素化时序变化） |

最终压缩后的参数集：

\[
\mathcal{L} \longrightarrow \Theta = \{F_{uvt}^{3D}, F_{uv}^{2D}, F_{ut}^{2D}, F_{vt}^{2D}, \Phi\} \tag{5}
\]

其中 \(\Phi\) 是 decoder MLP \(G_\Phi\) 的参数。

### 3.3 查询与解码的数学过程

给定运行时的查询坐标 \((u, v, t)\)：

**Step 1 — 采样各特征向量：**

从 3D 特征图中三线性插值采样：

\[
V_{uvt} = \text{sample}\big(F_{uvt}^{3D}, (u, v, t)\big) \in \mathbb{R}^4
\]

从三个 tri-plane 投影面分别双线性采样：

\[
\begin{aligned}
V_{uv} &= \text{sample}\big(F_{uv}^{2D}, (u, v)\big) \in \mathbb{R}^4 \\
V_{ut} &= \text{sample}\big(F_{ut}^{2D}, (u, t)\big) \in \mathbb{R}^2 \\
V_{vt} &= \text{sample}\big(F_{vt}^{2D}, (v, t)\big) \in \mathbb{R}^2
\end{aligned}
\]

**Step 2 — 时间的位置编码（Positional Encoding）：**

\[
\gamma(t) = \big[\sin(2^0 \pi t), \cos(2^0 \pi t), \sin(2^1 \pi t), \cos(2^1 \pi t)\big]
\]

即用 **2 阶傅里叶特征** 把标量时间 \(t \in [0, 1]\) 映射到 4 维，使 MLP 能更容易拟合时间域的周期性/高频变化（类似 NeRF 的 positional encoding 思路）。

**Step 3 — MLP 解码：**

\[
I(u, v, t) = G_\Phi\big(V_{uvt}, V_{uv}, V_{ut}, V_{vt}, \gamma(t)\big) \tag{4}
\]

\(G_\Phi\) 是一个**极轻量的 MLP**：2 个隐藏层，hidden size 仅 16~64，GELU 激活，输出层无激活，输出 float RGB。

> **为什么 tri-plane 分解有效？** —— 如果把 \((u, v, t)\) 的联合信号直接存 3D 体素，分辨率需求是 \(O(R^3)\)；tri-plane 将其拆成三个低维投影面，存储复杂度降为 \(O(R^2 \cdot R_t + R^2)\)，用少量额外平面换取对高频交叉项的表达能力。Section 12 的消融实验（Table 8）验证了 tri-plane vs. 单纯 8ch \(F_{uv}^{2D}\)：同等质量下 BPP 从 0.86 → 0.67，体积 358MB → 279MB。

---

## 四、Feature Map 压缩（Section 3.4）—— BC Simulation 的数学推导

特征图本身虽然比原始多套 lightmap 小得多，但仍占空间。作者分两层进一步压：

### 4.1 对 \(F_{ut}^{2D}\) 和 \(F_{vt}^{2D}\)：8-bit 后训练量化 + 噪声模拟

训练中注入均匀噪声来模拟量化误差，让网络学会对此鲁棒：

\[
\begin{aligned}
V_{ut}' &= V_{ut} + \mathcal{U}(-0.5, 0.5) \cdot \alpha_{ut} \\
V_{vt}' &= V_{vt} + \mathcal{U}(-0.5, 0.5) \cdot \alpha_{vt}
\end{aligned}
\tag{6}
\]

其中 \(\alpha_{ut} = \alpha_{vt} = \frac{1}{256}\)，对应 8-bit 的 LSB 幅度。这等价于经典的 **straight-through quant-aware training** 思路：前向传播假装已经量化（加噪声近似梯度的不连续），反向传播梯度直接穿过。

### 4.2 对大块特征 \(F_{uvt}^{3D}\) 和 \(F_{uv}^{2D}\)：BC 仿真（核心创新）

BC（Block Compression，如 BC7）的本质是把图像切成 4×4 块，每块用两个端点 + 16 个插值权重表示：

\[
\begin{cases}
E = \{e_1, e_2 \in [0, 1]^k\} \quad \text{(端点，\(k\) = 通道数)} \\
W = \{w_1, \dots, w_{16}\}, \quad w_p \in [0, 1] \quad \text{(逐像素权重)}
\end{cases}
\tag{7}
\]

块内第 \(p\) 个像素的特征值重建为：

\[
f_p = (1 - w_p) e_1 + w_p e_2, \quad p = 1, \dots, 16 \tag{8}
\]

**关键洞察**：如果在普通 float 特征图上直接训完再 BC 压，会产生严重 mismatch——特征优化走的是连续空间，BC 强制的端点-权重网格会带来额外量化残差（Table 7：不加 BC sim 的 PSNR 只有 37.96 dB vs. 加了的 44.20 dB）。

**BC Simulation 的训练策略**：

1. **不直接优化** \(F_{uv}^{2D}\) 和 \(F_{uvt}^{3D}\) 的每个像素值，而是优化**端点 \(E\) 和权重 \(W\)**（即 BC 的底层表征参数）；
2. 每次前向传播，用式 (8) 重建出特征图；
3. 再从重建后的特征图上做双线性/三线性采样得到 \(V_{uv}, V_{uvt}\)；
4. 同样加量化噪声 \(V_{uvt}', V_{uv}'\)；
5. 喂入 MLP 算 loss（vs. 原始 lightmap 真值），梯度回传到 \(E\) 和 \(W\)。

收敛后，最终制品 = 用学到的 \(E, W\) 重建 → 8-bit 量化 → 标准 BC7 编码。

> 对 3D 特征 \(F_{uvt}^{3D}\)，做法是沿时间维度切片，每层 2D slice 独立做 BC7。

### 4.3 压缩效果的量级

| 阶段 | BPP | 相对原始 |
|------|-----|----------|
| 原始（未压） | 4.62 | 4.8% |
| + 8-bit 量化 | 2.32 | 2.4% |
| + 8-bit 量化 & BC7 | **0.68** | **0.7%** |

即最终只用到原始多套 lightmap 总数据量的 **不到 1%** 的码率。

---

## 五、训练目标函数

论文用的是简单的 L2 loss：

\[
\mathcal{L} = \sum_{(u,v,t) \in \mathcal{B}} \big\| G_\Phi(V_{uvt}, V_{uv}, V_{ut}, V_{vt}, \gamma(t)) - L_{\text{true}}(u, v, t) \big\|_2^2
\]

**预处理**：先对 lightmap 做 gamma correction（增强暗部细节），再对每个时间步做 per-channel mean normalization，渲染时还原。

**训练细节**：Adam, lr = \(10^{-3}\), batch = \(2^{12}\), 最后阶段 freeze 特征图、fine-tune MLP 在量化 + BC 仿真条件下。

---

## 六、运行时解码（Section 3.5）—— Virtual Texturing 集成

这是让"神经表示"真正能实时跑的关键工程桥梁：

### 逻辑链路

> 每帧可见区域 → VT 反馈 buffer → 哪些 virtual tile 需要 → 对需要的 tile 调 NDGI decoder (compute shader) → 写进 physical texture atlas → shading 时通过 page table indirection 正常 sample

### 为什么 VT 天然契合 NDGI

| 传统做法的痛点 | VT 怎么解决 |
|---------------|------------|
| 每帧对所有像素调 MLP = 算力爆炸 | 只解码当前可见 tile，且 tile 一旦缓存就跨帧复用 |
| lightmap 全量驻留显存 | physical texture 容量有界，超出部分 evict |
| 神经解码不是硬件 sampler | 解码结果烘焙回一张标准 texture，shading 走正常纹理管线 |

tile 边缘用 mirror padding（训练时也对应 pad），保证无缝拼接。

> 实测：解一个 1024×1024 lightmap，NDGI Medium 仅 **0.203 ms** on RTX 4060——约 **6× 更快** than NTC 的同量级配置。

---

## 七、整体数学图景（把所有公式串起来）

```
原始问题：存 {L₀, L₁, ..., Lₙ₋₁} → 空间 O(n·H·W·C)  ❌ 太大

NDGI 替换：
I(u,v,t) ≈ GΦ( concat(
    trilin(F₃D, u,v,t),       ← 时空体素特征
    bilin(F₂D_uv, u,v),       ← 静态空间特征
    bilin(F₂D_ut, u,t),       ← u-t 面特征
    bilin(F₂D_vt, v,t),       ← v-t 面特征
    γ(t)                      ← 时间傅里叶编码
) )

参数化为 Θ = {E/W of F₃D, E/W of F₂D_uv, F₂D_ut(int8), F₂D_vt(int8), Φ}
BC7 编码 E/W → 最终码率 ~0.5–1.4 BPP

运行时：VT 按需选 tile → compute shader 跑 GΦ → 写 physical tex → 正常采样
```

---

## 八、一句话评价这个设计的数学美感

它本质上是把一个 **3D 信号（2D 空间 + 1D 时间）→ RGB** 的压缩问题，从"暴力存全部采样点"重构为：

> **低秩/低维投影（tri-plane 分解）+ 体素化的稀疏高维残差（\(F_{uvt}^{3D}\)）+ 可微的硬件对齐量化约束（BC simulation）+ 渲染管线的缓存论（VT）= 在随机访问约束下逼近信息极限**
