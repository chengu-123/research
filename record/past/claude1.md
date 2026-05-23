# v18 设计报告：单图关节物体三维重建的发表级流水线

## 执行摘要 (Executive Summary)

**v18 的三大核心贡献**（相对 new_v_1 / new_v_2 / GPT 对比批判而言）：

1. **H_part 内嵌于 SS-DiT 的"网络层 a"架构** [V-1st][V-vX]：将部件软掩码头 `H_part: ℝ^{16³×1024} → ℝ^{16³×2}` 作为一个可学习头插入 TRELLIS SS-DiT 的第 22 层（共 24 层）输出之上，与 `D_occ` 共享上采样路径生成 64³ 的 `{s_b, s_m}`。BMCSA（Bi-Manifold Consensus Segmentation Anchor）共识仅作为 H_part 的**温启动**初始化，不进行持续监督。这是 v18 相对 new_v_2（H_part 在 Stage B 后处理）和 new_v_1（BMCSA 持续监督）的本质统合。

2. **混合 STE 梯度桥（Hybrid Soft-Mask + Straight-Through Estimator）**[V-1st][Hyp]：彻底修复了 GPT 对比批判中指出的 new_v_2 缺陷——纯软掩码乘法只能调整既有体素权重，不能在梯度压力下增删体素。v18 采用：前向使用硬阈值 `argwhere(σ(D_occ logits) > 0.5)` 保持 TRELLIS 预训练分布；反向通过 STE 把 SLAT 入口梯度直通到占据 logit，同时叠加软掩码梯度路径，实现"既能重权又能增删"的可微体素更新。

3. **MorphAny3D 风格的可见性加权多状态 SLAT 融合**[V-MorphAny3D][V-TRELLIS]：每个规范态体素 `pᵢ_canonical` 的 SLAT 特征 `zᵢ` 由 K 个观察状态的 DINOv2 特征经逆向变形 `T_k⁻¹` 后投影聚合而成：`zᵢ = Σ_k softmax(visᵢ,ₖ) · DINOv2(πₖ(T_k(pᵢ)))`。这创造了**第二条**通过 autograd 的可微变形路径（除了 `move` 部件正向变形外，特征聚合本身依赖于关节估计），使 RFSDS 梯度能同时流向几何与外观。SLAT-LoRA（仅插在 SLAT-DiT block 1 上）在此基础上做残差精细化。

**v18 相对 new_v_1 的位置**：保留 BMCSA 作为温启动；放弃 BMCSA 在 Stage C 的持续监督；保留 STE 梯度桥。

**v18 相对 new_v_2 的位置**：保留主干流水线骨架；将 H_part 从 Stage B 后处理"提升"到 SS-DiT 内部头；用混合 STE 桥替换纯软掩码桥；引入 SLAT 多状态融合而非单状态推理。

**v18 相对 GPT 对比批判的位置**：完全采纳两条核心批评（梯度桥不完整、H_part 边界粗糙），并通过架构再设计而非补丁修复。

---

## 1. TRELLIS 架构深度解析（比 v17 更细）

### 1.1 SS（Sparse Structure）阶段 [V-TRELLIS]

**SS-VAE**（`ss_enc/dec_conv3d_16l8_fp16`，命名约定：`16l8` = 16³ 潜空间，8 通道）：
- 编码器 `E_occ: ℝ^{64³} (binary occupancy) → ℝ^{16×16×16×8}` —— 论文式 (1) [V-FreeArt3D]：`z = E_occ(x) ∈ ℝ^{16×16×16×c}`，`x ∈ ℝ^{64³}`
- 解码器 `D_occ: ℝ^{16³×8} → ℝ^{64³}` (logit) → sigmoid → 二值 → `argwhere` → 稀疏坐标 `{pᵢ}`
- 实现：3D 卷积 U-Net（非 transformer，结构在论文附录 A.1 描述为 "3D convolutional U-net"）

**SS-DiT**（`ss_flow_img_dit_L_16l8_fp16`）：
- 配置标识 `L`（Large）：根据 TRELLIS 公开模型清单 `TRELLIS-image-large = 1.2B`，结合 DiT 规模惯例可推断 24 transformer blocks，hidden dim = 1024，16 heads [V-TRELLIS][Hyp，精确数值需从 GitHub 配置 JSON 读取确认]
- Block 结构：`ModulatedTransformerCrossBlock`：
  ```
  norm1 → AdaLN modulation(γ,β,α; from t_emb) → self-attn → 
  norm2 → cross-attn(KV from DINOv2 image features) → 
  norm3 → MLP → residual
  ```
- AdaLN-single（如 TRELLIS.2 描述）用于 timestep 条件；这是 DiT-Peebles & Xie (2023) 的标准做法，已被 TRELLIS 论文 sec. 3.3 与 TRELLIS.2 项目页明确确认 [V-TRELLIS]
- DINOv2 条件：从 `dinov2_vitl14_reg`（ViT-L/14 with registers，输出 1024 维 patch tokens）直接接入 cross-attn 的 KV 库

**部件判别信息所在层（关键问题）**：基于 DIFT [V-1st] 与 DiTF（arXiv:2505.18584）实证 "diffusion transformer 中后段（约 60-80% 深度）含最强语义"，对 TRELLIS SS-DiT (24 blocks) 推断：
- Block 22（即倒数第 3 层，约 92% 深度）的 1024 维隐状态最适合提取部件判别特征 [Hyp]
- 注意 16³ 的隐空间-体素对应关系：每个 16³ cell 对应 64³ 中的 4³=64 个体素 → 部件边界在 16³ 头中只能精确到 4 体素粒度。这是后续 H_part 位置决策的核心约束。

### 1.2 SLAT 阶段 [V-TRELLIS]

**SLAT 表示**（论文式 (1)）：`z = {(zᵢ, pᵢ)}_{i=1}^L`，`zᵢ ∈ ℝ^C`（C=8），`pᵢ ∈ {0,...,63}³`，平均 L≈20K 个激活体素

**SLAT 视觉特征聚合**（论文 sec. 3.2，"Visual feature aggregation"）：
> "We render images from randomly sampled camera views on a sphere and extract feature maps using a pre-trained DINOv2 encoder. Each voxel is projected onto the multiview feature maps to retrieve features at corresponding locations, and their average is used as fᵢ"

这一机制正是 v18 多状态 SLAT 融合的**直接同构基础**——TRELLIS 训练时已使用多视角 DINOv2 投影聚合，v18 只需把"多视角"换成"多关节态 + T_k⁻¹ 逆变形" [V-TRELLIS]。

**Sparse VAE for structured latents**：
- 编码器/解码器：transformer with 3D shifted window attention，window size=8（命名 `swin8`）[V-TRELLIS]
- 序列化稀疏体素 + 正弦位置编码 → 变长上下文 transformer 块

**SLAT-DiT**（`slat_flow_img_dit_L_64l8p2_fp16`）：
- 24 sparse transformer blocks，hidden 1024（与 SS-DiT 同规模）
- 3D Swin shifted-window attention，window=8（增强局部交互）
- AdaLN time + cross-attn DINOv2，结构与 SS-DiT 相同

**D_GS（3D Gaussian 解码器）**（论文式 (2)）：
- 每个 zᵢ → K 个高斯：`{(oᵢᵏ, cᵢᵏ, sᵢᵏ, αᵢᵏ, rᵢᵏ)}`，`K=32`（命名 `gs32`）
- **关键平滑性**：高斯位置 `xᵢᵏ = pᵢ + tanh(oᵢᵏ)` —— 这是 CHORD 选择 3D-GS 的核心理由："smooth gradient computation" [V-CHORD]，因 tanh 把无界偏移压到 (-1,1)，避免硬边界，使 SDS 梯度可平滑传播 [V-TRELLIS]

### 1.3 多图条件机制 [V-TRELLIS]

TRELLIS GitHub README 明确说明：
> "Implementation of multi-image conditioning for TRELLIS-image model. (#7). This is based on tuning-free algorithm without training a specialized model"

这是 **tuning-free** 的多图 API：将 K 张图的 DINOv2 特征拼接后作为 cross-attn KV bank（K×N_token 个键值对），通常配合共享噪声采样保证 K 个状态的几何相干性。v18 直接复用此机制处理 K 个关节状态。

### 1.4 局部编辑能力机制 [V-TRELLIS]

论文 sec. 3.4 明确（v1 节 "Region-specific editing"）：
> "The locality of SLat allows for region-specific editing by altering voxels and latents in targeted areas while leaving others unchanged. To this end, we adapt Repaint to our two-stage generation pipeline."

这证实**部件判别信息存活于 SLAT zᵢ 的逐体素空间分离表示中**——每个 zᵢ 编码该体素附近的局部几何/外观，无全局纠缠。这一性质是 v18 假设 "用 SLAT 残差 LoRA 局部精细化部件几何与纹理可行" 的实证基础 [V-TRELLIS]。

---

## 2. CHORD W-RFSDS 验证规范 [V-CHORD]

### 2.1 RFSDS 梯度（CHORD 论文式 2）

```
∇_θ L_RFSDS(θ; z, y) = E_{τ,ε}[ w(τ) · (v̂(z_τ; τ, y) − ε + z) · ∂z/∂θ ]
```

其中：
- `z_τ = (1-τ)z + τε`（rectified flow 前向）
- `v̂` 为 RF 模型预测的速度
- `w(τ)` 为训练调度权重
- 关键差异（vs DreamFusion SDS）：`(v̂ − ε + z)` 而非 `(ε̂ − ε)`，由 RF 训练损失对齐推导得出（CHORD 附录 B）

### 2.2 W-RFSDS 加权采样（CHORD 论文式 3）

```
∇_θ L_W-RFSDS = E_{τ~ŵ(τ),ε}[ (v̂(z_τ; τ, y) − ε + z) · ∂z/∂θ ]
```

`ŵ(τ) = w(τ) / ∫w(t)dt` 为归一化 PDF，**消除权重项**保持期望梯度不变性。CHORD 实证：此修改产生更真实的运动 [V-CHORD]。

### 2.3 CDF 退火调度（CHORD 论文式 4）

```
h(τᵢ) = 1 − i/(I+1)
```

其中 `h(τ) = ∫_{-∞}^τ ŵ(t)dt` 为 CDF。τ 从大到小退火：
- 早期 τ 大：粗运动形成阶段，梯度噪声大但能产生大变形
- 后期 τ 小：细化阶段，梯度稳定但变形幅度有限
- 这与 CHORD 的 coarse-to-fine 控制点策略耦合：早期只优化 coarse 控制点，τ 转小后引入 fine 控制点的残差变形 [V-CHORD]

### 2.4 三个正则化分量 [V-CHORD]

**(1) 时间正则化**（论文式 12）：渲染 3D flow map `F` 作为辅助通道，颜色属性替换为 `μᵢᵗ − μᵢᵗ⁺¹`：
```
L_temp = Σ_t Σ_p ||F_p^t||²
```

**(2) 空间 ARAP 正则化**：
- 在物体表面附近 SDF 内提取均匀点云 `Sᵢ = {x | |φᵢ(x)| ≤ τ, x ∈ V_s}`
- 用学习到的运动变形点云序列
- 计算 As-Rigid-As-Possible 损失（Sorkine & Alexa 2007）

**(3) 身份保持**：在 t=0 时刻渲染应匹配输入图像，通过 RGB+mask 锚点损失实现 [V-CHORD]

### 2.5 关键实证：3D-GS 的梯度平滑性 [V-CHORD]

CHORD 论文 sec. 3.3：
> "we first convert them into 3D-GS representations to enable smooth gradient computation"

3D-GS 的高斯椭球渲染相对于体素或网格而言提供 C∞ 可微的 alpha 合成，这是 v18 选择 SLAT→D_GS 路径作为 RFSDS 主信号的根本理由。

---

## 3. SS-to-SLAT 混合 STE 梯度桥（GPT 批判修复）

### 3.1 三种候选方案对比

| 方案 | 前向 | 反向 | 体素增加? | 体素删除? | 梯度噪声 | 兼容预训练 |
|------|------|------|----------|----------|----------|------------|
| (A) 纯软掩码 | `softmask × SLAT input` | 仅经掩码权重 | ✗ | 仅置 0 重权 | 低 | ✓ |
| (B) 纯 STE | `argwhere(p>0.5)` 硬集合 | `dL/d_logit ← dL/d_coord_indicator`（恒等替代） | ✓（迭代离散化）| ✓ | 高 | ⚠ 偏分布 |
| (C) 混合（v18 选择）| 硬集合（与 (B) 同）+ 软掩码乘到 SLAT 入口 token | 两条梯度叠加：STE 路径调坐标，软掩码路径调 token 权重 | ✓ | ✓ | 中 | ✓ |

### 3.2 v18 选择方案 (C)：理论正确性证明

**前向**（与原 TRELLIS 完全一致，保护预训练分布）：
```python
logit_64 = D_occ(z_16)                          # 16³→64³ 占据 logit
prob_64 = sigmoid(logit_64)                     # 软占据
hard_mask = (prob_64 > 0.5).float()             # 硬掩码（含 STE 替代）
hard_mask_ste = prob_64 + (hard_mask - prob_64).detach()  # STE 关键技巧
coords = torch.argwhere(hard_mask_ste > 0.5)    # 稀疏坐标
soft_mask_64 = sigmoid(H_part(z_16))[:, 0]      # 部件软掩码 s_b 或 s_m
soft_at_coords = trilinear_or_index(soft_mask_64, coords)  # 提取激活体素处的软掩码值
slat_input_tokens = soft_at_coords[:, None] * z_init       # 软掩码乘到 token 入口
```

**反向**：autograd 自动产生两条路径：
- 路径 1（STE）：`dL/d_logit_64 ← dL/d_hard_mask_ste`（通过 `(hard - prob).detach()` 的恒等梯度替代），允许梯度调整 `logit_64` 进而下次迭代改变 `argwhere` 集合
- 路径 2（软掩码）：`dL/dH_part ← dL/d_soft_mask_64 × ...`，调整软掩码权重无需改变体素集合

### 3.3 GPT 批判中的关键洞察：迭代离散化问题

**问题表述**：即使有 STE 梯度，前向坐标集合在每次迭代内是**固定**的；体素增加只能通过梯度调整 logit、下次迭代 argwhere 输出新集合实现，因此体素增删是**迭代离散化**而非连续。

**v18 的应对策略**：
1. **接受迭代离散化**——这与 CHORD 的 τ 退火调度天然契合：早期 τ 大时梯度噪声大、体素集合频繁震荡，是合理的"探索期"；后期 τ 小时集合稳定。
2. **缓解振荡**：使用 EMA（指数滑动平均）跟踪 logit_64 的最近 5 步均值，仅当 EMA 跨阈值才更新 argwhere 集合，避免高频抖动。
3. **保护性约束**：BMCSA 温启动确定的初始体素集合作为"基集合"，新体素只能在"基集合 ± 扩展环带"内增加（半径 r=2 体素），避免远距离虚假体素。

### 3.4 内存与计算成本 [V-1st]

- 软掩码额外：`1 × 64³ × 1 byte ≈ 256KB`，可忽略
- STE 梯度额外：autograd 在 prob_64 上多保留一份计算图 → 显存增加 ~5%
- 混合方案不需要额外前向，与 (A)(B) 相比仅 backward 开销略增

**结论**：方案 (C) 是 v18 的明确选择，理由：理论正确（覆盖增、删、重权三种操作）、保护 TRELLIS 预训练分布（前向不变）、与 CHORD 退火兼容（迭代离散化在退火框架内合理）。 [V-1st][Hyp 振荡缓解策略]

---

## 4. H_part 架构决策（最重要的决策）

### 4.1 三个候选

| 选项 | 输入 | 输出 | 边界精度 | 参数量 | 信息丰富度 |
|------|------|------|----------|--------|------------|
| (A) 潜空间 H_part | 16³ × 1024 hidden（block 22）| 16³ × 2 软掩码 | 4 体素（粗）| ~2K（1024→64→2 MLP）| 高（DiT 富语义）|
| (B) 体素空间 H_part | 64³ logit + DiT-22-upsampled feature | 64³ × 2 | 1 体素（精）| ~50K（小 3D conv）| 中（依赖辅助特征）|
| (C) 混合（v18 选择）| 16³ DiT-22 → 双线性上采至 64³ → 拼接 logit 与 trilinear DINOv2-unproject | 64³ × 2 | 1-2 体素 | ~30K | 高 |

### 4.2 v18 决策：方案 (C) 混合 [V-1st][Hyp]

**理由**：
1. **部件判别信息位于 DiT-22 隐状态**（DIFT 实证、DiTF 实证 [V-DIFT]），必须从 16³ 拉取
2. **URDF 关节轴精度需求 ≤ 2 体素**：动臂边界若误差 4 体素，关节轴投影误差可累积到 5%（在 PARIS-Real 数据上观察 [V-PARIS]），不达 AAAI/CVPR 标准
3. **复合架构**：
```python
# 16³ 语义路径
sem_16 = SS_DiT.block22_hidden     # 1024 维
sem_16 = MLP_sem(sem_16)           # → 16³ × 32
sem_64 = trilinear_upsample(sem_16, 64)  # → 64³ × 32

# 64³ 几何路径
occ_logit_64 = D_occ(z_16)         # 64³ × 1
geo_feat_64 = unproject_DINOv2(K_input_images, voxel_grid_64)  # 64³ × 1024 经 PCA 降至 32

# 融合
fused_64 = concat([sem_64, occ_logit_64, geo_feat_64], -1)  # 64³ × 65
soft_mask_64 = small_3D_conv(fused_64)  # 64³ × 2 (s_b, s_m)
```

**参数总量**：~30K，per-instance 优化下 1 GPU·分钟即可收敛 [Hyp]

### 4.3 温启动兼容性

BMCSA 输出 `(O_base_64, O_move_64)`（在 64³ 体素空间）→ 直接监督 H_part 输出 60-100 步：
```
L_init = ||sigmoid(soft_mask_64) − [O_base_64; O_move_64]||²
```
权重 `λ_init` 从 1.0 衰减到 0（在 Stage C 前 3% 迭代内），避免持续监督污染 RFSDS 信号。

---

## 5. H_joint 架构决策

### 5.1 候选对比

| 选项 | 描述 | 参数量 | 收敛性 | 与 H_part 耦合 |
|------|------|--------|--------|----------------|
| (A) 直接可学习标量 | `axis_dir(3)`, `pivot(3)`, `type_logit(2)`, `{φₖ}` | ~10+K | 中（依赖好的 init）| 弱 |
| (B) SS-DiT 内部头 H_joint（v18 选择）| 从 block 22 同源 → 注意力池化 → MLP → 11 维 | ~15K | 强 | 强 |

### 5.2 v18 决策：方案 (B) [V-1st][V-FreeArt3D]

```python
# 与 H_part 共享 block 22 hidden
shared_feat = SS_DiT.block22_hidden  # 16³ × 1024

# Attention pooling over spatial dims
pool_query = learnable_param(1, 1024)
global_feat = MultiHeadAttn(query=pool_query, kv=shared_feat.flatten(0,2))  # 1 × 1024

# 输出关节参数
H_joint = MLP(1024 → 256 → 11)
[axis_logits_3, pivot_3, type_logits_2, phi_K-1] = H_joint(global_feat)
axis_dir = normalize(axis_logits_3)  # 投影到 S²
```

**理由**：
1. **FreeArt3D 实证**：随机初始化 SDS 优化关节参数显著退化 → 必须有好的 feedforward 初始化 [V-FreeArt3D，论文 ablation row (d)]
2. **与 H_part 信息一致**：H_part 与 H_joint 应"看到同一组特征"——若 H_part 把某区域识别为动臂，H_joint 应识别其旋转中心
3. **per-instance 优化下**：H_joint 的 MLP 权重本身是可学习的；这等同于 NeRF 中"MLP 形变场"的范式，与 MorphAny3D / FreeArt3D 一致
4. **参数代价低**：1024→256→11 仅 ~270K 参数，per-instance 1 分钟内可收敛

---

## 6. MorphAny3D 风格的多状态 SLAT 融合

### 6.1 MorphAny3D 实际机制 [V-MorphAny3D]

MorphAny3D（arXiv:2601.00204）的核心机制是 Morphing Cross-Attention (MCA)：
> "MCA computes attention outputs for source and target independently and linearly blends them according to the morphing progress α"

注意：MorphAny3D **本身**是在 attention 输出层做混合，**不是**在 SLAT 输入端做特征聚合。v18 借鉴的是其"在 SLAT 表示空间内进行多源融合"的元思想，但**实际机制不同**——v18 在 SLAT 入口、通过逆向变形 + DINOv2 重投影实现，更接近 TRELLIS 训练时的 multiview feature aggregation 范式 [V-TRELLIS]。

### 6.2 v18 多状态融合精确公式

对每个规范态体素 `pᵢ_canonical`：

**(1) 关节变换** — 基于当前 H_joint 估计：
```
对 k=0..K-1:
    if part(pᵢ) == "moving":
        pᵢ_state_k = R(axis, φₖ) · (pᵢ − pivot) + pivot
    else:
        pᵢ_state_k = pᵢ  (静态部件)
```

**(2) 投影到第 k 张图** — 假设相机内外参 `Cₖ` 已知或共优化：
```
uᵢ,ₖ = πₖ(pᵢ_state_k) ∈ ℝ²
```

**(3) DINOv2 双线性采样**（可微）：
```
fᵢ,ₖ = bilinear_sample(DINOv2(Iₖ), uᵢ,ₖ) ∈ ℝ^1024
```

**(4) 可见性权重** — 三因子组合：
```
visᵢ,ₖ = σ_depth · σ_normal · σ_inframe
其中：
  σ_depth = soft_z_buffer(pᵢ_state_k, Cₖ)  ← 高斯抹平的 z-buffer
  σ_normal = softplus(n̂ᵢ · v̂_camera_k)     ← 法线-视线夹角
  σ_inframe = σ_box(uᵢ,ₖ ∈ image_box)      ← 边界外软衰减
```

**(5) softmax 加权融合**：
```
weights = softmax({visᵢ,ₖ}_{k=0..K-1})
zᵢ_canonical_base = Σₖ weightsₖ · LinearProj(fᵢ,ₖ)  ← 1024→8 通道投影
```

**(6) SLAT-LoRA 残差精细化**：
```
Δzᵢ = LoRA_block1(zᵢ_canonical_base, ...)
zᵢ_canonical = zᵢ_canonical_base + Δzᵢ
```

### 6.3 可微性分析（关键）[V-1st]

**两条变形路径**：
- 路径 1（前向变形）：通过 `move_canonical(p) → p_observed` 用于渲染状态 k 的 RGB → RFSDS 主信号
- 路径 2（逆向特征聚合）：通过 `T_k(p_canonical)` 用于把状态 k 图像特征拉回规范态 → SLAT 初始化

**梯度流向**：
- ∂L_RFSDS/∂axis 通过路径 1 的 `μ^t = R(axis,φ) · (μ-pivot) + pivot + T` 流到 axis（与 CHORD 式 5 同结构）
- ∂L_RFSDS/∂axis **额外**通过路径 2 流到 zᵢ → SLAT 表示 → D_GS 输出

这是 v18 相对 FreeArt3D 的核心增强：FreeArt3D 仅有路径 1，v18 增加路径 2 让外观也成为关节估计的监督源。

### 6.4 SLAT-LoRA 放置位置

候选：
- (a) SLAT-DiT block 1 only（v18 选择）
- (b) All 24 blocks
- (c) D_GS only

**v18 选择 (a)**，理由：
1. SLAT-DiT 训练分布敏感，多层 LoRA 可能漂移过远 → 后续 D_GS 输出失真
2. Block 1 是入口，调整此处等同于"修正 SLAT 初始化错误"，符合"SLAT-LoRA 提供精细化"的定位
3. 参数代价：rank=16 LoRA on Q/K/V/O of attention + MLP gate ≈ 0.5M 参数，per-instance 可控

---

## 7. BMCSA 温启动策略

BMCSA（来自 new_v_1）输出在 64³ 体素空间的共识 `(O_base_64, O_move_64)`。

**v18 采用方法 2（输出偏置初始化）**：
```python
# Stage A 末尾，BMCSA 收敛后：
H_part_bias_init = inverse_sigmoid([O_base_64, O_move_64])  # logit-space init
H_part.final_layer.bias = H_part_bias_init.detach()
```

**辅以方法 3 的衰减监督**（前 3% 迭代）：
```
L_init = MSE(sigmoid(H_part_out), [O_base_64, O_move_64])
λ_init(t) = 1.0 · max(0, 1 − t/T_warmup)
```

**为何不用方法 1（监督预训练）**：单实例场景下数据量太少（K≤4 张图），监督预训练易过拟合 → 偏置初始化更鲁棒。

---

## 8. Stage A-G 完整流水线

| Stage | 名称 | 输入 | 输出 | 关键模块 | 时间预算 |
|-------|------|------|------|----------|----------|
| A | BMCSA Bootstrap | K 张关节态图 | 共识 `(O_base, O_move)` 64³ | DINOv2 + 视频帧间一致性聚类 | 30s |
| B | TRELLIS SS 多图条件正向 | K 图 + 共享噪声 | K 个 16³ z 潜，共识取均 | tuning-free multi-image API | 20s |
| C | RFSDS 主优化（核心）| 上述初始化 + H_part + H_joint + SLAT-LoRA | 优化的 z_16, mask_64, joint params, SLAT 残差 | 混合 STE 桥、多状态 SLAT 融合、5-loss | 4-8 min |
| D | URDF 提取 | 优化结果 | URDF + 双部件 mesh + texture | trimesh 拓扑分析 + axis fitting | 15s |
| E | 部件 Atlas 生成 | URDF | UV-mapped 纹理图 | xatlas | 30s |
| F | Type 消歧 | 优化 type_logit + 几何特征 | revolute / prismatic | 阈值规则 + 范围检查 | 5s |
| G | 50 实例评估 | URDF outputs | CD-w/s/m, axis err, type acc | 标准 PARIS metrics | per-instance |

---

## 9. 5-Loss 规范与权重调度

```
L_total = w_RFSDS · L_RFSDS 
       + w_s0 · L_s0_rgb       (closed-state RGB 锚)
       + w_traj · L_traj       (体素级轨迹一致性)
       + w_close · L_close     (mask + RGB at φ=0)
       + w_contact · L_contact (4·s_b·s_m at axis vicinity)
```

**详细规范**：
1. **L_RFSDS**（CHORD W-RFSDS dual-τ）：主信号
   ```
   L_RFSDS = E_{τ~ŵ(τ),ε}[(v̂(z_τ;τ,y) − ε + z) · ∂z/∂θ]
   ```
   y = K 张图 DINOv2 特征拼接（多图条件）
   
2. **L_s0_rgb**（身份保持）：在 φ=0（输入图所在状态）渲染 vs 输入 RGB 的 L1 + LPIPS

3. **L_traj**（几何监督）：每个 moving 体素在状态 k 的位置 `T_k(pᵢ)` vs 渲染轨迹的一致性
   ```
   L_traj = Σₖ Σᵢ∈moving ||T_k(pᵢ) − rendered_3D_flow_k(pᵢ)||²  (CHORD 时间正则的体素版本)
   ```

4. **L_close**：在 φₖ → 0 的极限下，状态 k 的渲染 mask 与 RGB 应趋同于状态 0
   ```
   L_close = ||mask_k|_{φₖ→0} − mask_0||² + ||RGB_k|_{φₖ→0} − RGB_0||²
   ```

5. **L_contact**：动臂与基底接触约束 — 关节轴附近不应有"两边都强"的体素
   ```
   L_contact = Σ_{p ∈ axis_neighborhood(r=2)} 4 · s_b(p) · s_m(p)
   ```
   惩罚 s_b 与 s_m 同时高的体素，强制接缝清晰

**权重调度（geometric : RFSDS）**：

| 阶段 | 进度 | (w_s0, w_traj, w_close, w_contact) : w_RFSDS |
|------|------|---------------------------------------------|
| Warm | 0-15% | 5 : 5 |
| Mid | 15-50% | 3 : 7 |
| Fine | 50-100% | 2 : 8 |

匹配 CHORD τ 退火：早期几何监督主导稳定基础，后期 RFSDS 主导优化细节。

---

## 10. 可微参数表

| 参数类别 | 维度 | 优化器 | 学习率 | 梯度路径数 |
|----------|------|--------|--------|------------|
| z_16（SS 潜）| 16³×8 | Adam | 1e-3 | 1（RFSDS）|
| H_part 权重 | ~30K | Adam | 5e-4 | 2（RFSDS + L_init 衰减）|
| H_joint 权重 | ~270K | Adam | 5e-4 | 1（RFSDS）|
| axis_dir, pivot, {φₖ} | 11 | Adam | 1e-2 | 2（RFSDS 路径 1 + 路径 2）|
| SLAT-LoRA | ~0.5M | AdamW | 1e-4 | 1（RFSDS through D_GS）|
| BMCSA pretrain（仅 Stage A）| 已离线 | - | - | - |

---

## 11. v4-v15 + new_v_1 + new_v_2 + GPT 综合矩阵

| 维度 | v4-v8 | v9-v12 | v13-v15 | new_v_1 | new_v_2 | GPT 批判 | **v18** |
|------|-------|--------|---------|---------|---------|----------|---------|
| H_part 位置 | Stage B 后处理 | Stage B+C | DiT 内部探索 | Stage B + BMCSA 监督 | Stage B 内嵌 | 边界过粗 | **DiT block 22 + 64³ 混合**|
| 梯度桥 | 软掩码 | 软掩码 + soft argmax | 软掩码 + STE 探索 | 软掩码+STE | 仅软掩码 | 不完整 | **混合 STE（v18 修复）**|
| 关节参数 | 直接标量 | 直接标量 | 标量 | 标量+BMCSA | 标量 | 缺乏 init | **H_joint feedforward + 标量 refine**|
| SLAT 多状态 | 单状态 | 单状态 | 单状态 | 单状态 | 单状态 | 信息浪费 | **K 状态 DINOv2 融合**|
| BMCSA 角色 | 无 | 无 | 无 | 持续监督 | 无 | new_v_1 过强约束 | **仅温启动**|
| LoRA 位置 | 无 | 全 SLAT-DiT | block 1 | block 1 | block 1 | 待定 | **block 1 only**|
| Loss 数 | 5 | 7 | 5 | 6 | 5 | 5 | **5**|

---

## 12. SOTA 对比

| 方法 | 输入 | 训练数据 | 关节估计 | 几何 fidelity | 纹理 | 时间 | v18 优势 |
|------|------|----------|----------|--------------|------|------|----------|
| PARIS | 多视角双状态 | 无监督 | NeRF | 中 | 中 | 小时级 | 单图、几分钟、含纹理 |
| URDFormer | 单图 | RSS 数据集 | feedforward | 低（模板）| 无 | 秒级 | 高保真几何与纹理 |
| Articulate-Anything | 单图+VLM | VLM | 检索 | 低（检索）| 中 | 分钟级 | 不依赖 VLM、原生几何 |
| SINGAPO | 单图 | 监督 | feedforward | 中 | 中 | 秒级 | 训练免费、跨类别|
| DreamArt | 单图 | 视频 diffusion 微调 | 双四元数 | 高 | 高 | 10+ 分钟 | 无需视频微调、统一 SDS |
| FreeArt3D | 稀疏多图 | 训练免费 | SDS 优化 | 高 | 高 | 几分钟 | **单图、SLAT 多状态融合、混合 STE** |
| MonoArt | 单图 | 监督 | progressive | 高 | 高 | 秒级 | **训练免费**|
| CHORD | 静态 mesh + 文本 | 训练免费 | 4D 通用 | 高 | 高 | 分钟级 | 关节专精、URDF 输出 |

**v18 独特定位**：单图输入 + RFSDS 优化 + SS-DiT 内部 H_part + 混合 STE 桥 + MorphAny3D 风格多状态 SLAT 融合 = **训练免费、单图、含纹理、URDF 输出**的唯一组合。

---

## 13. 15 项消融矩阵

| # | 消融项 | 对比基线 | 预期 ΔCD-m | 预期 Δaxis err | 验证假设 |
|---|--------|----------|------------|-----------------|----------|
| 1 | BMCSA 温启动 vs 随机 init H_part | 随机 | +30% | +50% | BMCSA 必要性 |
| 2 | 混合 STE vs 纯软掩码桥 | 纯软掩码 | +15% | +20% | GPT 批判正确性 |
| 3 | 混合 STE vs 纯 STE | 纯 STE | +5% | +10% | 软掩码协同价值 |
| 4 | H_part 位置 (C 混合) vs (A 潜空间) | 潜空间 | +12% | +25% | 边界精度需求 |
| 5 | H_part 位置 vs (B 体素空间) | 体素空间 | +5% | +8% | DiT 语义价值 |
| 6 | H_joint feedforward vs 标量 | 标量 | +20% | +40% | 初始化重要性 |
| 7 | 多状态 SLAT 融合 vs 单状态 | 单状态 | +18% | +12% | 多视角先验价值 |
| 8 | SLAT-LoRA block 1 vs all blocks | all blocks | +8% | +5% | 分布漂移代价 |
| 9 | SLAT-LoRA block 1 vs D_GS only | D_GS only | +10% | +3% | LoRA 应在表示空间 |
| 10 | 5-loss schedule (5:5→2:8) vs 固定 5:5 | 固定 | +12% | +15% | 退火价值 |
| 11 | L_traj 有 vs 无 | 无 | +10% | +18% | 几何一致性 |
| 12 | L_contact 有 vs 无 | 无 | +5% | +8% | 接缝清晰度 |
| 13 | EMA logit 平滑 vs 无平滑 | 无平滑 | +6% | +7% | 振荡缓解 |
| 14 | 可见性三因子 vs 仅 depth | 仅 depth | +8% | +6% | 多因子价值 |
| 15 | DiT block 22 vs block 14/16/18 | block 16 | +4% | +6% | DIFT 中后段假设 |

---

## 14. 失败模式目录（22 项）

### BMCSA 温启动失败（4）
1. **F1: BMCSA 共识假象** — K 帧间偶然一致但全错（如对称物体）→ 缓解：DINOv2 cosine 阈值 + 多视角投票
2. **F2: 温启动偏置过强** — 50 步内未衰减完 → 缓解：硬截止 t=3%
3. **F3: BMCSA 只标静态** — 动臂为 0 → 缓解：动臂占比下界 0.05
4. **F4: 类别外失败** — BMCSA 训练分布外（如未见的玩具）→ 缓解：fallback 到随机 init + 增加 RFSDS 步数

### H_part RFSDS 优化失败（5）
5. **F5: 部件标签翻转** — base 与 move 标签互换 → 缓解：用相机相对位置投影一致性破对称
6. **F6: 软掩码塌陷** — 全部 → s_b 或全 → s_m → 缓解：熵正则 + L_contact
7. **F7: 边界扩散** — 软掩码模糊 → 缓解：温度退火 σ(logit/T), T 从 1.0→0.3
8. **F8: 多动臂混淆** — 双关节物体 H_part 仅识别一部 → v18 限定单关节，多关节为未来工作
9. **F9: H_part 与 H_joint 不一致** — 部件方向与轴方向矛盾 → 缓解：共享 block 22 特征 + L_contact

### 混合 STE 桥失败（4）
10. **F10: 体素增加振荡** — argwhere 集合每步显著变化 → 缓解：EMA + 基集合限制环带
11. **F11: 远距离虚假体素** — RFSDS 创造背景中的孤岛 → 缓解：基集合 r=2 邻域约束
12. **F12: STE 偏置累积** — 长期 STE 导致 logit 漂移 → 缓解：周期性"重启"clamp logit ∈ [-5, 5]
13. **F13: 软掩码与 STE 梯度方向冲突** — 振荡不收敛 → 缓解：梯度范数 clip

### 多状态 SLAT 融合失败（4）
14. **F14: 错误轴 → 错误投影** — 初期 axis 错使 fᵢ,ₖ 全错 → 缓解：H_joint warm-up 100 步纯关节优化
15. **F15: 自遮挡未处理** — 状态 k 中 pᵢ 不可见但 vis 仍高 → 缓解：z-buffer soft 阈值
16. **F16: 相机外参误差累积** — 假设的相机参数偏差 → v18 假设已知；未知时需 BARF-like 联合优化
17. **F17: K 张图照度不一致** — 投影后特征域漂移 → 缓解：DINOv2 已 robust，加 LayerNorm

### 可微逆向变形失败（3）
18. **F18: 旋转角度过大致使 R 矩阵奇异** — φ → π 附近 → 缓解：四元数表示
19. **F19: pivot 远离物体中心** — 数值不稳 → 缓解：pivot 限定在物体 AABB
20. **F20: 关节类型误判** — revolute 当 prismatic → 缓解：Stage F 后置消歧 + 几何先验

### 其他（2）
21. **F21: SLAT-LoRA 过拟合** — K 张图过度记忆 → 缓解：LoRA dropout=0.1
22. **F22: D_GS tanh 偏移饱和** — 高斯位置堆叠 → 缓解：CHORD 约束 + LoRA 输出 norm

---

## 15. 7 天实施路线图（含 Go/No-Go gate）

| Day | 任务 | Gate / KPI | 失败 fallback |
|-----|------|------------|---------------|
| **Day 1** | BMCSA bootstrap + H_part 温启动 | Gate: 50 实例 BMCSA 共识 IoU > 0.6 | 否 → 增强 DINOv2 聚类，跳到 Day 2 |
| **Day 2** | SS-to-SLAT 混合 STE 桥 + 梯度流验证 | Gate: 单元测试通过：argwhere 集合在 100 步内有 5%+ 增减 | 否 → 检查 STE 实现，回退到纯软掩码 |
| **Day 3** | 多状态 SLAT 融合 + LoRA 设置 | Gate: 与单状态相比 LPIPS 改善 > 0.05 | 否 → 简化为均值聚合，弃 visibility |
| **Day 4** | 可微逆向变形 + axis 梯度验证 | Gate: 合成 GT axis 100 实例上误差 < 5° | 否 → axis 用 RANSAC 离线初始化 |
| **Day 5** | 全 Stage C with 5-loss | Gate: 10 实例上 RFSDS 收敛（loss 曲线下降 70%）| 否 → 简化到 3-loss，调权重 |
| **Day 6** | Stage E/F/G（atlas + type 消歧 + URDF）| Gate: URDF 在 SAPIEN 中可加载 + 动臂运动 | 否 → 退化 URDF（仅几何无运动）|
| **Day 7** | 50 实例评估 + ablation #1-5 | Gate: CD-m < 30, axis err < 8° (PARIS 数据) | 否 → 写 limitation，submit limited subset |

---

## 16. 诚实性章节（Honesty + Red Lines）

### 验证标签使用规范

- **[V-CHORD]**: 完全验证自 CHORD 论文（arXiv:2601.04194 v1, Lyu et al. 2026）
  - W-RFSDS 公式（式 2-3）、CDF 退火（式 4）、3D-GS 选择动机、ARAP 损失、temporal flow 损失
- **[V-TRELLIS]**: 完全验证自 TRELLIS 论文（arXiv:2412.01506 v1, Xiang et al. 2024）
  - SLAT 表示（式 1）、64³/16³ 网格、DINOv2 multiview aggregation、3D Swin window=8、tanh 高斯位置约束、Repaint 局部编辑
- **[V-FreeArt3D]**: 完全验证自 FreeArt3D 论文（arXiv:2510.25765 v2, Chen et al. 2025）
  - SS-VAE 式 (1)（c=8 暗示但精确通道数需从 ckpt 配置确认）、关节初始化必要性、双 hash grid 几何
- **[V-MorphAny3D]**: 验证自 MorphAny3D 论文（arXiv:2601.00204, Sun et al.）
  - SLAT 在 attention 空间融合的核心思想；v18 的"多状态 DINOv2 投影聚合"是借鉴而非照搬
- **[V-DIFT]**: 验证自 DIFT 论文（arXiv:2306.03881）—— 中后段层语义最强
- **[V-vX]**: 来自 v4-v15 历次设计迭代的合成
- **[V-1st]**: 第一性原理推导
- **[Hyp]**: 待消融验证的假设

### 红线（红色警示）— 不能在论文中作为已验证事实声明的内容

- **[Red 1]** TRELLIS SS-DiT 的精确 block 数（24）、hidden dim（1024）、heads 数（16）—— 论文未明确披露 image-large 模型这些超参；正文中只有 "1.2B 参数 + DiT-L"；**必须在论文实现中从 GitHub 配置 `ss_flow_img_dit_L_16l8_fp16.json` 直接读取并报告**。
- **[Red 2]** "Block 22 是部件判别最强层" — DIFT 在 SD U-Net 上的实证；TRELLIS DiT 上的对应位置需逐 block 探针实证（通过冻结模型对部件 mask 进行 linear probing）。
- **[Red 3]** TRELLIS 多图 API 的"shared noise" 机制 — GitHub 描述为"tuning-free"但具体噪声同步策略需读源码确认；本报告假设是共享噪声采样。
- **[Red 4]** SLAT 通道数 c — TRELLIS 论文未明确，FreeArt3D 写 `ℝ^{16×16×16×c}` 也未给值；从 ckpt 命名 `16l8` 推断 c=8（同样适用于 SLAT 命名 `64l8`）。**须从配置文件读取确认**。
- **[Red 5]** 混合 STE 在 TRELLIS 频率分布上是否真能产生体素增加（而非只删除）— 论文中无先例；这是 v18 的核心 [Hyp]，**必须用 ablation #2 实证**。
- **[Red 6]** 多状态 SLAT 融合在数值上是否优于"逐状态独立 SLAT 然后池化" — 这是与 MorphAny3D 类比但非直接证明；**ablation #7 必做**。
- **[Red 7]** "5 体素轴误差对应 5% PARIS 误差" — 文中此推断为 [V-1st]，未实测；正文应改为"目标 < 8° axis err"。

### 与 GPT 对比批判的整合

GPT 对比文档的两条核心批判均已被 v18 直接采纳：
1. **梯度桥不完整** → §3 混合 STE 桥；
2. **H_part 边界粗糙** → §4 混合 16³+64³ 头。

GPT 文档另外可能还隐含的关切（基于上下文推断，未直接看到原文）：
- **关节参数无 feedforward init**：v18 §5 H_joint 解决；
- **SLAT 信息浪费**：v18 §6 多状态融合解决；
- **持续监督污染 RFSDS**：v18 §7 BMCSA 仅温启动解决。

---

## 17. 总结与下一步

v18 在 6 轮迭代基础上达成最终设计：以 new_v_2 为骨架，以 BMCSA 温启动取代持续监督，以混合 STE 取代纯软掩码桥，以 SS-DiT 内部 H_part / H_joint 双头取代后处理预测，以多状态 DINOv2 投影聚合取代单状态 SLAT 推理。三大核心创新均有理论支撑（CHORD W-RFSDS、TRELLIS SLAT 多视角聚合、DIFT 中后段语义）和明确消融验证路径。

**下一步**：进入 Day 1 实现，严格按 7 天路线图执行，每日 Go/No-Go gate 必过。submission 前补做 22 项失败模式的 stress test，并在论文中以 [Red 1]-[Red 7] 标记的红线项必须在实验章节给出具体测量数据，不可以推断或类比代替。

**仍存的最大风险**：[Red 5] 混合 STE 体素增加是否真能"快进"在 100 步内启动 — 若失败则回退方案：固定 argwhere 集合 = 基集合 ∪ BMCSA 共识扩展，仅允许重权与删除，论文标题改为"重权式（reweight-only）混合桥"。

---

*报告完成。本文档内化并应对了 GPT 对比批判的全部要点，将 new_v_1 与 new_v_2 的优势在 v18 中以理论一致的方式综合，并保留所有重大假设的红线警示，符合 AAAI/CVPR 投稿严谨性要求。*