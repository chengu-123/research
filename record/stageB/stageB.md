# Stage B — Base-Consistent Cross-State Occupancy Generation

> **Last updated**: 2026-04-22
> **Status**: v4.3（BMCSA + upstream M_attn filter）为生产默认
> **Authority**: 本文档是 Stage B 唯一 authoritative 来源。以下文件均被本文档 supersede：
> - `record/stageB/stageb_detail.md`（2026-04-18 历史快照，仅覆盖到 v4.1 + v4.3 proposed）
> - `record/stageB/2026-04-18-stageb-v3-sdedit-design.md`（v3 设计 spec，legacy / ablation baseline）
> - `record/stageB/2026-04-18-stageb-v4-bmcsa-design.md`（v4 设计 spec，已过时）
> - `record/stageB/redesign.md`（v3 初版任务清单，纯历史）
> - `record/design.md §3`（v4 之前的"当前实现"快照，已过时）

---

## Table of Contents

1. [定位与 I/O 契约](#1-定位与-io-契约)
2. [当前方案（v4.3）](#2-当前方案-v43)
3. [关键公式与数据流](#3-关键公式与数据流)
4. [超参表](#4-超参表)
5. [代码位置与修改索引](#5-代码位置与修改索引)
6. [产物与诊断输出](#6-产物与诊断输出)
7. [演进时间线（v0 → v4.3）](#7-演进时间线)
8. [历史问题清单（13 条）](#8-历史问题清单)
9. [当前已知限制](#9-当前已知限制)
10. [消融计划（CVPR / AAAI）](#10-消融计划)
11. [文献锚点与设计决策](#11-文献锚点与设计决策)

---

## 1. 定位与 I/O 契约

### 1.1 Stage B 在 pipeline 中的角色

```
[I_0..I_5 带地毯分割图]             ← Stage A (video diffusion, K=6 states)
         │
         ▼
    Stage B (本阶段)
         │
         ▼
[O_k (K, 64³) + z_final (K,8,16³) + M_attn_64 (64³)]
         │
         ▼
    Stage C/D/F 下游
```

**Stage B 解决的核心问题**：TRELLIS SS flow model 在 K-parallel 模式下（即使共享噪声 + per-state c_k 图像条件），各 state 的 DiT forward 仍独立；最终输出的**柜体几何存在跨状态 drift**，state 0 还有额外的 **TRELLIS 闭合态系统性缩小 bias**。Stage B 消除这两个问题，同时**保留 per-state move 差异**（抽屉/门在各状态位置不同）。

### 1.2 输入契约

- **`I_0..I_{K-1}`**：K 张带地毯（grounding disk）的分割图，`outputs/<sample_id>/rendering_joint_00_state_{00..K-1}.png`
- **`c_0..c_{K-1}`**：DINOv2 patch features（ViT-L/14, frozen，每个 1374 tokens × 1024 dim）
- **`eps ~ N(0, I)`**：共享噪声，shape `(1, 8, 16, 16, 16)`，扩为 `z_0^(k) = eps.repeat(K)` 作为 K-parallel 采样起点

### 1.3 输出契约

- **`O_stack: (K, 64, 64, 64)`** — 二值占据图，**base 跨状态一致 + move per-state 保持**
- **`O_stack_soft: (K, 64, 64, 64)`** — sigmoid 软概率版
- **`z_final: (K, 8, 16, 16, 16)`** — Pass 2 终态 SS latent，下游 Stage C 拿来做 correspondence feature
- **`M_attn_64: (64, 64, 64)`** — 跨状态语义一致性 mask，下游 Stage C 作 graph-cut unary prior

### 1.4 静态假设

- 单物体、单关节、关节类型 ∈ {revolute, prismatic}
- **相机固定，物体动**（world 坐标系，无需 camera pose 解算）
- 所有神经模型（DINOv2、TRELLIS SS VAE + SS DiT）**frozen**，**training-free**
- 带地毯输入下 TRELLIS 不产生显著位置漂移，仅轻微表面抖动

---

## 2. 当前方案（v4.3）

### 2.1 三句话 Summary

1. **Pass 1**：SCAR sampler 用 symmetric mix（无 push）做 K-parallel 粗对齐，得 per-state z_final_p1 和 soft_p1。
2. **M_attn 上游过滤**：从 z_final_p1 计算跨状态特征一致性 M_attn（16³），trilinear 上采到 64³，**先乘进 P_base_shared**（几何 mean），得到几何 AND 语义都同意的 base 场。
3. **Pass 2**：从 `z_guide_k + shared_noise` 起点 SDEdit 12 步，每个 DiT self-attn 层做 **BMCSA blend**（per-state self-attn vs cross-state K/V mean，按 M_base 加权），K=6 同批刷新。

### 2.2 完整数据流

```
                 ┌───────────── Pass 1 (25 Euler steps) ─────────────┐
                 │                                                    │
c_0..c_{K-1}  →  │  SCAR sampler                                      │
eps (shared)  →  │    mix_steps=8, symmetric, weights=(0.3,0.4,0.3)   │
                 │    alpha_peak=0.0 (push disabled)                  │
                 │                                                    │
                 └──┬─────────────────────────────────────────────────┘
                    │
                    ▼
              z_final_p1 (K, 8, 16, 16, 16)  ─── decode ──→  soft_p1 (K, 64³)
                    │                                              │
                    │  [M_attn 通道]                                │  [几何通道]
                    ▼                                              ▼
        agree(ℓ) = mean_{k≠k'} cos(ẑ_k(ℓ), ẑ_{k'}(ℓ))    P_base_shared_raw
        M_attn_16 = sigmoid((agree - τ)/κ)                    = mean_{k=0..K-1} soft_p1[k]
        M_attn_64 = trilinear_upsample(M_attn_16, 64³)        (不排除 state 0; 1/K 稀释)
                    │                                              │
                    └────────────── 乘 (v4.3 关键步) ──────────────┘
                                         ▼
                     P_base_shared = P_base_shared_raw · M_attn_64
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                                       ▼
              per-state guide                         M_base token-space
              ─────────────────                       ──────────────────
              P_excl_k = ReLU(soft_p1[k] -            M_base_64 = sigmoid((P_base_shared - 0.5)/τ_M)
                             max_{j≠k} soft_p1[j])    avg_pool3d(kernel=4) → (1, 16³, 1)
              P_guide_k = max(P_base_shared, P_excl_k)       = M_base_flat (1, L=4096, 1)
              O_guide_k = (P_guide_k > 0.5)
              z_guide_k = SSEncoder(O_guide_k)
                     │                                       │
                     ▼                                       │
              guides (K, 8, 16³)                             │
                     │                                       │
                     ▼                                       │
              x_{t*}^(k) = (1-t*) · z_guide_k +              │
                           (σ_min + (1-σ_min)·t*) · eps_shared
                     │                                       │
                     ▼                                       │
              ┌──────── Pass 2 (12 fixed Euler steps) ───────▼─────────┐
              │                                                         │
              │  at each DiT self-attn layer (all 24 blocks):          │
              │      y_self = self_attn(h)            # per-state       │
              │      y_shared = self_attn(h, share_kv_across_batch=T)   │
              │      h = (1 - M_base_flat) · y_self                     │
              │         + M_base_flat · y_shared                        │
              │  (cross-attn to per-state c_k 不变)                     │
              │                                                         │
              └──┬──────────────────────────────────────────────────────┘
                 │
                 ▼
          z_final (K, 8, 16³)  ─── decode ──→  O_stack (K, 64³)
                 │                                    │
                 ▼                                    ▼
          [下游 Stage C/D 输入]                remove_disk → base_move_preview
```

### 2.3 三个关键设计点

**(A) M_attn 上游过滤是 v4.3 的核心差异**（相对 v4.2）。v4.2 把 M_attn 仅用于 BMCSA blend 阶段；v4.3 把 M_attn 先升到 64³ 直接过滤 P_base_shared，**让 guide 起点就正确**。这解决了 Problem 13（26525 样本 s0-2 collapse）。

**(B) P_base_shared 用 mean over ALL K**（含 state 0），不是 mean over s1..K-1。理由：K=6 下 state 0 贡献仅 1/K ≈ 17%，state 0 shrunk bias 被稀释到不跨阈值；同时保留 state 0 的真实图像贡献。`exclude_state_0=true` 作 ablation 旋钮保留。

**(C) Shared noise 在 Pass 1/2 之间复用**。Pass 2 的 `eps_shared` 就是 Pass 1 的初始化噪声（K 份拷贝），保持 K-parallel 的"同一随机家族"不变量。不重新采 Gaussian。

---

## 3. 关键公式与数据流

### 3.1 Pass 1 symmetric mix（步 0-7）

对每个 state `k`，前 8 步里：

```
z_t^(k) ← w_first · z_t^(0) + w_self · z_t^(k) + w_last · z_t^(K-1)
```

默认 `(w_first, w_self, w_last) = (0.3, 0.4, 0.3)`。**每个 state 保留 40% 自己**，避免 `mean_of_middles` 那样把 K 个 latent 归到同一值引起的过度同质化。

### 3.2 P_base_shared（v4.1 canonical, v4.3 加 M_attn 过滤）

```
# v4.1: shared base 在所有 K targets 统一
P_base_shared_raw = mean_{k=0..K-1} soft_p1[k]              # (64, 64, 64)

# v4.3: 上游过滤，语义一致性 AND 几何共识
P_base_shared = P_base_shared_raw × M_attn_64                # (64, 64, 64)
```

**为什么 mean 不是 min**：min 对单一离群 state（= state 0 shrunk）过度敏感会拖垮 base 识别。mean 下 state 0 贡献 1/K = 17%，即使 state 0 = 0.2 而 state 1-5 = 0.9，mean = 0.78 仍然 > 0.5 阈值。

### 3.3 M_attn（跨状态语义一致性）

对每个 token 位置 ℓ：

```
ẑ_k(ℓ) = z_final_p1[k, :, ℓ] / ||z_final_p1[k, :, ℓ]||       # L2 normalize over channel
agree(ℓ) = 1/(K·(K-1)) · Σ_{k≠k'} <ẑ_k(ℓ), ẑ_{k'}(ℓ)>       # pairwise cosine, avg off-diag
M_attn_16(ℓ) = sigmoid((agree(ℓ) - τ) / κ)                  # τ=0.7, κ=0.05 default
M_attn_64 = trilinear_upsample(M_attn_16, 16³→64³)
```

- **输出是单通道 scalar field**（不是 per-state feature）
- 高 M_attn → 跨状态 feature 一致 → 柜体样式 → likely base
- 低 M_attn → state 0 air vs state k drawer → likely move（或 trajectory 中段）

**M_attn 盲点**：只区分"是否一致"，不区分"一致的是什么材质"。revolute hinge 线上的 voxel（跨 state 都是 drawer material，位置不动）会被误判为 base。盲点影响范围小（~O(10) voxel），在 prismatic 上完全不存在。

### 3.4 Per-state guide

```
P_excl_k = ReLU(soft_p1[k] - max_{j ≠ k} soft_p1[j])        # state k 独占 voxel
P_guide_k = max(P_base_shared, P_excl_k)                     # shared base + per-state excl
O_guide_k = (P_guide_k > 0.5).float()                         # binarize to match encoder dist
z_guide_k = SparseStructureEncoder(O_guide_k.unsqueeze(0).unsqueeze(0))  # (1, 8, 16³)
```

所有 K 个 `z_guide_k` 的 base 部分完全相同（共享 P_base_shared），差异只在 `P_excl_k`。

### 3.5 M_base token-space gate（供 BMCSA 使用）

```
M_base_64 = sigmoid((P_base_shared - 0.5) / τ_M)             # τ_M=0.05; soft gate around 0.5
M_base_tok = avg_pool3d(M_base_64[None, None], kernel=4, stride=4)  # 64³ → 16³
M_base_flat = M_base_tok.view(1, L=4096, 1)                   # 广播到 (K, L, D) via L 轴
```

TRELLIS-image-large `resolution=16, patch_size=1` → token count = 16³ = 4096（**不是 8³**），pool_kernel = 64/16 = 4。

### 3.6 SDEdit 起点

```
t_star = 0.5
eps_shared = Pass 1 的初始噪声 (扩为 K 份)
x_{t*}^(k) = (1 - t*) · z_guide_k + (σ_min + (1-σ_min) · t*) · eps_shared
```

`σ_min ≈ 1e-5`（TRELLIS SS flow model 配置）。Signal 权重 0.5，噪声权重约 0.5。

### 3.7 Pass 2 BMCSA 内循环

**Schedule**：`np.linspace(0.5, 0, 13)` → **12 fixed steps**。不按 `ceil(total_steps × t_star)` 绑定。

**每个 DiT `ModulatedTransformerCrossBlock._forward` 里**（全 24 blocks 默认全开）：

```python
# Standard per-state self-attn
y_self = self.self_attn(h)                              # (K, 4096, D)

# Shared self-attn: Q 每个 state 自己, K/V 跨 batch 取 mean
y_shared = self.self_attn(h, share_kv_across_batch=True)   # (K, 4096, D)

# Spatial blend
M = M_base_flat.to(y_self.dtype)                         # (1, 4096, 1)
eff_M = clamp(bmcsa_strength × M, 0, 1)
h = (1 - eff_M) × y_self + eff_M × y_shared
```

Cross-attn（to per-state `c_k`）**不变**。

### 3.8 最终输出

```
z_final = Pass 2 denoise 到 t=0 的 latent           # (K, 8, 16, 16, 16)
O_stack_soft = sigmoid(decoder(z_final))             # (K, 64, 64, 64)
O_stack = (O_stack_soft > 0.5).float()
remove_disk(O_stack)                                  # 可选，去地毯 voxel
```

**替换 ALL K states**（不只 state 0）。

---

## 4. 超参表

### 4.1 Pass 1（SCAR）

| 参数 | 默认 | 作用 |
|---|---|---|
| `sampler.total_steps` | 25 | Euler 步数，TRELLIS default + 匹配 FreeArt3D |
| `scar.mix_steps` | 8 | Mix 阶段步数（步 0-7） |
| `scar.extreme_mix_mode` | `symmetric` | Mix 公式；`mean_of_middles` 为 legacy ablation |
| `scar.mix_weights` | `(0.3, 0.4, 0.3)` | `(w_first, w_self, w_last)`；self-retention 40% |
| `scar.alpha_peak` | `0.0` | Push 禁用（Problem 1-2 的结论） |
| `scar.icp_enabled` | `false` | Post-hoc ICP 禁用（Problem 见 §8） |

### 4.2 Pass 2（SDEdit + BMCSA + M_attn）

| 参数 | 默认 | 作用 |
|---|---|---|
| `stage_b_sdedit.mode` | `bmcsa` | v3 遗留 / `bmcsa`；`bmcsa` 即 v4.x 路径 |
| `stage_b_sdedit.t_star` | `0.5` | SDEdit 噪声水平 |
| `stage_b_sdedit.pass2_steps` | `12` | 固定，**独立于 t_star**（v3 bug 修复） |
| `stage_b_sdedit.tau_M` | `0.05` | M_base sigmoid 锐度 |
| `stage_b_sdedit.token_resolution` | `16` | DiT token 空间分辨率（auto 推断） |
| `stage_b_sdedit.exclude_state_0` | `false` | P_base_shared 是否排除 state 0（ablation） |
| `stage_b_sdedit.bmcsa_blocks` | `all` | BMCSA 应用到哪些 DiT block（`all` 或 list） |
| `stage_b_sdedit.bmcsa_strength` | `1.0` | M_base × strength 的 cap 系数 |
| `stage_b_sdedit.attn_m_enabled` | `true` | 启用 M_attn 分支 |
| `stage_b_sdedit.attn_m_apply_at` | `guide` | **v4.3 关键**：`guide`（上游 P_base 过滤）或 `bmcsa`（仅 attention blend） |
| `stage_b_sdedit.attn_m_threshold` | `0.7` | M_attn sigmoid 中心 τ |
| `stage_b_sdedit.attn_m_tau` | `0.05` | M_attn sigmoid 锐度 κ |
| `stage_b_sdedit.guide_mode` | `augmented_intersection` | 或 `pure_intersection` |

所有其他 sampler 参数（`rescale_t`, `cfg_strength`, `cfg_interval`）继承 Pass 1 默认。

---

## 5. 代码位置与修改索引

### 5.1 TRELLIS 内部（source edit，遵循 MorphAny3D 的 kwargs pattern，无 hook 无 monkey-patch）

| 文件 | 函数/类 | 改动 |
|---|---|---|
| `mine/TRELLIS/trellis/modules/attention/modules.py` | `MultiHeadAttention.forward` | 新增 `share_kv_across_batch: bool = False` kwarg。True 且 B>1 时，K/V 跨 batch 取 mean 再 expand。做在 RoPE 之后、qk_rms_norm 之前 |
| `mine/TRELLIS/trellis/modules/transformer/modulated.py` | `ModulatedTransformerCrossBlock._forward` | 加 BMCSA 分支：`bmcsa_flag=True` 且 `block_idx ∈ bmcsa_blocks` 时，y_self + y_shared + M_base blend |
| `mine/TRELLIS/trellis/models/sparse_structure_flow.py` | `SparseStructureFlowModel.forward` | 接受 `**kwargs` 并透传 `block_idx=i` 给每个 block |

### 5.2 Pipeline

| 文件 | 函数 | 职责 |
|---|---|---|
| `mine/pipelines/stage_b_scar.py` | `run_scar` | 主 driver |
| — | `_compute_P_base_shared` | 从 soft_p1 计算 mean-over-K base 场 |
| — | `_compute_M_base_tokenspace` | P_base → 64³ sigmoid → 16³ avg_pool → (1, 4096, 1) |
| — | `_compute_M_attn_tokenspace` | z_final → pairwise cosine agreement → sigmoid → (1, 4096, 1) + (L,) raw agreement |
| — | `_build_augmented_intersection_guide` | P_stack + P_base_shared → P_guide_k → encode → z_guide_k |
| — | `_sdedit_refine_k6_bmcsa` | Pass 2 12 步 Euler + BMCSA kwargs 注入 |
| `mine/pipelines/recon.py` | `_load_sparse_structure_encoder` | 手动加载 encoder（TRELLIS 默认 pipeline.json 不含） |
| `mine/pipelines/sajo/anchors.py` | `joint_free_split` | **2026-04-22 bug-fix**: 新增 `mode="footprint"` (default) 和 `M_attn` 参数。footprint 公式 `p_move = footprint - p_base` 修 prismatic 长抽屉 endpoint 丢 voxel bug；M_attn 乘进 p_base 对齐 v4.3 语义先验 |
| `mine/configs/v1.yaml` | `stage_b_sdedit` block | 所有 v4.x 超参 |

---

## 6. 产物与诊断输出

### 6.1 目录结构

```
<output_dir>/stage_b/
├─ z_final.pt                         # 最终 SS latent (K, 8, 16, 16, 16)
├─ O_stack.npy / _soft.npy             # 最终 K 个 occupancy (Pass 2 后) (K, 64³)
├─ O_stack_pass1.npy / _soft.npy       # Pass 1 对照（未经 Pass 2）
├─ p_base_preview.npy                  # joint_free_split(footprint, M_attn) 预览
├─ p_move_preview.npy                  # 同上
├─ sdedit_report.json                  # mode / t_star / pass2_steps / voxel_deltas / M_attn stats
│
├─ viz/
│   ├─ O_stack.html                    # 合并 K state dropdown viz
│   ├─ O_k_{00..0(K-1)}.html          # 单 state viz
│   ├─ base_move_preview.html          # footprint × M_attn 预览（Stage B 是否真干净的证据）
│   │
│   ├─ guide/
│   │   ├─ states_1_to_Km1_overlap_*.html   # s1..K-1 soft intersection
│   │   ├─ state{k}_P_base.html              # 所有 k 相同（= P_base_shared）
│   │   ├─ state{k}_P_excl.html              # 每个 k 不同
│   │   ├─ state{k}_P_guide.html             # max(P_base, P_excl_k)
│   │   └─ state{k}_O_guide_bin.html         # (P_guide > 0.5)
│   │
│   ├─ bmcsa/
│   │   ├─ M_base_64.html / .npy             # sigmoid 后的 soft gate (64³)
│   │   ├─ M_attn_16.html / .npy             # v4.2+ 语义 mask (16³ token space)
│   │   ├─ M_attn_64.html / .npy             # v4.3 trilinear 上采到 64³
│   │   ├─ P_base_shared_64_raw.html         # mean over K 的 raw 场（未经 M_attn 过滤）
│   │   └─ P_base_shared_64_filtered.html    # × M_attn_64 后
│   │
│   └─ scar_diagnostics.html                 # Pass 1 每步 mask / push_norm 曲线
│
├─ per_step/                                 # Pass 1 25 步演化
│   └─ O_step_00.html .. O_step_24.html
│
└─ pass2_per_step/                           # Pass 2 12 步演化
    ├─ O_step_00.html                        # t*=0.5 noisy starting point
    └─ O_step_11.html                        # t=0 最终（应等同 O_k_*.html）
```

### 6.2 关键诊断 key

- **`sdedit_report.json` 的 `voxel_deltas`**：Pass 1 vs Pass 2 的 per-state voxel 数差。state 0 应 > 0（被 fill 更大）；其他 state 接近 0。
- **`base_move_preview.html`**：Stage C 判准。若 base 侵入 move 区域，说明 M_attn 过滤还不够强（旧版表现；v4.3 基本消除）。
- **`M_attn_64.html`**：在"柜体"voxel 上应该接近 1（高一致），在 trajectory 中段应接近 0 甚至更低（state 0 air vs 其他 state drawer 不一致）。若 trajectory 中段 M_attn > 0.5，说明 τ/κ 需调。

---

## 7. 演进时间线

| 版本 | 机制 | 状态 | 主要缺陷 |
|---|---|---|---|
| **v0** pre-SCAR | Plain K-parallel | deprecated | 无 consistency，base 大幅 drift |
| **v1 VGCF** | Variance-gated consensus force（Tweedie 空间梯度推）| deprecated | 力场方向不可控，时好时坏 |
| **v2 BCAC** | Block-level consensus attention control（仅 DiT 中层 block）| deprecated | 作用不充分 |
| **v3a** SCAR + α/t push | Mix（latent 混合）+ Tweedie push (α/t)| legacy ablation | push 在 t→0 发散 |
| **v3b** SCAR + α·t push | 同上，α·t 替换 α/t | legacy ablation | push 仍然时好时坏 |
| **v3c** SDEdit + augmented intersection | Pass 1 SCAR → Pass 2 SDEdit with per-state guide | legacy | Pass 2 12 步内 K state 独立演化；P_base per-target 不一致 |
| **v4** BMCSA | 保留 SDEdit 起点 + Pass 2 去噪期每 DiT self-attn 做 cross-state blend | 已实现 | base 判定在 long-drawer 平移下有 ~2 voxel 污染 |
| **v4.1** shared P_base | v4 的 P_base 修复为所有 target 共享 mean(s0..K-1) | legacy | 同 v4 |
| **v4.2** M_attn in BMCSA | 加入 M_attn；`M_eff = M_base × M_attn` 只在 BMCSA blend 生效 | legacy | guide 里的 P_base 没过滤，起点依然 over-large |
| **v4.3** M_attn upstream | **当前默认**。M_attn 上采到 64³ 直接过滤 P_base_shared，guide 源头就正确 | **生产** | 下见 §9 |

---

## 8. 历史问题清单

### Problem 1 ─ Tweedie push (α/t) 在 t→0 发散
- **症状**：push 强度在采样末期爆炸
- **根因**：`push ∝ α/t`，t→0 时 1/t → ∞
- **修复**：换成 `α·t`（σ_t normalization, Kynkäänniemi 2024 arxiv:2404.07724）
- **当前**：push 完全禁用（见 Problem 2）

### Problem 2 ─ Push 时好时坏（fundamental）
- **症状**：有的数据 push 后好，有的漂得更厉害
- **根因**：push 朝 cross-state consensus 拉，consensus 被 state 0 outlier 污染（自指）；push 方向与 DiT 原生 denoising direction 不一致
- **修复**：`alpha_peak = 0.0`，push 完全禁用；改走 mix + Pass 2 架构

### Problem 3 ─ `mean_of_middles` mix 过度同质化
- **症状**：state 1-4 之间相似度过高，几乎融合
- **根因**：每步把 K 个 latent 归到同一 mixed
- **修复**：换 `symmetric`，每个 state 保留 40% 自己

### Problem 4 ─ v4-draft 的 per-target P_base 不一致
- **症状**：每个 state 的 `state{k}_P_base.html` 长得都不一样，且 state 5 的 P_base 被 state 0 shrink 污染（min 池含 state 0）
- **根因**：用 `min_{j != target} P^(j)` 算 per-target base
- **修复**：v4.1 共享 `mean_{k=0..K-1} P^(k)`

### Problem 5 ─ Pass 2 步数被 `ceil(25 × t*)` 绑定
- **修复**：加 `pass2_steps: 12` config，固定步数，t_star 和步数解耦

### Problem 6 ─ `patch_size=2` 假设错 → token 数 8³ 错算
- **症状**：BMCSA hook `M_base` shape (1, 512, 1)，但 attention 输入 (K, 4096, D)，broadcast 失败
- **根因**：TRELLIS-image-large 实际 `patch_size=1`，token = 16³ = 4096
- **修复**：从 flow_model 属性自动推 `token_resolution = resolution / patch_size`

### Problem 7 ─ SparseStructureEncoder 不在默认 `pipe.models`
- **修复**：`pipelines/recon.py` 新增 `_load_sparse_structure_encoder()`，从 `JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16` 手动拉

### Problem 8 ─ `_sdedit_refine_k6_bmcsa` 里 `edict` 未 import
- **修复**：加 local `from easydict import EasyDict as edict`

### Problem 9 ─ Viz P_base 和 M_base 的 exclude_state_0 不一致
- **修复**：统一走 `_compute_P_base_shared` helper

### Problem 10 ─ 长抽屉 + 小平移下 trajectory 中段误判 base（theoretical）
- **场景**：prismatic 长 drawer（L=5 voxel）平移小（shift=1/state），trajectory 中段被 5-6 个 state 覆盖 → `mean > 0.5` → 误判 base
- **v4.3 修复**：M_attn 过滤掉，因为 state 0 在 trajectory 中段是 air，跨 state 语义不一致
- **残留**：若 M_attn 阈值过松可能漏过部分；26525 是 problem 13 的典型

### Problem 11 ─ P_excl 设计缺陷（mild）
- **描述**：在平移下 `P_excl_k` 只能捕捉每个 state 相对其他 state 新增的 voxel；中间 state 的 P_excl 可能几乎全 0
- **为什么 OK**：Pass 2 去噪阶段 BMCSA 只在 M_base 高处 blend；中间 state 的 drawer 靠 c_k + DiT 重生，不依赖 P_excl_k
- **决定**：graceful degradation，机制本身没错，不修

### Problem 12 ─ TRELLIS 软/硬空输出分布未验证
- **场景**：决定"软 state-0 veto"方案是否可行的关键实验
- **状态**：未执行，方案 3（M_attn）直接跳过

### Problem 13 ─ guide 里 P_base 过大侵占 move voxel
- **实验场景**（2026-04-18, v4.2）：30857（平移，好）/ 26525（平移，**state 0-2 near-collapse**）/ 7128（旋转，好）/ 7201（旋转，好）
- **根因**：M_attn 在 16³ token 空间只作用于 BMCSA blend，guide 构造用的 P_base_shared_64 完全不受 M_attn 影响；26525 trajectory 极长 + 平移小 → mean over K 在中段 5/6 或 6/6 → over-large base → guide 几乎把 drawer 区域也算进去 → Pass 2 12 步救不回来
- **v4.3 修复**：M_attn 上采到 64³ **直接过滤 P_base_shared**，源头就正确。代码：`P_base_shared_filtered = P_base_shared_raw * M_attn_64`，下游所有用 P_base 的地方（guide, M_base token）都用 filtered 版
- **为什么源头修而不是 downstream 修**：(a) 源头错 Pass 2 12 步去噪救不回来（26525 实证）；(b) 单点真相（一个 M_base 概念融合几何+语义），代码路径干净；(c) paper story 更清爽

### Problem 14 ─ `joint_free_split` 旧公式在 prismatic 长抽屉 endpoint 丢 voxel（2026-04-22）
- **症状**：`p_move = mu · sigmoid((std - σ_m)/τ_m)` 在 endpoint（仅 1 state 占据）下 `mu ≈ 1/6, std ≈ 0.37 → p_move ≈ 0.165 < 0.5` → 二值化后 drop
- **修复**：`anchors.py joint_free_split` 加 `mode="footprint"` (default)：`p_move = clamp(max_k O_k - p_base, 0)`；加 `M_attn` 参数乘进 `p_base` 对齐 v4.3 语义先验
- **已知限制**：footprint 公式对 TRELLIS 单 state 幻想 voxel 不鲁棒（`p_move → 1` 包含孤立噪声）。目前仅用于 viz preview，若 Stage C production 复用需加 `min_observers ≥ 2` 投票滤噪

---

## 9. 当前已知限制

### 9.1 M_attn 的语义盲点（§3.3 详述）

**不能区分 "consistently occupied by drawer" vs "consistently occupied by cabinet"**。两者 M_attn 均高 → 均被判 base。

- **Prismatic 下几乎不出现**：drawer 平移时没 voxel 在 world 系跨 state 不动
- **Revolute 下影响**：hinge 轴线上少数 voxel（~O(10)）被误判 base
- **Mitigation 路径**（future work）：(a) z_final 作 feature 做 per-voxel k-means；(b) DiT mid-layer hidden hook (1024-dim)；(c) 接 PartField 或 Diff3F
- **当前决定**：MVP 接受；AAAI paper 写 ablation

### 9.2 P_excl 在中间 state 几乎为 0（Problem 11）

Graceful degradation，不修。

### 9.3 Stage B 输出不是 PartField 意义上的 "part feature"

`z_final` 是 shape-reconstructive latent（8-dim VAE 压缩），`M_attn` 是 1-dim agreement scalar。**都不是 intrinsic part-discriminative feature**。下游 Stage C 的 part 判别只能靠 motion-derived signal + z_final feature correspondence，不能直接 clustering 出 part。

### 9.4 Pass 2 OOD 风险（v3 遗留，部分缓解）

Encoder 训练在 Objaverse 真实 binary occupancy，`E(augmented_intersection)` 的 synthetic union 是轻度 OOD。BMCSA 把特征拉回 DiT native manifold，部分缓解。检测路径：比较 `||z_guide_k||` vs `||z_final_p1^(k)||` magnitude。

### 9.5 BMCSA 过 homogenization at move boundary（轻微）

M_base 是 sigmoid soft 值，move/base 边界 voxel 有 `M ≈ 0.5` → 50% shared attention → 可能 homogenize 边界特征。检测：比较相邻 state drawer 边界的"锐度"。Mitigation：`tau_M=0.02` 更陡，或 hard threshold。

---

## 10. 消融计划

### 10.1 主消融矩阵（AAAI experiments 表）

| ID | 配置 | 作用 |
|---|---|---|
| **A0** | Plain K-parallel（无 mix / push / BMCSA） | 下界 baseline |
| A1 | Pass 1 only（mix + push） | legacy SCAR baseline |
| A2 | Pass 1 mix-only（push off） | Mix 单独效应 |
| A3 | v3 SDEdit（augmented intersection guide, 无 BMCSA） | SDEdit 单独效应 |
| **A4** | v4.1 BMCSA（无 M_attn）| BMCSA 单独效应 |
| A5 | v4.2（M_attn 仅 BMCSA blend）| M_attn 在 downstream 是否够 |
| **A6** | **v4.3（M_attn upstream 过滤 P_base）** | **full method, 当前默认** |
| A7 | v4.3 但 `bmcsa_blocks = middle half only` | block selectivity |
| A8 | v4.3 但 `bmcsa_strength = 0.5` | blend strength |
| A9 | v4.3 但 `exclude_state_0 = true`（M_base = mean s1..K-1）| state 0 是否该排除 |
| A10 | v4.3 但 `guide_mode = pure_intersection` | per-state P_excl 是否必要 |
| A11 | v4.3 不同 `t_star ∈ {0.3, 0.5, 0.7}` | SDEdit 噪声水平 sweep |
| A12 | v4.3 不同 `attn_m_threshold ∈ {0.5, 0.6, 0.7, 0.8}` | M_attn 敏感度 |

Expected story：A0 < A2 ≈ A3 < A4 < A5 ≈ A6（后者领先 Problem 13 case）；A7-A12 接近 A6 但各弱一点。

### 10.2 失败样本 pathological case（paper Limitations）

- `26525`：long-drawer short-shift prismatic — v4.2 崩、v4.3 修好
- 人工合成: hinge axis 线（revolute）上的 voxel 被误判 base — §9.1

---

## 11. 文献锚点与设计决策

### 11.1 核心文献锚

| 思想 | 文献 | 我们借用的部分 |
|---|---|---|
| SDEdit 起点机制 | Meng et al. ICLR 2022 | Pass 2 的 `x_{t*} = (1-t*)·guide + σ_{t*}·ε` |
| Cross-state sharing | MVDream (Shi et al. ICLR 2024) | K-parallel 同批采样思想 |
| Source-edit + kwargs pattern | MorphAny3D (Sun et al. arXiv:2601.00204) | `share_kv_across_batch` / BMCSA 实现方式 |
| Post-attn blend（不 pre-attn KV concat） | MorphAny3D §Moving Cross-Attention | y_self + y_shared blend 而非 K/V concat（后者引入 OOD attention pattern） |
| Cross-frame token matching | TokenFlow (Geyer et al. ICLR 2024 arXiv:2307.10373) | M_attn 的 cross-state feature agreement 类似 |
| Attention 编辑为语义 mask | Prompt-to-Prompt (Hertz et al. ICLR 2023 arXiv:2208.01626) | M_attn 作为可干预的 attention-level mask |
| Mutual self-attention | MasaCtrl (Cao et al. ICCV 2023 arXiv:2304.08465) | training-free K/V 共享思想 |
| α·t normalization | Kynkäänniemi 2024 arXiv:2404.07724 | Problem 1 的 α/t 发散修复（legacy; 当前 push 关） |

### 11.2 关键架构决策（可搬去 paper methods）

1. **两阶段架构（Pass 1 粗对齐 + Pass 2 精修）**：对齐 SDEdit + MVDream 家族
2. **Pass 1 shared noise 在 Pass 2 继承**：K-parallel 的"同一随机家族"不破坏
3. **Per-state z_guide_k = shared base + state-specific excl**：每个 state 的 Pass 2 起点含自己的 drawer hint
4. **BMCSA 是 post-attention blend，不是 pre-attention K/V concat**：遵循 MorphAny3D 的设计原理
5. **Source edit + kwargs 透传**：无 hook、无 monkey-patch；可追溯可 diff
6. **Encoder 专门加载**：不依赖默认 pipeline.json
7. **`exclude_state_0: false`（mean over all K）**：1/K 稀释 state 0 bias，不排除其贡献
8. **v4.3 M_attn 前置过滤 P_base**：(a) 源头修错而非 downstream 修；(b) 几何 + 语义融合单点真相；(c) paper story 干净

---

_End of stageB.md (v4.3 authoritative)_
