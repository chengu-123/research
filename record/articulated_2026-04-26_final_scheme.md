# ArtRASA — Final Scheme for AAAI Submission

**项目**：单图 → URDF-ready articulated 3D asset
**版本**：v5 final (2026-04-26)
**Status**: 锁定主线，准备实施
**之前文档归档**：`archive/articulated_v1_v4/`

---

## 0. One-line Thesis

> 给定一张闭态图 I₀，我们使用 frozen Wan2.2 I2V 自生成 K=6 个 pseudo-state conditioning frames；在 frozen TRELLIS SS-DiT 内部插入可学习 RASA 层，**每个优化迭代 (per-iteration) 都 in-loop 跑一次完整 SS-DiT canonical forward**，产出**单一 canonical sparse structure**；canonical 显式分解为 static base 与 movable part，K 个 state 占用通过 single-DoF rigid transform **确定性派生**（base bit-identical by construction）；frozen TRELLIS SLAT-DiT 内部用 MorphAny3D-inspired K-way MCA/TFSA 在 **共享的 canonical coordinates** 上融合 K 个 pseudo-state donor，输出 **单一 canonical SLAT**；最后用 Wan2.2 W-RFSDS 反传优化所有插入层和关节参数，输出一个物理一致的 URDF asset。

> **关键不变量** (per AAAI defensibility):
> 1. **Per-iteration in-loop forward**: 不是 cache-once-then-train-heads；每 iter SS-RASA 都在计算图里
> 2. **Bit-identical base by construction**: 不靠 alignment loss / merge，靠 `v_k = B ∪ T(q_k)·M` 数学上保证
> 3. **K Wan frames as conditions, not observations**: 严格区分 self-generated pseudo-state conditions 与 multi-state ground-truth observations
> 4. **One canonical SLAT, not K state-specific**: 物理一致的 URDF 必须有单一 base/move texture

---

## 1. 问题界定 & 约束

### 1.1 输入

- **`I₀`**：单张闭态 (rest-state) RGB 图
- **prompt 文本** (locked-camera template)
- **joint type 先验** (prismatic / revolute) — MVP 由用户/prompt 指定，paper version 可学习

### 1.2 输出

- `canonical_base.glb` — 静态 base mesh (with texture)
- `canonical_move.glb` — movable part mesh (in canonical pose, with texture)
- `joint.json` — single-DoF joint params (axis, pivot/origin, range)
- `object.urdf` — physically simulatable
- `optimized_video.mp4` — locked-camera articulation rendering
- `texture_provenance.json` — 各表面纹理的来源标注

### 1.3 范围与约束

- **单关节、single-DoF**（prismatic 或 revolute），多关节扩展进 future work
- **TRELLIS 全冻结，Wan2.2 全冻结**
- **每物体 test-time optimization**（~2000 iter）
- **不修改 TRELLIS 原 attention 权重**（只插 RASA residual）
- **K=6 pseudo-state conditions, NOT observations** — 论文严格区分
- **Hardware**: NVIDIA H800 80GB，**质量优先**（不做 24GB consumer GPU 妥协）
  - Wan2.2 I2V-A14B (full strength), full 480p resolution, K=6 frames, 25-step Euler
  - 设计目的不是 fit consumer GPU；是为了 first-principle 正确性

### 1.4 与 Stage B (legacy) 的关系

**Stage B (v4.3 BMCSA, 当前 `pipelines/stage_b_scar.py`)** 是我们的**第一阶段尝试**。它通过：

```
Wan2.2 video (=Stage A) → I_0..I_5 → K-batched SS-DiT sampling
  + SCAR symmetric mix (0.3·s₀ + 0.4·s_k + 0.3·s₅)
  + SDEdit Pass 2 with BMCSA (cross-state attention)
  + ICP alignment for residual drift
→ K aligned 64³ occupancies
```

试图保持 base 跨 K 状态一致。**实测显示 voxel-level base jitter (~±2 voxels) 不可避免**——这是 K 次独立 SS 采样 + `argwhere(>0)` 离散阈值的固有产物，soft cross-state attention 能 reduce 但不能 eliminate。

**v5 完全摒弃 K-batched SS 采样范式**：单一 canonical SS + 刚体 warp 派生 K 状态。

**v5 中的 Stage B**：
- ✅ **Wan2.2 K-frame generation** (= 现 Stage A) **保留**，作为 v5 Phase 0 (`pipelines/oad/pseudo_state/wan22_generator.py`)
- ✅ **DiT hidden hook 机制** (`stage_b_scar.py:capture_dit_hidden_states`) **复用** ~30 行迁移到 v5 SS-RASA wrapper
- ✅ **DINOv2 batched encoding** for K images **复用**
- ❌ **SCAR symmetric mix / BMCSA / SDEdit Pass 2 / ICP alignment / M_attn / joint_free_split preview** 全部**废止**
- 📋 Stage B 完整 pipeline 保留为 **ablation baseline**（论文 Table 中作为 "K-batched + alignment" 对比组）+ **motivational failure case**

**论文叙事**：

> "Our previous attempt (Stage B, supplementary) used K-batched SS sampling
> with cross-state attention alignment (BMCSA). Despite explicit alignment,
> voxel-level base jitter (~±2 voxels) persisted because K independent
> samplings + hard `argwhere(>0)` thresholding inevitably amplify any
> continuous-logit difference into discrete topology change. This motivates
> our v5 single-canonical paradigm: avoid K independent samplings entirely
> via canonical SS + deterministic single-DoF rigid transform derivation."

**代码组织建议**：
```
pipelines/
  stage_b_scar.py               # legacy, marked deprecated, ablation only
  oad/                           # ★ v5 主线 (NEW)
    __init__.py
    pseudo_state/                # 复用 Stage A 的 Wan2.2 generation
    ss_rasa/                     # 新写, 复用 hook 机制
    ...
```

---

## 2. 核心架构洞察

### 2.1 Stage B 失败的根因

我们之前的 Stage B 用 K=6 batched SS-DiT sampling + symmetric mix (`0.3·s₀ + 0.4·s_k + 0.3·s₅`) + BMCSA cross-state attention，结果：
- base 大致对齐 ✓
- 但 voxel 级 jitter 仍存在 ✗

**根因**：K 个独立采样**不可避免**产生 ±1-2 voxel boundary jitter。`argwhere(decoder(z_s) > 0)` 这一 hard threshold 把连续 logit 的微小差异放大成 voxel topology 差异。Cross-state attention 软对齐能 reduce jitter 但不能 **eliminate** 之。

### 2.2 解决方案：reverse the paradigm

**不再让 K 状态各自采样后对齐——只采一次 canonical，K 状态由 rigid transform 确定性派生**：

```
B = v* ∩ m_base       (canonical base voxels)
M = v* ∩ m_move       (canonical move voxels in state 0 frame)
v_k = B ∪ T(q_k)·M    (state k geometry)
```

由于 K 状态的 base 是 **同一组 voxel**（不是 K 次采样的结果），bit-identical by construction。**零 jitter 可能**。

### 2.3 K 张 Wan 帧的角色重定位

Wan 抽帧 K=6 不再驱动 K 次 SS-DiT 采样，而是作为 **RASA 层的 cross-condition signals**：
- RASA 在 SS-DiT 内部 cross-attend K 个 cond，从 K 帧间 appearance 差异学 base/move 区分
- RASA 在 SLAT-DiT 内部用 MorphAny3D K-way MCA/TFSA 融合 K 个 donor texture

**K 帧依旧是条件**（满足导师要求），但**不触发 K 次 SS 采样**（消除 jitter 来源）。

### 2.4 SLAT 也必须是单一 canonical

物理上一个 articulated asset 只有 **1 套** base texture + **1 套** move texture。如果保留 K 套 state-specific SLAT，导出 URDF 时仿真器会出现 state-dependent appearance（不物理）。

所以 **SLAT-DiT 跑 K 次但仅作 donor 计算，最终融合成 1 套 canonical SLAT**。渲染 K 状态时只对 mesh 做刚体 warp，**纹理不变**。

---

## 3. Pipeline Overview

```
INPUT: I₀ + locked-camera prompt + joint_type prior
   │
   ↓ Phase A (一次性, no_grad)
   │
   Wan2.2 I2V (frozen) → opening video → K=6 frames I₀..I₅
   DINOv2 batched → cond_K ∈ R^{K×1374×1024}
   Build SUPPORT_COORDS₀ (loose threshold, dilation, top-K uncertain)
   │
   ↓ Phase B (per-iteration, training loop, 2000 iters)
   │
   ┌─ B1. Single canonical SS forward (in-loop, gradient ON for RASA) ─┐
   │   Frozen SS-DiT 25-step Euler:                                   │
   │   - Primary cond: I₀ (canonical anchor)                          │
   │   - SS-RASA at blocks {16, 19, 22} reads ALL K conds via         │
   │     cross-condition attention                                    │
   │   - Output: canonical SS latent z*                               │
   │   - SS-VAE decode → canonical occupancy v* (64³ logits)          │
   └──────────────────────────────────────────────────────────────────┘
   │
   ┌─ B2. PartHead + JointHead (gradient ON) ──────────────────────────┐
   │   Dual PartHead:                                                 │
   │   - 16³ semantic head: SS-DiT multi-tap features                 │
   │   - 64³ boundary head: SS-VAE decoder intermediate features      │
   │   - Fusion → m_base, m_move, m_uncertain (∈ [0,1]³, sum=1)      │
   │                                                                   │
   │   JointHead: pool(features, m_move) → (axis, pivot, q_k)         │
   └──────────────────────────────────────────────────────────────────┘
   │
   ┌─ B3. Deterministic K-state geometry (no new sampling) ────────────┐
   │   Soft (training):                                                │
   │     v_k_soft(x) = m_base(x)·v*(x) + m_move(x)·v*(T_k⁻¹·x)        │
   │                                                                   │
   │   Hard (export):                                                  │
   │     v_k_hard = (v* ∧ m_base>τ) ∪ T_k(v* ∧ m_move>τ)              │
   │                                                                   │
   │   Fixed-support gate: g_i = σ(SS_decoder_logit_i / temp)         │
   │   gates SLAT features / Gaussian opacity / SDF                    │
   └──────────────────────────────────────────────────────────────────┘
   │
   ┌─ B4. K-donor SLAT-DiT branches → ONE canonical SLAT (no K outputs!) ┐
   │   For k=0..K-1: run SLAT-DiT (frozen) with cond=I_k, coords=v_k    │
   │     ★ K parallel runs are donor computations, not K state outputs │
   │                                                                    │
   │   SLAT-RASA at blocks {16, 19, 22}:                               │
   │     - K-way MCA: per-donor cross-attn → post-attention fusion     │
   │     - K-way TFSA: cached K/V donor blending                       │
   │     - Region-aware donor weights (visibility, part_type, conf)    │
   │                                                                    │
   │   Output: ONE canonical SLAT s* = {s*_base, s*_move}              │
   │           (NOT K state-specific SLAT)                             │
   └──────────────────────────────────────────────────────────────────┘
   │
   ┌─ B5. Render + Wan2.2 W-RFSDS ─────────────────────────────────────┐
   │   For k=0..K-1:                                                  │
   │     mesh_k = SLAT_VAE.decode_mesh(s*, coords=v_k)                │
   │     # mesh_k 用 canonical s* 但在 v_k coords 上 decode            │
   │     # rigid warp: base 同位置, move 在 T_k 位置                   │
   │   V_render = nvdiffrast(mesh_0..K-1, locked_camera) (9 frames)   │
   │                                                                   │
   │   z = wan_vae.encode(V_render)        # autograd ON wrt V         │
   │   τ = inverse_cdf_w_hat(1 - iter/2001) # high→low schedule       │
   │   ε ~ N(0, I)                                                    │
   │   z_τ = (1-τ)·z.detach() + τ·ε                                   │
   │   with no_grad:                                                  │
   │     v_hat = CFG(Wan22_DiT, z_τ, τ, image=I₀, prompt, s=25→12)    │
   │   grad = (v_hat - ε + z.detach())     # CHORD-style stop-grad    │
   │   z.backward(gradient=grad)                                      │
   └──────────────────────────────────────────────────────────────────┘
   │
   ↓ Phase C (export, 一次性, no_grad)
   │
   Hard threshold m_base/m_move
   Decode mesh + texture from canonical SLAT
   Write joint.json, URDF, simulation diagnostics
```

---

## 4. 模块详细设计

### 4.1 Phase A — Wan2.2 K Pseudo-State Generation

**输入**：I₀ + prompt
**Wan2.2 模型**：I2V-A14B (80GB GPU) / TI2V-5B (24GB PoC)

**Locked-camera prompt template**（必须严格使用）：

```
Positive: "A locked-off single-camera video. The first frame is exactly the
provided image. The camera is static, with no zoom, pan, or orbit. Only the
[movable_part_name] [slides/rotates] open. The object body remains stationary.
Material, color, and lighting remain consistent across all frames."

Negative: "camera movement, zoom, pan, orbit, object morphing, identity drift,
background change, texture flicker, non-rigid deformation, floating parts"
```

**采样**：~9 frames at 256×256（PoC）/ 41 frames at 480×832（full）
**抽帧**：均匀 K=6 帧 `I₀, I₁, ..., I₅`，I₀ 强制等于输入图

**预处理**：
- DINOv2 batched encode → `cond_K ∈ R^{K, 1374, 1024}`
- Frame 0 标记为 hard identity anchor (confidence=1.0)
- Frame 1..5 confidence 可学习（但有上限避免过度信任）

**支持集初始化** (Phase A 唯一一次)：

```python
with torch.no_grad():
    z_init = TRELLIS_SS_DiT(noise, cond=I₀)        # quick canonical sample
    occ_init = SS_VAE_decoder(z_init)              # 64³ logits
    SUPPORT_COORDS = (
        (occ_init > 0).nonzero()                   # standard occupied
        | dilation((occ_init > 0), radius=1)       # boundary safety
        | top_k_uncertain(occ_init, K_extra=2000)  # boundary candidates
        | motion_sweep(joint_type, axis_init)      # rough motion zone
    )
```

`SUPPORT_COORDS` 在所有 2000 iter 中 **保持固定**，以避免 argwhere 离散断点。

### 4.2 Phase B1 — Single-Canonical SS-RASA (per-iteration)

**关键决定**：**每个优化 iter 跑一次完整 SS-DiT forward** (frozen weights, 25-step Euler)，让 RASA 在计算图里。

#### 4.2.1 SS-RASA 模块结构

插入位置：`pipelines/stage_c_segmatch/...` → 新建 `pipelines/oad/ss_rasa.py`

插入到 SS-DiT 第 16, 19, 22 个 block（共 3 处）的 `block_output_residual` 处（self-attn 后、cross-attn 前那条 path）。

**步骤 1 — State-aware token augmentation**:
```python
H̃_l = H_l + α_state · E_state(s_l)  # 不进 frozen TRELLIS, 只在 RASA 内部
                                       # E_state 是 sinusoidal embedding
```

**步骤 2 — Cross-condition attention** (K 张图作为 condition 信号)：
```python
# H_l ∈ R^{L=4096, C=1024}, cond_K ∈ R^{K=6, 1374, 1024}
# K-way cross-attention: each voxel queries each frame, frame-confidence weighted
H_cross_K = []
for k in 0..K-1:
    H_cross_k = CrossAttn(Q=H̃_l, KV=cond_K[k])         # (L, C)
    H_cross_K.append(c_k · H_cross_k)                    # confidence-weighted
H_cross = sum(H_cross_K) / sum(c_k)                      # (L, C)
```

**步骤 3 — Part-slot bottleneck**:
```python
# 4 learnable slots (base / move / uncertain / interior)
slot_queries = nn.Parameter(torch.zeros(4, C))           # learnable
S = CrossAttn(Q=slot_queries, KV=H_cross)                # (4, C) part evidence
ΔH_l = CrossAttn(Q=H_cross, KV=S)                        # (L, C) scatter back
```

**步骤 4 — Region-gated zero-init residual**:
```python
# m̂ 是 PartHead 在前一 forward 给出的 16³ part probability
# (training 中 m̂ 来自当前 forward, 通过 detach + EMA 避免循环)
m̂_l = downsample_to_16³(m_64_prev_iter)                  # detached, 16³

ΔH_gated = m̂_move · ΔH_move + m̂_uncertain · ΔH_uncertain
           + m̂_base · 0                                  # base region: no residual
                                                          # 强制 base 接近原 TRELLIS 输出

H'_l = H_l + α_l · ΔH_gated                              # α_l zero-init
```

**Trainable params**:
- `α_state`, `slot_queries` (4×C)
- `α_l` per block (3 scalars, init 0)
- `c_k` frame confidences (5, K-1, init from prior)
- 3 sets of cross-attn weights (Q/K/V/O) × 3 blocks ≈ **3M params**
- `ΔH_move`, `ΔH_uncertain` MLPs ≈ **0.5M**
- **Total SS-RASA: ~3.5M**

**关键不变量**：iter=0 时 α_l=0，整个 SS-RASA 输出 ΔH=0，**SS-DiT forward 等价于 frozen TRELLIS**（保证基线可复现）。

### 4.3 Phase B2 — Dual PartHead + JointHead

#### 4.3.1 Dual PartHead

**16³ Semantic branch**:
```python
# 多 block tap (12, 16, 20) feature concat → MLP
F_16 = concat(H_12, H_16, H_20)           # (4096, 3072)
m_sem_16 = MLP_sem(F_16).reshape(16³, 4)   # (16, 16, 16, 4) base/move/uncertain/interior
```

**64³ Boundary branch**:
```python
# SS-VAE decoder 中间特征 (32³ stage) → 64³
F_dec_32 = SS_VAE_decoder.intermediate_32(z*)  # (1, 128, 32, 32, 32)
F_dec_64 = upsample(F_dec_32, 64)               # (1, 32, 64, 64, 64)
m_bnd_64 = ConvHead(F_dec_64)                   # (4, 64, 64, 64)
```

**Fusion**:
```python
m_logit_64 = upsample_trilinear(m_sem_16, 64) + γ · m_bnd_64
m_64 = softmax(m_logit_64 / T, dim=channel)     # (4, 64³): base/move/uncertain/interior
```

#### 4.3.2 JointHead

```python
# Pool RASA features by m_move
F_move = (m_move_16.unsqueeze(-1) · H_16).sum(dim=L) / m_move_16.sum().clamp(eps)

# Joint type (MVP: from prompt; paper version: learnable softmax)
joint_type = "revolute" or "prismatic"  # given by prompt

# Axis: S² classification + residual refinement (PARIS-style)
axis_logits = MLP_axis_cls(F_move)              # (162,) icosahedral S² grid
axis_bin = argmax(axis_logits)
axis_residual = MLP_axis_refine(F_move)         # (3,) within-bin offset
axis = normalize(S²_grid[axis_bin] + axis_residual)

# Pivot (revolute only)
pivot = MLP_pivot(F_move)                        # (3,)

# Per-state q_k
q_k_raw = MLP_q(F_move)                          # (K-1,)
q_k = monotonic_increasing(softplus(q_k_raw))    # ensure q monotonic
q_full = [0, q_k_raw[0], q_k_raw[0]+q_k_raw[1], ...]  # cumulative
```

**Trainable params**: PartHead ~0.6M, JointHead ~0.4M. **Total: ~1M**.

### 4.4 Phase B3 — Deterministic K-State Geometry

#### 4.4.1 Single-DoF transform `T(q_k)`

**Prismatic**: `T_k(x) = x + q_k · axis`
**Revolute**: `T_k(x) = R(axis, q_k)·(x - pivot) + pivot`，Rodrigues:
```
R(a, θ) = I + sin(θ)·[a]× + (1-cos(θ))·[a]×²
```

#### 4.4.2 Soft (training) — Gradient flows through warp

```python
# canonical occupancy v*(x) ∈ [0, 1] (soft)
# Per-state occupancy via probabilistic noisy-OR
P_base(x) = m_base(x) · v*(x)
P_move(x) = m_move(x) · v*(x)

# State k via differentiable trilinear warp
P_move_warped_k(x) = trilinear_sample(P_move, T_k⁻¹(x))    # grid_sample

# Composition (noisy-OR: union of base and warped move)
P_k(x) = 1 - (1 - P_base(x)) · (1 - P_move_warped_k(x))
```

**重要**: 这一步 **只用 v*** （单一 canonical），不引入新 SS 采样。

#### 4.4.3 Hard (export)

```python
B_hard = (v* > 0.5) & (m_base > τ_base)
M_hard = (v* > 0.5) & (m_move > τ_move)
for k in 0..K-1:
    v_k = B_hard ∪ apply_rigid_transform(M_hard, T_k)
```

### 4.5 Phase B4 — Canonical SLAT-RASA (K-donor → ONE canonical)

**关键设计 (Option B)**：所有 K 个 SLAT-DiT donor branches **共享同一组 canonical coordinates**，区别只在 image conditioning (I_k 不同)。融合输出**单一 canonical SLAT**，不保留 K 套 state-specific texture。

#### 4.5.1 Shared canonical coords (Option B — 替代旧 Option A 的 inverse warp)

```python
# ★ 所有 K donor branches 用同一组 canonical coords
# 原因 (1): Token-wise correspondence by construction，不需 inverse warp
# 原因 (2): MorphAny3D 实证 cond ≠ structure 的设置 work
#         (MorphAny3D 自己就是这么做的: target shape canonical coords + src/tar image conds,
#          见 paper/MorphAny3D/trellis/modules/sparse/transformer/modulated.py:146-180)

canonical_coords = active(v* AND (m_base ∪ m_move))    # state 0 frame, fixed across K

# K parallel donor branches, all on SAME coords:
for k in 0..K-1:
    s_k_donor = SLAT_DiT(noise', cond=I_k, coords=canonical_coords)
    # ★ 所有 s_k_donor token 一一对应 canonical_coords[i]
    # SLAT-DiT 的 cross-attn 看 I_k 给出 "如果 canonical 长 I_k 那样的 texture"
```

**对照 MorphAny3D**：他们的 MCA 也是同一组 canonical coords + 两组 image conds（src, tar），不做 inverse warp。我们 K-way 推广只是 K 张图代替 2 张。

#### 4.5.2 SLAT-RASA — K-way MCA / TFSA (token-wise fusion, no inverse warp)

插入位置：每个 SLAT-DiT block 的 `cross_attn` 和 `self_attn` (blocks 16, 19, 22)
代码参考：`paper/MorphAny3D/trellis/modules/sparse/transformer/modulated.py:146-180`

**K-way MCA** (post-attention output fusion, NOT pre-attn KV fusion):

```python
# 每个 donor cond I_k 独立做 cross-attn, 然后 token-wise weighted blend
# 因为所有 donor 共享 canonical coords，token i 在所有 donor 中是同一位置
for j in 0..K-1:
    o_cross_ij = CrossAttn(Q=h_i, KV=encode(I_j))      # cross-attn output for donor j

h_MCA_i = sum(w_ij · o_cross_ij for j in 0..K-1)        # token-wise blend
```

**K-way TFSA** (cached K/V from K donor self-attn outputs):

```python
# 对每个 token i 选 top-r donors (r=2 or 3) 减少噪声
for j in top_r_donors(i):
    o_self_ij = SelfAttn(Q=h_i, KV=cached_KV_j)        # using donor j's self-attn K/V

h_TFSA_i = sum(w̃_ij · o_self_ij for j in top_r_donors(i))
```

**关键简化**：因为所有 donor 在 **同一组 canonical_coords**，token i 在每个 donor 中是同一位置，可以**直接 token-wise blend**，不需要 inverse warp / trilinear interp / nearest-neighbor matching。这是相对 v5 之前 Option A 的关键修正——避免插值误差，避免 missing voxel 问题。

#### 4.5.3 Donor weights (region-aware)

```python
# Per token i, per donor frame j:
w_ij = softmax_j(λ_v · visibility_ij           # is voxel i visible in frame j?
              + λ_c · cam_facing_ij            # surface normal · view dir > 0
              + λ_p · part_compat_ij           # part-prob class agrees with frame j
              + λ_f · frame_confidence_j       # I₀ hard 1.0, I_1..5 learned
              + λ_t · temporal_proximity)      # for TFSA: cross-state distance
```

**Region-aware behavior**:
- **Base tokens** (m_base[i] > τ): w 偏向 I₀ + 跨状态一致 donors（base 不变, 应该是 state-invariant）
- **Move-exterior tokens** (m_move[i] > τ, visible in early states): w 偏向闭态或部分开态
- **Newly-exposed interior tokens** (m_move[i] > τ, only visible in late states): w 偏向开态 donors
- **Uncertain tokens** (m_uncertain[i] > τ): low overall weight, 更多依赖 TRELLIS prior

**输出**: `s* = {s*_canonical[i]}_{i ∈ canonical_coords}`，单一 canonical SLAT。**没有 K 套 state-specific SLAT。**

#### 4.5.4 Texture Adapter (AdaptFormer parallel bottleneck)

仅作用在 SLAT 的 color slice (last 48 channels)，base/move 路由不同 delta：
```python
# canonical SLAT s* has 8 channels; SLAT_VAE color decoder maps to 48-d color
ΔC_base = AdaptFormer_base(F_color_features)    # bottleneck 8→32→48
ΔC_move = AdaptFormer_move(F_color_features)
ΔC = m_base · ΔC_base + m_move · ΔC_move
C_final = C_TRELLIS + α_color · ΔC               # α_color zero-init
```

**Trainable params (SLAT-RASA)**:
- 3 K-way MCA (per block) ≈ 3M
- 3 K-way TFSA ≈ 3M
- AdaptFormer (base + move) ≈ 0.5M
- Donor weight MLP ≈ 0.1M
- **Total SLAT-RASA: ~6.6M**

### 4.6 Phase B5 — Render + Wan2.2 W-RFSDS

#### 4.6.1 训练阶段：Gaussian (smooth gradient) ; 导出阶段：FlexiCubes mesh

**关键设计决定**：训练循环 (Phase B, 2000 iter) 使用 **Gaussian Splatting decoder** 渲染；mesh decoding 只在 Phase C export 时执行一次。

**理由（梯度稳定性，不是显存）**：
- FlexiCubes 在 SDF=0 切换处面拓扑不可微（measure-zero non-differentiable set）
- 2000 iter 反复迭代时，topology change 会导致 gradient discontinuity / sparse gradient on thin structures
- CHORD 同样选 3D Gaussian Splatting 训练 → mesh post-extract，因为 smooth gradient 更稳定
- TRELLIS SLAT-VAE 原生支持 mesh / Gaussian / Radiance Field 三种 decoder，切换是合理且零成本的

**Phase B (training) 渲染流程**:
```python
# K-state Gaussian rendering, using SAME canonical SLAT s*
for k in 0..K-1:
    # Build per-state coord set with rigid transform of move
    coords_k_base = canonical_base_coords            # same across k
    coords_k_move = T_k(canonical_move_coords)       # rigid warped
    coords_k = coords_k_base ∪ coords_k_move
    
    # Build per-state SLAT features:
    # - base voxels at unchanged position: copy s*_base directly
    # - move voxel at T_k(p) (world position): use s*_move[p] (canonical position p)
    s_k_for_render = build_state_k_slat(s*, T_k, coords_k)
    
    # ★ Gaussian decoder (frozen weights, autograd ON for activations)
    gaussians_k = SLAT_VAE.decode_gaussian(s_k_for_render, coords_k)
    # gaussians_k: per-Gaussian (xyz, rotation, scale, opacity, sh_color)
    # SAME canonical color across K (only xyz changes per state)

# Differentiable Gaussian splatting (smooth, stable gradients)
V_render = gaussian_splat_renderer(
    gaussians_0..K-1, locked_camera,
    image_size=(H, W),  # Wan2.2 input resolution
)
# Output: (T_frames, H, W, 3), padded to 4n+1 for Wan2.2 (K=6 → 9 frames)
```

**Phase C (export only) 切换到 mesh**:
```python
# Once per asset, after 2000 training iters complete
mesh_canonical_base = SLAT_VAE.decode_mesh(s*_base, canonical_base_coords)  # FlexiCubes
mesh_canonical_move = SLAT_VAE.decode_mesh(s*_move, canonical_move_coords)  # FlexiCubes
# Bake textures, write GLB / URDF
```

#### 4.6.2 物理一致保证

- **K state 几何**：base 区域跨 K identical（同一 voxel 集合）；move 区域只刚体变换
- **K state 纹理**：来自单一 `s*`，**纹理跨 K 完全一致**（base/move texture 都是 state-invariant）
- **导出 URDF 后仿真器看到的 articulated asset**: 一套 base 几何 + 一套 base 纹理 + 一套 move 几何 + 一套 move 纹理 + 一个 joint。物理上 valid。

#### 4.6.3 Wan2.2 W-RFSDS (CHORD style)

```python
z = wan_vae.encode(V_render)               # autograd ON wrt V_render
                                            # frozen weights, but grad flows through

# Schedule: high τ early (geometry/motion), low τ late (texture/detail)
τ = inverse_cdf_w_hat(1 - iter/2001)        # high→low annealing
ε = randn_like(z)
z_τ = (1 - τ) · z.detach() + τ · ε

with torch.no_grad():
    cfg_scale = 25 - 13 · (iter / 2000)     # 25 → 12 linear decay
    v_uncond = wan_dit(z_τ, τ, image_cond=I₀, text_cond=null)
    v_cond   = wan_dit(z_τ, τ, image_cond=I₀, text_cond=prompt)
    v_hat    = v_uncond + cfg_scale · (v_cond - v_uncond)

grad_z = (v_hat - ε + z.detach())           # CHORD Eq. 16, stop-grad on v_hat
loss_rfsds = (z * grad_z.detach()).sum()    # surrogate loss
loss_rfsds.backward()
```

**Backward 路径**：
```
z (Wan latent)
  ← ∂z/∂V (Wan VAE encoder, frozen weights, autograd ON)
  ← ∂V/∂mesh (nvdiffrast)
  ← ∂mesh/∂(s*, coords_k) (SLAT_VAE decoder, frozen weights, autograd ON)
  ← ∂s*/∂SLAT-RASA (我们插的层，trainable)
  ← ∂coords_k/∂T_k (deterministic warp from joint params)
  ← ∂T_k/∂(JointHead axis/pivot/q_k) (trainable)
  ← ∂(part mask, joint params)/∂(SS-RASA, PartHead, JointHead) (trainable)
```

所有 frozen TRELLIS 和 Wan2.2 的 weights 不更新；梯度流过它们但不更新它们。

---

## 5. Loss Functions

```
L_total = λ_rfsds · L_W-RFSDS               # 主监督
        + λ_first · L_first_frame_anchor     # I₀ identity 锚定
        + λ_base  · L_base_static            # base 在 K 状态间 photometric 一致
        + λ_part  · L_part_regularization    # 防止 move 吞物体或为空
        + λ_joint · L_joint_prior            # axis 单位、q 单调、范围合理
        + λ_supp  · L_support_regularization # gate 稀疏 + margin
        + λ_tex   · L_texture_anchor         # state-0 texture 强锚定
        + λ_cans  · L_canonical_consistency  # K donor 蒸馏成 canonical 时一致性
        + λ_reg   · L_adapter_reg            # RASA residual 范数 / α 衰减
```

### 5.1 关键 loss 详解

**L_first_frame_anchor** (重点，防止 identity drift)：
```
L_first = ‖V_render[k=0] − I₀‖₁ + λ_LPIPS · LPIPS(V₀, I₀)
        + λ_DINO · ‖DINO(V₀) − DINO(I₀)‖₂² + λ_mask · L_mask(V₀, I₀)
```

**L_base_static** (通过 architecture 而非 loss 主要保证；这条是 backup)：
```
L_base = Σ_k ‖render(B, view_k) − render(B, view_0)‖₁
       (since base 在 K 状态共享 same canonical voxels, this should be ~0 by construction;
        but include as a check / soft regularizer)
```

**L_part_regularization**:
```
L_overlap = E[m_base · m_move]                       # base 和 move 不重叠
L_size = (m_move_volume / total_volume - r₀)²        # move 体积合理 (r₀ ≈ 0.1-0.4)
L_TV = total variation on m_64                       # 平滑 part 边界
L_entropy = annealing entropy of part_probs          # 早期软探索, 晚期硬决断
```

**L_joint_prior**:
```
L_axis_unit = (‖axis‖ - 1)²                                  # unit norm
L_monotonic = Σ_k max(0, q_k - q_{k+1})²                     # q 单调
L_range = max(0, q_max - q_limit_physical)²                  # range 合理
L_rigid = Σ_{i,j ∈ M, k} |‖T_k(x_i) - T_k(x_j)‖ - ‖x_i - x_j‖|  # 刚性
```

**L_canonical_consistency** (新增，关键):
```
# 强制 K donor 蒸馏到 canonical 后一致
L_cans = Σ_k ‖inverse_warp(s_k_donor, T_k) - s_canonical‖²
       (donor 反 warp 到 canonical frame 应该跟 fused canonical 一致)
```

### 5.2 Confidence regularization (Wan frame trustworthiness)

```
# c_k ∈ [0, 1] 是 frame confidence
# c_0 = 1.0 hard-coded (identity anchor)
# c_1..c_5 learnable but bounded
L_conf = -λ_conf · Σ_k log(c_k + ε) - λ_drift · Σ_{k>0} c_k · ‖I_k - warp(I_0, T_k)‖
       # 第一项: 鼓励合理高 confidence
       # 第二项: 如果 I_k 与"假设 I_0 经 T_k warp 而成的图"差太多, 降低 confidence
```

---

## 6. Optimization Schedule

### 6.1 Phase A — Init (一次性, no_grad)

```
- Wan2.2 I2V → K=6 frames
- DINOv2 batched
- TRELLIS SS-DiT zero-RASA forward → SUPPORT_COORDS
- Init JointHead: axis from optical flow on K frames, q_k linear init
- Init RASA α_l = 0
```

### 6.2 Phase B — In-loop optimization (2000 iter)

| Sub-phase | Iters | τ schedule | 优化 | Frozen |
|---|---|---|---|---|
| **B-warmup** | 0-100 | high τ | RASA α_l warmup (0→0.1) | texture, JointHead axis |
| **B-geometry** | 100-1200 | high-mid τ | SS-RASA, PartHead, JointHead, support gate | SLAT-RASA, texture |
| **B-boundary** | 1200-1500 | mid-low τ | PartHead boundary, JointHead refine | base region |
| **B-texture** | 1500-2000 | low τ | SLAT-RASA, texture adapter, donor weights | geometry, joint params (low LR) |

### 6.3 Phase C — Export (一次性, no_grad)

```
- Hard threshold m_base, m_move
- Decode canonical_base mesh (with texture from s*_base)
- Decode canonical_move mesh (with texture from s*_move)
- joint.json = {type, axis, pivot/origin, range=[0, max(q_k)], state_values=q_k}
- Write URDF
- Simulate in PyBullet for sanity
```

---

## 7. Differentiability Audit

| 组件 | Frozen weights | Autograd ON for activations | Trainable params |
|---|---|---|---|
| DINOv2 | ✓ | ✓ | × |
| SS-DiT 25-step Euler | ✓ | ✓ (gradient checkpointed per Euler step) | SS-RASA |
| SS-VAE decoder | ✓ | ✓ | PartHead boundary tap |
| PartHead | × | ✓ | All (~0.6M) |
| JointHead | × | ✓ | All (~0.4M) |
| K-state warp `T_k(·)` | (analytical) | ✓ via grid_sample | implicit via JointHead |
| SLAT-DiT × K branches (canonical coords) | ✓ | ✓ (checkpointed) | SLAT-RASA |
| **SLAT-VAE Gaussian decoder** (training) | ✓ | ✓ (fully differentiable, smooth) | Texture adapter on color slice |
| SLAT-VAE Mesh decoder (FlexiCubes) | ✓ | ✓ (vertex pos diff, face topology measure-zero non-diff) | × — **export only** |
| Gaussian splatting renderer (training) | ✓ | ✓ | × |
| nvdiffrast render (export viz) | ✓ | ✓ | × — **export only** |
| Wan2.2 VAE encode | ✓ | **✓ (CRITICAL: gradient ON wrt V_render)** | × |
| Wan2.2 DiT velocity | ✓ | **× (no_grad teacher forward)** | × |
| RFSDS gradient | (analytical) | injected via stopgrad-of-(v̂-ε+z) | × |

**Hard breakpoint**: `argwhere(decoder(z_s) > 0)` at `pipelines/trellis_image_to_3d.py:191`. **Bypass**: 用 fixed `SUPPORT_COORDS` (Phase A 一次性算) + per-iter differentiable gate `g_i = σ(SS_decoder_logit_i / temp)`。

**关键设计选择 (gradient stability, not memory)**:
- **训练阶段** Phase B 用 **Gaussian decoder** (smooth, fully differentiable)
- **导出阶段** Phase C 才用 **mesh decoder** (FlexiCubes face-topology event 只在 final extraction 发生一次)
- 这跟 CHORD 用 4D-GS 训练的逻辑一致

**Per-Euler-step gradient checkpointing**: SS-DiT 25 步 Euler 全展开，per-step checkpoint
- Forward: 仅 cache 25 个 Euler step input z_t (各 ~50 MB) ≈ 1.25 GB activation overhead
- Backward: 每步重算 24-block forward → 总 compute = forward + recompute + backward ≈ 2.5× 单 forward 成本
- 这是 SDS-style optimization 的标准做法

---

## 8. Implementation Plan

### 8.1 文件结构

```
pipelines/oad/
├── __init__.py                        # API
├── pipeline.py                        # 主驱动 (run_oad)
├── config.py                          # OadHParams
│
├── pseudo_state/
│   ├── wan22_generator.py             # Phase A: Wan2.2 I2V K-frame gen
│   └── dinov2_batched.py              # K-frame DINOv2 encoding
│
├── ss_rasa/
│   ├── ss_rasa_module.py              # SS-RASA 主模块
│   ├── cross_condition_attn.py        # K-cond cross-attn
│   ├── part_slot_bottleneck.py        # 4-slot bottleneck
│   └── ss_dit_wrapper.py              # frozen SS-DiT + RASA hooks
│
├── heads/
│   ├── part_head.py                   # 16³ semantic + 64³ boundary dual
│   ├── joint_head.py                  # S² classification + refine + q_k
│   └── support_gate.py                # fixed-support differentiable gate
│
├── articulation/
│   ├── transform.py                   # T_k(prismatic) / T_k(revolute) Rodrigues
│   ├── kstate_geometry.py             # B ∪ T_k(M) soft / hard
│   └── warp.py                        # differentiable trilinear warp
│
├── slat_rasa/
│   ├── slat_rasa_module.py            # SLAT-RASA 主模块
│   ├── kway_mca.py                    # K-way Morphing Cross-Attn
│   ├── kway_tfsa.py                   # K-way Temporal-Feature Self-Attn
│   ├── donor_weights.py               # visibility + part_compat + conf
│   ├── canonicalization.py            # K → 1 canonical SLAT
│   └── texture_adapter.py             # AdaptFormer for color slice
│
├── render/
│   └── kstate_renderer.py             # nvdiffrast K-frame video
│
├── rfsds/
│   ├── wan22_rfsds.py                 # W-RFSDS surrogate loss (CHORD style)
│   └── tau_schedule.py                # inverse-CDF annealing
│
├── losses/
│   └── articulation_losses.py         # all loss terms
│
└── export/
    ├── threshold_and_decode.py        # canonical_base/move mesh decode
    └── urdf_writer.py                 # joint.json + URDF

configs/oad_v5.yaml                    # full hyperparameters
record/articulated_2026-04-26_final_scheme.md   # ← 此文档

scripts/
├── run_oad.py                         # CLI 主入口
├── eval_oad.py                        # GT 对比评估
├── viz_oad.py                         # 可视化输出
└── ablation/                          # ablation 脚本
```

### 8.2 Milestone

| Milestone | 内容 | 预计时间 |
|---|---|---|
| **M0** | Phase A pipeline (Wan2.2 → K frames + DINOv2 cond) | 3 天 |
| **M1** | SS-RASA + PartHead + JointHead 起步 (zero-init verify) | 5 天 |
| **M2** | Phase B3 deterministic K-geometry + soft warp | 3 天 |
| **M3** | Render + Wan2.2 W-RFSDS gradient sanity test | 4 天 |
| **M4** | First end-to-end PoC on 1 drawer (PartNet 30857) | 5 天 |
| **M5** | SLAT-RASA + canonical SLAT fusion | 7 天 |
| **M6** | Texture phase + 4-sample test | 5 天 |
| **M7** | Full ablation matrix on 30857/7201/7128/26525 | 7 天 |
| **M8** | 50-sample benchmark + paper writing | 14 天 |
| **Total** | | ~7 周 |

### 8.3 PoC 优先级（M0-M4）

```python
# Sanity test 1: zero-init RASA produces frozen TRELLIS output
def test_zero_init():
    out_with_rasa = SS_DiT_with_RASA(noise, cond_K, α_l=0)
    out_frozen = SS_DiT(noise, cond=I_0)
    assert torch.allclose(out_with_rasa, out_frozen, atol=1e-5)

# Sanity test 2: gradient reaches RASA params
def test_gradient_flow():
    loss.backward()
    for param in ss_rasa.parameters():
        assert param.grad is not None and param.grad.abs().mean() > 0

# Sanity test 3: wrong axis synthetic produces corrective gradient
def test_corrective_gradient():
    init_axis = wrong_axis(angle_error=30°)
    grad = compute_rfsds_grad()
    new_axis = init_axis - lr * grad.axis
    assert angular_error(new_axis, true_axis) < angular_error(init_axis, true_axis)
```

---

## 9. AAAI Novelty (4 contributions, 严格收敛)

### Contribution 1: Single-canonical articulated sparse structure

> From a single closed image and K Wan2.2 pseudo-state conditions, we generate
> ONE canonical sparse structure with an in-loop RASA adapter, decompose it
> into static base and movable part, and derive all K states through a
> single-DoF rigid transform. **Base is bit-identical across K states by
> construction**, eliminating the voxel-jitter problem of K-batched generation.

**Differentiator**:
- vs Stage B (K batched + symmetric mix): 我们 1 次采样消除 jitter
- vs Nano3D (K independent + Voxel-Merge): 我们不需要 merge fix
- vs FreeArt3D (canonical + per-vertex assignment): 我们用 frozen TRELLIS prior

### Contribution 2: K-conditioned in-loop SS-RASA

> K Wan2.2 pseudo-state frames serve as **internal conditioning signals to SS-RASA**,
> not as supervision for K independent SS-DiT samplings. This preserves
> Wan2.2's articulation/exposure information without inducing the
> independent-sampling jitter.

**Differentiator**:
- vs cached-hidden PAM (我们之前的 Stage B/C v8.1): in-loop, 梯度真的回到 RASA
- vs RFSDS-only methods: 我们有 K-cond evidence injection inside generation

### Contribution 3: Canonical-coordinate K-way SLAT-RASA

> We generalize MorphAny3D's two-endpoint SLAT-DiT MCA/TFSA fusion to **K-state
> articulated donor fusion on shared canonical coordinates**. All K SLAT-DiT
> branches operate on the same canonical voxel coords (state-0 frame), with
> only the image conditions varying (I_0..I_{K-1}). Via region-aware
> token-wise donor weights, K cross-attention outputs and self-attention
> activations are blended into a SINGLE canonical textured asset
> (canonical_base + canonical_move), avoiding inverse-warp interpolation
> errors and ensuring physical URDF consistency (one base + one move texture).

**Differentiator**:
- vs MorphAny3D: 2-endpoint morphing → K-state articulated; same canonical-coord trick generalized
- vs CHORD: 不是 free 4D motion，而是 articulated rigid + single canonical texture
- vs 2D pixel-level texture fusion (TEXTure, Paint3D): SLAT latent-space fusion 而非 pixel-space inverse warp
- vs v5 prior version (Option A inverse-warp): 共享 canonical coords 直接 token-wise blend, 无 trilinear 插值误差

### Contribution 4: Wan2.2 W-RFSDS for inserted-module optimization

> Frozen Wan2.2 I2V-A14B provides a fixed-camera, first-frame-anchored video
> prior. Through W-RFSDS gradient (CHORD Eq. 16) flowing back through
> differentiable Gaussian splatting, SLAT-VAE Gaussian decoder, the K-state
> rigid warp, and reaching the inserted neural layers (SS-RASA, SLAT-RASA,
> PartHead, JointHead, support gate, texture adapter), we optimize all
> articulation variables (axis, pivot, q_k) and adapter parameters.
> All TRELLIS and Wan2.2 backbone weights remain frozen throughout.
>
> Wan2.2 A14B's two-expert MoE design naturally supports our coarse-to-fine
> schedule: high-noise expert (early iters) refines geometry/motion, low-noise
> expert (late iters) refines texture/detail. Wan2.2 is **never** treated as
> ground-truth supervision; K Wan frames are self-generated pseudo-state
> conditions, not observations.

**Differentiator**:
- vs CHORD: 优化 inserted modules + articulation params, 不是 4D-GS control points
- vs FreeArt3D: 不需要 K 真实状态图, 只用 self-generated Wan pseudo conditions
- vs DreamFusion-family: 显式 articulation structure + Wan2.2 video prior + frozen TRELLIS

**(NANO3D's Voxel/Slat-Merge philosophy informs our region-preservation thinking
but is not used directly — by-construction base sharing through `B ∪ T(q_k)·M`
removes the need for post-hoc merge. NANO3D appears in Related Work, not as
a contribution source.)**

---

## 10. Ablation Matrix

| Ablation | Goal | Setup |
|---|---|---|
| **A1** No RASA | Baseline lower bound | frozen TRELLIS + zero RASA, no gradient flow to inserts |
| **A2** Cached-hidden RASA | Verify in-loop is critical | 用 v8.1 segmatch 路径作 baseline |
| **A3** K independent SS sampling + merge | Verify single canonical is critical | Nano3D-style 路径 |
| **A4** No K Wan frames (only I_0) | Verify K-cond signal helps | cond_K = [I_0] × 6 |
| **A5** State-code only (no K Wan) | Simpler conditioning ablation | I_0 + 6 state codes |
| **A6** SS-RASA only (no SLAT-RASA) | Geometry-only contribution | 跳过 Phase B4 K-way fusion |
| **A7** SLAT-RASA only (no SS-RASA) | Texture-only contribution | SS 用 TRELLIS 默认 + Slat-Merge |
| **A8** No canonicalization (K SLAT outputs) | Verify canonical-is-better | 保留 K state-specific texture |
| **A9** No PartHead boundary (16³ only) | Verify dual-head is needed | 单 16³ head |
| **A10** Direct 3-vec axis regression | Verify S² discretization | Replace JointHead axis branch |
| **A11** No locked-camera prompt | Verify prompt-locking | use freeform Wan prompt |
| **A12** No first-frame anchor | Verify anchor importance | drop L_first |
| **A13** Uniform τ schedule | Verify CHORD annealing | uniform τ in [0,1] |
| **A14** Different RASA blocks | Sensitivity to placement | {12,16,20}, {16,19,22}, {8,12,16,20} |
| **A15** TI2V-5B vs I2V-A14B | Wan model strength | 比较 PoC 与 full model |

---

## 11. Paper Claims (safe / unsafe)

### Safe to claim

- ✅ Frozen TRELLIS + frozen Wan2.2 + DINOv2 backbones
- ✅ In-loop RASA adapter inside SS-DiT and SLAT-DiT
- ✅ Base bit-identical across K states by architectural construction
- ✅ Single canonical articulated asset output (URDF-ready)
- ✅ K Wan-generated frames are pseudo-state conditions (not observations)
- ✅ W-RFSDS gradient flows to inserted layers, not TRELLIS/Wan weights
- ✅ MorphAny3D-inspired K-way SLAT fusion (with our generalization)

### NEVER claim

- ❌ TRELLIS already understands articulated parts (it doesn't)
- ❌ Local manipulation in TRELLIS implements articulation (it doesn't)
- ❌ MorphAny3D proves K-way articulated fusion works (only proves morphing)
- ❌ Wan2.2 provides part segmentation labels (it doesn't)
- ❌ K Wan frames are real multi-state observations (they aren't)
- ❌ argwhere is differentiable (we bypass via fixed support, not fix it)
- ❌ All hidden interior textures are accurately recovered (mark provenance)
- ❌ Multi-joint kinematic trees solved (single-DoF only in MVP)

---

## 12. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Wan2.2 generates camera drift | High | Locked-camera prompt + negative prompt + first-frame latent clamp |
| Self-confirming loop (bad Wan → bad PartHead → Wan still happy) | Medium | I_0 hard anchor + L_first weight 5×, frame_confidence 学习上限 0.7 (除 c_0) |
| RASA changes base (against region gating) | Low | Region-gated residual: m_base · ΔH = 0 by construction; check 渲染 K state base voxels are bit-identical |
| 24GB GPU OOM | Medium-Low | TI2V-5B + gradient checkpointing + K=4 instead of 6 in PoC |
| Move part too small / empty | High in early iter | L_size with target ratio r_0 = 0.15-0.40, anneal entropy schedule |
| Joint axis local minimum | High for revolute | S² discretization (162-bin icosahedral) + sub-bin refine, multi-start 13 q candidates |
| Gradient vanishes through SLAT-VAE FlexiCubes face topology | Low | FlexiCubes vertex/SDF gradient still flows; topology change is measure-zero in SDS |
| RFSDS texture-only shortcut (geometry doesn't change) | Medium | Phase B-geometry first 1500 iter freeze texture, only optimize geometry |

---

## 13. 参数与计算预算

### Trainable parameters

| Module | Params | % of TRELLIS frozen ~1.2B |
|---|---|---|
| SS-RASA (3 blocks) | ~3.5M | 0.29% |
| Dual PartHead | ~0.6M | 0.05% |
| JointHead (S² + refine + q) | ~0.4M | 0.03% |
| Support gate adapter | ~0.05M | 0.004% |
| SLAT-RASA (3 blocks K-way MCA + TFSA) | ~6.5M | 0.54% |
| Texture AdaptFormer | ~0.5M | 0.04% |
| Frame confidence MLP | ~0.05M | 0.004% |
| **Total** | **~11.6M** | **~1%** |

### Hardware: NVIDIA H800 80GB (quality-first, no consumer-GPU compromise)

我们的实现目标 **H800 80GB**，使用 Wan2.2 **I2V-A14B (full strength)**，K=6 frames, full 480p resolution, 25-step Euler with per-Euler-step gradient checkpointing。

**显存不是设计瓶颈**——v5 的所有架构选择 (single canonical SS, canonical SLAT, Gaussian training) 是为 **first-principle 正确性 / 梯度稳定性** 服务，不是为 fit consumer GPU。

### Memory peak per iter (H800 配置)

| Item | Memory |
|---|---|
| Frozen TRELLIS weights | ~5 GB |
| **Frozen Wan2.2 I2V-A14B weights** (full model) | ~28 GB (14B active × 2 bytes fp16) |
| K=6 batched DINOv2 cond | ~0.1 GB |
| SS-DiT 25-step Euler forward + per-step ckpt | ~3 GB activation + 1.25 GB ckpt |
| SS-VAE decode + grad | ~1 GB |
| PartHead + JointHead + grad | ~0.5 GB |
| K SLAT-DiT donor branches (canonical coords) + ckpt | ~5 GB |
| **SLAT-VAE Gaussian decoder + grad** (training) | ~1.5 GB |
| **Gaussian splatting renderer** (training) | ~1 GB |
| Wan2.2 forward (no_grad teacher, K=6→9 frames @ 480p) | included in weights + ~3 GB activation |
| Wan2.2 VAE encode + grad wrt rendered video | ~1.5 GB |
| **Peak** | **~50-55 GB on H800 80GB (舒服, 25 GB headroom)** |

### Time per asset

- **Per-iter time**: ~10-15 sec on H800 80GB（per-step ckpt 全展开 + I2V-A14B teacher forward）
- **2000 iter / asset**: **~6-8 hours**
- vs CHORD (20h on H200 / I2V-A14B): **~3× 加速**（TRELLIS prior 提供更强的 articulated geometry init，比 CHORD 起点 4D-GS 收敛快）

**注**: Consumer GPU (24GB 4090) deployment 是 future work，可通过 (a) Wan2.2 TI2V-5B 替 A14B (10GB weights), (b) K=4 替 K=6, (c) 降低分辨率 等 deployment-side 优化实现。**这些不是 v5 方法学约束**。

---

## 14. 一句话最终立场

> 我们不生成 K 个独立的 sparse structure。从单张闭态图与 K 个 Wan2.2 self-generated pseudo-state 条件，我们用 in-loop RASA adapter 在 frozen TRELLIS SS-DiT 内生成**单一 canonical sparse structure**；将其分解为静态 base 与可动 part，全部 K 状态通过 single-DoF rigid transform 派生。K-way SLAT-RASA（受 MorphAny3D 启发）将 pseudo-state appearance cues 融合成**单一 canonical textured articulated asset**；frozen Wan2.2 W-RFSDS 优化所有插入模块和关节参数。

---

## 15. References

### TRELLIS family
- TRELLIS (CVPR 2025): arXiv:2412.01506
- MorphAny3D (CVPR 2026): arXiv:2601.00204, https://github.com/XiaokunSun/MorphAny3D
- Nano3D (ICLR 2026): arXiv:2510.15019, https://github.com/JAMESYJL/Nano3D

### Video diffusion priors
- Wan2.2 (HF): Wan-AI/Wan2.2-I2V-A14B, Wan-AI/Wan2.2-TI2V-5B
- CHORD (CVPR 2026): arXiv:2601.04194 — RFSDS formulation we adopt

### Articulated 3D
- FreeArt3D (SIGGRAPH Asia 2025): arXiv:2510.25765
- ArtGS (ICLR 2025): arXiv:2502.19459
- PARIS (ICCV 2023): arXiv:2308.07391 — S² axis discretization
- Articulate Anything via Gaussian Splatting

### Adapter techniques
- LoRA: arXiv:2106.09685
- IP-Adapter: arXiv:2308.06721 (parallel cross-attn pattern)
- AdaptFormer: arXiv:2205.13535 (parallel bottleneck for color)
- Houlsby Adapter: arXiv:1902.00751

### Multi-view / consistency
- AnimateDiff: arXiv:2307.04725 (temporal modules at subset of blocks)
- MVDream: arXiv:2308.16512 (cross-view soft attention)
- DIFT: arXiv:2306.03881 (mid-late layer semantic sweet spot)

### Differentiable rendering / mesh
- nvdiffrast: arXiv:2011.03277
- FlexiCubes: arXiv:2308.05371

---

## Appendix A — Glossary

- **RASA**: RFSDS-optimized Articulated State Adapter — 我们插入 frozen TRELLIS 的可学习层
- **SS / SS-DiT / SS-VAE**: Sparse Structure (TRELLIS 第一阶段，16³ token, 1024-dim)
- **SLAT / SLAT-DiT / SLAT-VAE**: Structured LATent (TRELLIS 第二阶段，sparse voxel features)
- **MCA**: Morphing Cross-Attention (MorphAny3D)
- **TFSA**: Temporal-Feature Self-Attention (MorphAny3D)
- **W-RFSDS**: Weighted Rectified Flow Score Distillation (CHORD §3.2)
- **K-state**: K=6 articulation states (state 0 = closed, state K-1 = max-open)
- **canonical**: state 0 reference frame
- **base / move**: static / movable part
- **single-DoF**: 1 degree of freedom (prismatic 1-D translation OR revolute 1-axis rotation)
- **pseudo-state**: Wan2.2 self-generated condition frame (NOT real observation)
