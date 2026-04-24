# Pipeline Redesign — Training-free Single-image Articulated 3D Reconstruction

**Date**: 2026-04-17
**Status**: Design proposal, awaiting spec review
**Supersedes (partially)**: `mine/record/methods.md` §B and §C；`mine/record/pipeline.md` Stage B/C 描述
**Keeps**: 整体六阶段 pipeline 结构、Stage D/E/F 的 methods.md 原描述（本文件补足 placeholder 实现细节）

---

## 0. 背景与范围

本 spec 是对现有 pipeline 的**结构性修正**，动机：
1. Stage B 当前代码（BCAC、VGCF）的 schedule 窗口错位（中段干预，而形状在前 3 步已定型），且 VGCF 的力度 1/K 归一导致效果不足；BCAC 的 Δz 方差 mask 在相干漂移下退化。
2. Stage C 当前 EM 未利用"刚体 canonical 补全"这一铰接物体第一性先验；轴约束仅用 state-0 anchor 的软惩罚，未用跨状态持续接触集 A\*。
3. Stage D、E 是 placeholder，methods.md §D/§E 的机制未实现。
4. 整个 pipeline 需要一次**无鸡生蛋**的端到端审计。

**本 spec 的输入假设**：

- 单物体、单关节、关节类型 ∈ {revolute, prismatic}
- 固定相机
- 输入为 6 张带地毯（grounding disk）的分割图 `outputs/<id>/rendering_joint_00_state_{00..05}.png`（Stage A 暂跳过）
- 所有神经模型（DINOv2、TRELLIS Stage-1/2、SS VAE、SLat 解码器）frozen，training-free
- 经验观察：带地毯输入下，TRELLIS 不产生显著位置漂移，仅有轻微表面抖动（由 `diagnose_trellis_steps.py` 确认）

---

## 1. 设计原则

| 原则 | 内容 |
|---|---|
| 无循环依赖 | 每阶段的输入只依赖前序阶段或静态几何；EM 内部每一轮只用上一轮的值 |
| 物理先验驱动 | 刚体不形变 → canonical backwarp 补全；持续接触 → 轴位置 |
| 多 epoch 优化 | Stage C 做 20 epoch EM，交替更新分割/轨迹/轴/canonical 几何 |
| 最短路径 | Stage B 保留 TRELLIS 分布内的干预方式；Stage C/D/E 的机制都能用 O_k 和 T_k 直接算 |
| 可消融 | 每个创新点都有明确的 on/off 开关和 baseline 对比 |

---

## 2. Pipeline DAG（前向无环）

```
[I_0..I_5 带地毯]
       │
       ▼
Stage A: 跳过（预生成 6 张图作为输入）
       │
       ▼
[DINOv2(I_k), k=0..5] ──► Stage B: SCAR v3 + post-hoc ICP ──► [O_k (64^3), SCAR diag M_scar]
                                                                      │
                                                                      ▼
                                                       Stage C 初始化（静态几何）
                                                                      │
                                                                      ▼
                                             Stage C: enhanced SAJO EM (≤20 epoch)
                                             dual-fit (revolute / prismatic) + BIC
                                                                      │
                                                                      ▼
              [joint_type, (omega, q, v), phi_k, T_k, M_base_canonical, M_move_canonical, A*]
                                                                      │
                                                                      ▼
                                                    Stage D: swept-volume + boundary cleanup
                                                                      │
                                                                      ▼
                                [M_base_clean, M_move_clean] ──► Stage E: MSV-CA Stage-2 per-part
                                                                                  │
                                                                                  ▼
                                                                    [base mesh, move mesh, u(x)]
                                                                                  │
                                                                                  ▼
                                                                       Stage F: URDF + pybullet
```

**所有边单向，无反向。**

---

## 3. Stage B：SCAR（当前实现）

> **当前状态**（2026-04-18 实现定稿）。Stage B 经过多轮迭代，最终的机制与最初的 "SCAR v3" spec 有显著差异。本节描述**当前代码实现**，而不是历史 spec。关键差异见 §3.8（演进说明）。
>
> **ICP 已从 Stage B 主流程移除**。原因见 §3.5。

### 3.1 输入、输出、流程总览

**输入**：
- 6 张带地毯的分割图 `I_0..I_5`
- DINOv2 patch features `c_0..c_5`（ViT-L/14，每个 1374 tokens × 1024 dim）
- 共享噪声 `eps ~ N(0, I)`, shape `(1, 8, 16, 16, 16)`；扩为 `z_0^(k) = eps.repeat(K)`

**输出**：K 个 64³ 二值占据图 `O_k`，跨状态对齐（base 一致、move 区分）。

**总步数**：**25** Euler 步（TRELLIS-image-large 默认，匹配 FreeArt3D baseline）。

**三相重叠工作流**（每步都做多件事）：

| 阶段 | 步数范围 | 动作 |
|---|---|---|
| Mix | 0..7（8 步） | latent 空间统一混合（所有 K 状态 mix 后得到相同 latent） |
| Push | 0..24（全程 25 步） | Tweedie 空间梯度推动，α 按二次衰减 |
| Plain | 自动 | Mix 结束（步 ≥ 8）且 α → 0 后 → 纯 TRELLIS K-parallel |

Mix 和 push **在步 0-7 重叠**；步 8-24 只有 push；步 24 时 α=0 完全退出。

**完成 25 步 Euler 后**：解码 `z_final → O_k (K, 64³)` 并去除地毯 voxel（`remove_disk`），**直接输出**，不做 ICP 或其他后置对齐。

### 3.2 Mix 阶段（步 0-7）：统一均值混合

**Formula**（`extreme_mix_mode = "mean_of_middles"`，**所有 K 状态使用同一公式**）：

```
middle_mean = (1/(K-2)) · sum_{j=1..K-2} z_t^(j)              ← 中间状态平均
mixed = w_first · z_t^(0) + w_self · middle_mean + w_last · z_t^(K-1)

对所有 k：z_t^(k) ← mixed          ← 6 个状态被设为同一 latent
```

默认权重 `mix_weights = (0.3, 0.4, 0.3)`：

```
mixed = 0.3 · z^(0) + 0.4 · mean(z^(1..K-2)) + 0.3 · z^(K-1)
```

**关键性质**：mix 后 K 个 latent **完全相同**。per-state 差异只能来自后续 DiT forward 给出的 K 个不同 velocity `v^(k)`（由 `c_k` 驱动）和 push 阶段的 per-state 修正。

**传播**：mix 后的 `mixed` 直接作为 DiT 输入 AND Euler 的起点，所以混合轨迹**沿时间传播**（不是临时的 DiT input hack）。

**物理动机**：彻底消除 per-state latent-init 偏置。任何 state 的个性（包括极值状态）都必须靠 DiT 在 plain 阶段的 17 步里重新建立，完全由 `c_k` 条件决定。

**退化分支**：`K < 4` 时没有中间状态可取均值，自动退回 `"symmetric"` 模式（`mixed^(k) = w_first·s_0 + w_self·s_k + w_last·s_{K-1}`，per-state）。

### 3.3 Push 阶段（全程 25 步，α 衰减）

**每个 Euler 步执行顺序**（不论是否在 mix 阶段）：

1. 标准 TRELLIS CFG forward：
   `v_cond^(k) = DiT(z_t^(k), t, c_k), v_uncond^(k) = DiT(z_t^(k), t, null)`
   `v_cfg^(k) = v_uncond^(k) + w · (v_cond^(k) - v_uncond^(k))`（w=7.5，TRELLIS 默认）
2. Tweedie 估计：`x_0^(k) = z_t^(k) - t · v_cfg^(k)`
3. 共识：`x0_bar = (1/K) · sum_k x_0^(k)`
4. **Mask（全程在 latent 16³ 空间）**：
   - 跨状态方差：`sigma²(x) = (1/K) · sum_k ||x_0^(k)(x) - x0_bar(x)||²`（channel 维求和）
   - 空气过滤：`energy(x) = ||x0_bar(x)||²`，`active = energy ≥ percentile_{1-0.1}(energy)` → **保留 top 10% 能量**（物体通常占 16³ 体积的 5-15%）
   - 百分位阈值：`log_sigma² = log(sigma² + 1e-6)`，`τ = percentile_65(log_sigma²[active])`
   - 软 mask：`M(x) = sigmoid((τ - log_sigma²(x)) / 0.5) · active(x)` ∈ [0, 1]
     - 低方差 → M ≈ 1（跨状态一致 = base）
     - 高方差 → M ≈ 0（跨状态差异 = move）
5. **软权重（soft gate with floor）**：
   - `w(x) = [M(x) · (1 - w_floor) + w_floor] · active(x)`，`w_floor = 0.2`
   - 在 base（M=1）：`w = 1` → 满力 push
   - 在 move（M=0）：`w = w_floor = 0.2` → push 力度 `w² = 0.04`（**保留 4% 共识拉力，不归零**）
   - 在空气（active=0）：`w = 0` → 完全无 push
6. **Push**：`v_aug^(k) = v_cfg^(k) + (α(s) / t) · w²(x) · (x_0^(k)(x) - x0_bar(x))`
7. **Euler**：`z_{t-dt}^(k) = z_t^(k) - dt · v_aug^(k)`

**α schedule（二次衰减）**：

```
α(s) = α_peak · (1 - s / (N - 1))²    其中 α_peak = 0.5, N = 25
```

关键步的值：

| step | t | α(s) | α/t（有效推力） |
|---|---|---|---|
| 0 | 1.000 | 0.500 | 0.500 |
| 4 | 0.940 | 0.347 | 0.369 |
| 8 | 0.864 | 0.222 | 0.257（mix 结束） |
| 12 | 0.765 | 0.125 | 0.163 |
| 16 | 0.628 | 0.056 | 0.090 |
| 20 | 0.429 | 0.014 | 0.032 |
| 24 | 0.111 | 0.000 | 0.000 |

**有效推力 α/t 近似线性下降**（mask 在早期最 bimodal，此时推力最大；晚期 mask 退化时推力已自然衰减到接近 0）。

### 3.4 机制背后的三个设计决定

**为什么 mask 在 latent 空间（不在 occupancy 空间）？**
- Push 修改 latent → mask 和 push **必须在同一空间**（同一代数）
- TRELLIS SS VAE 训练时 `lambda_kl = 0.001`（近 AE 而非 VAE）→ latent ≈ 确定性 occupancy 编码，跨状态 latent variance ≈ 跨状态 occupancy variance
- 解码到 64³ 后 pool 回 16³：每个 latent voxel 对应 4³=64 个 occupancy voxel。若这 64 个里半 base 半 move，avg pool 直接抹平 bimodal 结构
- 解码每步成本 100-200 ms × 25 = 3-5 秒浪费

**为什么 `active_fraction = 0.1`（top 10%）？**
- 16³ = 4096 latent voxels，物体典型占 5-15%
- 之前误设为 0.8（保留 top 80%）时，active = 3277 voxel（80% 空气）。空气 voxel 的 sigma² 稀释了物体内部 base/move 的 bimodal 分布
- 改 0.1 → active ≈ 400 voxel（物体核心）→ sigma² 分布重获 bimodal 结构

**为什么 `w_floor = 0.2`（软掩码，不是硬门控）？**
- 硬门控（`w_floor = 0`）给 move voxel 零 push。但方差-based 分类有两类假阴性：
  - Base 内部生成噪声（有时填实有时空） → 被误判为 move → 无修正
  - 单状态 rigid 漂移边界 voxel → 被误判为 move → 漂移无修正
- 软 floor 保证所有物体 voxel 至少收到 `w_floor² = 4%` 的 push
- True-move 区域被"污染" `w_floor² × α ≈ 2%` 的均化——代价远小于收益

### 3.5 Post-hoc ICP（已移除）

**ICP 已从 Stage B 主流程移除**（`v1.yaml: icp_enabled: false`）。

**移除理由**：

1. **Trilinear resample 在二值 64³ 上有损**。ICP 修的是 < 1.5 voxel 的 sub-voxel 漂移，但修正手段（trilinear 平移 + 二值化）本身在物体表面引入抖动。对 < 1 voxel 的微小漂移，"修的毛病"常常比"不修的毛病"更大。
2. **ICP 参考选错**。当前实现把 state 1..K-1 对齐到 state 0。但 state 0 可能是**离群状态**（闭合态图像空间约束弱、柜体位置浮动）。用 state 0 当基准相当于把 5 个正确的拉向 1 个漂的。
3. **Stage C 有更强的对齐机制**。Stage C 的 EM 外层做 canonical 聚合（`move_canonical = max_k(T_k^{-1}(O_k × M_move_k))`），天然处理跨状态 base 的聚合与漂移。再加一道 rigid ICP 反而**两次对齐，一次不当**。
4. **`mean_of_middles` mix 已消除 latent-init 偏置**。SCAR 内部的 mix 使所有 K latent 在 mix 阶段完全相同，base 对齐主要由 DiT 的 c_k-driven v 和 push 机制完成。剩余漂移若存在也是**非刚性**（表面抖动、state-0 拓扑异于 1-5），ICP 这种纯平移工具处理不了。

**代码保留**：`mine/pipelines/utils/icp.py` 和 `stage_b_scar.py` 里的 ICP 分支都保留，通过 `icp_enabled: true` 可以启用作 ablation 对比（验证"加 ICP"是否真能改善，大概率证明是负贡献）。

**未来**：若 Stage C 测出 base 漂移过大（超出 EM 收敛半径），可在 Stage C 的初始化阶段做一次 ICP（使用跨状态共识作参考，不是 state 0）。这是 Stage C 的问题，不是 Stage B 的。

### 3.6 副产物：SCAR diagnostic

- 每步 SCAR mask `M(x)` 保存到 `scar_diagnostics.json`（M_mean, M_mean_active, push_norm, active_voxels, alpha, mixed 等）
- Per-step Tweedie 解码生成 `per_step/O_step_XX.html`（K 状态 dropdown，标注 mix/push/plain 相位）
- `viz/base_move_preview.html`：对最终 O_stack 做 `joint_free_split` 预览 base/move 分类
- **这些 diagnostic 不进入主依赖链**，只用于可视化和 sanity check

### 3.7 消融 / 配置旋钮（v1.yaml）

| 参数 | 默认 | 作用 |
|---|---|---|
| `sampler: scar` | ✓ | 主采样器选 scar（可改 bcac、vgcf） |
| `mix_steps: 8` | ✓ | mix 阶段步数（0 禁用） |
| `mix_weights: [0.3, 0.4, 0.3]` | ✓ | (w_first, w_self, w_last) |
| `extreme_mix_mode: mean_of_middles` | ✓ | 统一均值混合（所有 K 相同）；可改 `symmetric` 恢复 per-state |
| `alpha_peak: 0.5` | ✓ | Step 0 的最大 push 强度 |
| `alpha_decay: quadratic` | ✓ | α 衰减形状（可改 `linear` / `cosine`） |
| `active_fraction: 0.1` | ✓ | Top-N% 能量 voxel 作 active 物体 |
| `tau_percentile: 0.65` | ✓ | log σ² 百分位阈值 |
| `eta: 0.5` | ✓ | sigmoid 锐度 |
| `w_floor: 0.2` | ✓ | 软 mask 最低权重（0 = 硬门控） |
| `icp_enabled: false` | ✓ | Post-hoc ICP 默认关闭（见 §3.5），true 保留作 ablation |
| `icp_max_translation: 1.5` | — | ICP cap（legacy ablation 用） |

消融设计：
- `mix only`：`alpha_peak = 0`
- `push only`：`mix_steps = 0`
- `hard mask`：`w_floor = 0`
- `symmetric mix`：`extreme_mix_mode = symmetric`
- 合成对比：`mix + push`（当前默认） vs 各 ablation

### 3.8 演进说明（历史差异）

当前实现与最初 "SCAR v3" spec 的关键演进：

| 维度 | 初版 (spec) | 中间版本 | **当前** |
|---|---|---|---|
| Euler 步数 | 12（spec 笔误） | 25（修正） | **25**（匹配 FreeArt3D） |
| Mix 步数 | 4 | 8 | **8** |
| Mix 模式 | per-state `(0.3, 0.4, 0.3)` | 同左 | **mean_of_middles**（所有 K 状态同一 mixed latent） |
| α schedule | `[0.7, 1.0, 1.0, 0.5]` 固定 4 项 at 步 4-7 | α=[0.3,0.5,0.5,0.2] at 步 8-11（mix 后） | **二次衰减 0.5→0 全程 25 步** |
| Mask 来源 | latent variance + energy filter | decoder 64³ + pool | **latent variance + energy filter（top 10%）** |
| Mask 空间 | 16³ latent | 64³ → pool → 16³ | **16³ latent 直接** |
| 门控 | 硬门控（M²） | 硬门控（M²） | **软门控 w² with floor 0.2** |
| ICP | 启用，cap 1.5 voxel | 启用 | **禁用**（trilinear 二值化有损 + 参考状态可能是离群，见 §3.5） |

当前版本的核心设计原则：
1. **Mix 和 push 都在 latent 空间**，避免跨空间翻译
2. **Mix 用 uniform `mean_of_middles`**：彻底消除 per-state latent-init 偏置
3. **Push 全程开着，衰减匹配 mask 质量**：mask 最 bimodal 时推力最大，退化时自然归零
4. **软门控**：承认 mask 有假阴性，保留 4% 兜底 push
5. **不做 post-hoc ICP**：rigid 平移 + 二值化的副作用大于它能修的残余漂移；对齐交给 Stage C 的 canonical EM

### 3.9 鸡生蛋检查

- 输入：仅 `DINOv2(I_k)`（frozen）+ 共享噪声
- 输出：`O_k`
- 依赖：DiT forward + SS VAE 解码（都 frozen）
- **无循环依赖**（mask 不依赖 Stage C/D/E 任何输出）

---

## 4. Stage C：SAJO 增强 EM

### 4.1 初始化（epoch 0，完全基于静态几何）

```
# 输入：O_k (K=6, 64^3)

# 跨状态方差分割（沿用 methods.md §C.1 形式，参数继承）
mu_O(x) = mean_k(O_k(x))
sigma_O(x) = std_k(O_k(x))
p_base(x) = mu_O(x) * (1 - sigmoid((sigma_O - sigma_b) / tau_b))  # sigma_b=0.25, tau_b=0.05
p_move(x) = mu_O(x) * sigmoid((sigma_O - sigma_m) / tau_m)        # sigma_m=0.15, tau_m=0.05

# 二值化
B_base = (p_base > 0.5)
B_move = (p_move > 0.5)

# 初始 per-state 分割
M_move_k^(0)(x) = B_move(x) AND O_k(x) > 0.5
M_base_k^(0)(x) = B_base(x) AND O_k(x) > 0.5

# 初始 contact set（state-0 语义下的 anchor）
A^(0) = {x : B_base(x)=1 AND (exists y in N_26(x) with B_move(y)=1)}
connected_component_filter(A^(0), min_size=8, max_components=2)
w(x) = p_base(x) * max_{y in N_26(x)}(p_move(y))   # per-voxel confidence

# 初始 SE(3)
## revolute
omega^(0) = PC1 of A^(0) （加权 PCA，weights = w(x)）
q^(0) = weighted_centroid(A^(0))     # 直接用加权质心；projection 证明等价（线性投影下 centroid 即其在 Pi_omega 中的位置）
v^(0) = q^(0) cross omega^(0)        # Plucker moment，满足 omega . v = 0
phi_k^(0) = projected_angle(
               centroid(M_move_k^(0)) - q^(0),
               centroid(M_move_0^(0)) - q^(0),
               omega^(0))              # 以 omega 为轴的投影角差

## prismatic
v_hat^(0) = PC1 of (centroid(M_move_k) - centroid(M_move_0)) over k
phi_k^(0) = ||centroid(M_move_k) - centroid(M_move_0)|| with sign
```

**所有初值只用 O_k 和静态几何，不依赖任何 T_k 或后续 epoch 的值。**

### 4.2 外层 EM 迭代（每 epoch 7 步）

```
for e = 0, 1, ..., E_max - 1:      # E_max = 20

    # ──────── Step 1：刚体 canonical 几何聚合 ────────
    # 关键：max_k 的 backwarp union 实现"从所有状态补全被挡 move"
    for each k:
        move_k_canonical = trilinear_warp(O_k * M_move_k^(e), T_k^(e).inverse())
    move_canonical^(e+1) = max_k(move_k_canonical)     # 并集聚合，刚体假设下所有 k 应该一致

    # base 不动，直接均值聚合去抖
    for each k:
        base_k_canonical = O_k * (1 - M_move_k^(e))    # identity warp
    base_canonical^(e+1) = mean_k(base_k_canonical)

    # ──────── Step 2：canonical 前投到每状态 ────────
    for each k:
        expected_move_k = trilinear_warp(move_canonical^(e+1), T_k^(e))
        expected_base_k = base_canonical^(e+1)         # 不需要 warp

    # ──────── Step 3：per-state 分割更新 ────────
    for each k:
        M_move_k^(e+1) = expected_move_k * O_k        # 两者都在 [0,1]，乘积仍在 [0,1]，无需 normalize
        M_base_k^(e+1) = O_k * (1 - M_move_k^(e+1))
    # 注意：expected_move_k 在 O_k=0 的位置对 M_move_k^(e+1) 无影响，
    #      但 move_canonical 本身保留了该位置（用于 downstream Stage D/E）

    # ──────── Step 4：持续接触集 A* ────────
    # 对每个状态 k，接触带 = base 与 "move 在状态 k 的物理位置" 的邻近交集
    for each k:
        move_k_phys = (T_k^(e) applied to (move_canonical^(e+1) > 0.5))   # binary 3D mask
        contact_k = (base_canonical^(e+1) > 0.5) AND dilate(move_k_phys, radius=1)
    # 跨状态投票：每个 voxel 被多少状态视为接触
    vote(x) = (1/K) * sum_{k=0..K-1} [1 if x in contact_k else 0]
    A*^(e+1) = {x : vote(x) >= 0.5}    # 半数以上状态都接触的 voxel 集合
    w*(x) = vote(x) for x in A*^(e+1)  # 加权置信度（与投票比例相同）

    # 若 A*^(e+1) < 3 voxels：退化分支
    if |A*^(e+1)| < 3:
        A*^(e+1) = A^(0) with reduced weight 0.3       # fall back to state-0 anchor
        # 不降级到"无约束"：保留静态接触作兜底

    # ──────── Step 5：轴硬投影到 A* ────────
    if joint_type == revolute:
        # 加权 PCA 给出 A* 的主方向和质心
        omega^(e+1) = PC1 eigenvector of weighted_covariance(A*^(e+1), weights=w*)
        q^(e+1) = weighted_centroid(A*^(e+1), weights=w*)     # 质心已经是规范 gauge choice
        v^(e+1) = q^(e+1) cross omega^(e+1)
    elif joint_type == prismatic:
        # prismatic：v_hat 沿 A* 在某一平面内的主方向（rail direction）
        eigvals, eigvecs = eigh(weighted_covariance(A*^(e+1), weights=w*))
        v_hat^(e+1) = eigvec corresponding to largest eigenvalue    # 最长延伸方向

    # ──────── Step 6：Adam M-step，每 inner 步后硬投影 ────────
    S = Parameter(concat([omega, v]))    # 或 v_hat for prismatic
    phi = Parameter(phi_k)
    for inner = 0, 1, ..., n_inner-1:    # n_inner = 10
        # Forward
        T_k = exp_se3([S] * phi_k)       # (K, 4, 4)
        L_data = sum_k sum_x M_move_k^(e+1) * (warp(O_0, T_k.inverse()) - O_k)^2
        L_anchor = sum_{a in A*^(e+1)} w*(a) * (perp distance from a to axis)^2
        L_prior = beta * sum_k phi_k^2
        L = L_data + alpha * L_anchor + L_prior         # alpha=0.1, beta=1e-4

        # Backward + Adam step
        L.backward()
        adam.step()                                      # lr_S=1e-2, lr_phi=5e-3

        # 流形重投
        with torch.no_grad():
            omega = omega / ||omega||
            v = v - (v . omega) * omega                  # Plucker: omega . v = 0
            # 硬投影到 A* 约束线
            if joint_type == revolute:
                q_from_v = omega cross v                # 回推 q
                q_proj = project q_from_v to A* line(omega^(e+1))
                v = q_proj cross omega
            elif joint_type == prismatic:
                v_hat = project v_hat onto A*^(e+1) principal direction

    T_k^(e+1) = exp_se3([concat([omega, v])] * phi_k)

    # ──────── Step 7：收敛检查 ────────
    dT = max_k ||T_k^(e+1) - T_k^(e)||_F
    dM = max_k mean_x |M_move_k^(e+1) - M_move_k^(e)|
    if dT + dM < tol:    # tol = 1e-3
        break

# 输出
return joint_type, (omega, q, v), phi_k, T_k, A*, move_canonical, base_canonical,
       {M_move_k}, {M_base_k}
```

### 4.3 Dual-fit + BIC

Stage C 的上述 EM 独立跑两次：
- `revolute`：初始化 `omega` from anchor PCA，参数空间 R^6 + K
- `prismatic`：初始化 `v_hat` from cross-state move centroid 差方向，参数空间 R^3 + K

完成后比较 BIC：
- `k_rev = 4 + K, k_pris = 2 + K`（参数自由度）
- `N = sum_k |{x : M_move_k(x) > 0.5}|`（有效样本数）
- `BIC(t) = 2 * L_data_final(t) + k_t * log(N)`
- `type* = argmin_t BIC(t)`

Posterior 置信度：`p(type* | data) = exp(-BIC(type*)/2) / sum_t exp(-BIC(t)/2)`。
若 `p(type*) < 0.7`，标记为 low-confidence。

### 4.4 鸡生蛋检查

- epoch 0 初始值：仅 `O_k` + 静态几何（方差、anchor PCA）
- epoch e+1 的输入：**严格** 仅来自 epoch e 的输出
- 每步内部：Step 1 → 2 → 3 → 4 → 5 → 6 → 7 顺序无环
- **无循环依赖**

---

## 5. Stage D：冲突清理

### 5.1 输入

`M_base_canonical, M_move_canonical, T_k, A*, phi_min, phi_max` from Stage C。

### 5.2 步骤

```
# D1: 冲突分类
Overlap = {x : M_base_canonical(x) > 0.5 AND M_move_canonical(x) > 0.5}
for x in Overlap:
    d(x, A*) = min_{a in A*} ||x - a||
    Containment(x) = (d(x, A*) > 2)
    Boundary(x) = (d(x, A*) <= 2)

# D2: Containment cleanup via swept-volume
theta_samples = linspace(phi_min, phi_max, 32)
S = union_{theta in theta_samples} T_theta(M_move_canonical)
# evidence restriction：排除未被任何观测支持的 swept voxel
Observed = union_k (T_k^{-1}(O_k > 0.5))
S_evidence = S intersect Observed
for x with Containment(x) = True:
    M_base_swept(x) = M_base_canonical(x) * (1 - I(x in S_evidence))
for x elsewhere:
    M_base_swept(x) = M_base_canonical(x)

# D3: Boundary cleanup via 对称形态学修正
for x with Boundary(x) = True:
    M_move_final(x) = M_move_canonical(x) * (1 - I(x in dilate(boundary(M_base_swept), 1)))
    M_base_final(x) = M_base_swept(x) OR erode(boundary(M_move_canonical), 1)(x)
for x elsewhere:
    M_move_final(x) = M_move_canonical(x)
    M_base_final(x) = M_base_swept(x)

# D4: 接触带保护
for x with d(x, A*) <= 2:
    M_base_final(x) = M_base_canonical(x)     # 强制恢复
    M_move_final(x) = M_move_canonical(x)
```

### 5.3 鸡生蛋检查

- 输入：Stage C 输出
- 纯几何操作，无迭代
- **无循环依赖**

---

## 6. Stage E：MSV-CA per-part Stage-2

### 6.1 输入

`M_base_final, M_move_final, T_k, {I_k}, {DINOv2(I_k)}, {O_k}, camera_pose`。

### 6.2 E.1 可见性计算（Amanatides-Woo DDA）

对每个 canonical voxel x 和每个状态 k：
```
# 物理位置
if M_base_final(x) > 0.5:
    x_k_phys = x
elif M_move_final(x) > 0.5:
    x_k_phys = T_k(x)
else:
    v_k(x) = 0
    continue

# 射线
r_origin = camera_center
r_dir = normalize(x_k_phys - camera_center)
t_end = ||x_k_phys - camera_center||

# 沿射线累积占据
alpha_acc = 0
for voxel y hit along ray with t_hit < t_end:
    alpha_acc = 1 - (1 - alpha_acc) * (1 - O_k(y))
v_k(x) = 1 - alpha_acc
```

实现：NerfAcc 或自定义 CUDA kernel，并行 over (x, k)。H800 上约 5 秒/物体。

### 6.3 E.2 K-parallel DINOv2 conditioning

`c_dino^k = DINOv2_ViT_L_14(I_k) in R^{1374 x 1024}`，k=0..5 独立保存。

### 6.4 E.3 MSV-CA 融合

在 Stage-2 的每个 `ModulatedSparseTransformerCrossBlock`：

```
# 标准 per-state cross-attn（K 并行）
for k in 0..K-1:
    Q = z_slat W_Q^b
    K_k = c_dino^k W_K^b
    V_k = c_dino^k W_V^b
    A_k(x) = softmax(Q(x) K_k^T / sqrt(d)) V_k

# Visibility-weighted 融合
for x:
    if max_k v_k(x) < 0.05:
        weight_k(x) = 1/K                    # always-occluded，fallback 均匀
    else:
        weight_k(x) = exp(tau * v_k(x)) / sum_j exp(tau * v_j(x))    # tau=5

A_fused(x) = sum_k weight_k(x) * A_k(x)
```

A_fused 替换原 block 的 cross-attn 输出，self-attn、AdaLN、FFN 不动。

### 6.5 E.4 Per-part decoding

- **Base pass**：active voxel mask = `{x : M_base_final(x) > 0.5}`，跑完整 MSV-CA Stage-2 sampling（12 Euler steps），SLat 解码为 base mesh
- **Move pass**：active voxel mask = `{x : M_move_final(x) > 0.5}`，同上，解码为 move mesh
- 两个 pass 独立，可并行

### 6.6 E.5 不确定性图

`u(x) = 1 - max_k v_k(x)`；`u(x) > 0.95` 标记为 always-occluded（定性图用半透明叠加，定量表作为 "U_ratio" 报告）。

### 6.6b U_ratio（定量报告指标）

```
U_ratio = |{x : u(x) > 0.95 AND (M_base_final(x) > 0.5 OR M_move_final(x) > 0.5)}|
        / |{x : M_base_final(x) > 0.5 OR M_move_final(x) > 0.5}|
```

表示物体体积中**完全依赖生成先验**（任何状态都未真实观测到）的比例，每个实验必须报告。

### 6.7 鸡生蛋检查

- 输入：Stage C + Stage D 输出 + Stage B 的 `O_k` + 6 张图 + DINOv2
- 纯 forward，无迭代
- **无循环依赖**

---

## 7. Stage F：URDF + pybullet

### 7.1 步骤

- **Mesh 提取**：Stage E 输出的 SLat 通过 frozen SLat mesh decoder（Flexicubes）得到 visual mesh
- **凸分解**：CoACD threshold=0.05，生成 collision mesh
- **URDF emission**：two-link (base_link, move_link)，joint 类型从 Stage C，axis=omega 或 v_hat，origin=q 或 centroid(M_base_final)，limit=[phi_min, phi_max]
- **pybullet 验证**：pybullet.DIRECT 模式，均匀采样 20 个 joint 值，每个采样调 `performCollisionDetection()` 和 `getContactPoints()`，若 penetration depth > 1e-3 归一化单位 → 标记 fail

失败时**不回退 Stage D**（spec 规定不允许降级兜底）；报告失败物体为 unrecoverable，供实验阶段人工分析。

### 7.2 鸡生蛋检查

- 输入：Stage E + Stage C 输出
- 纯 CPU 操作
- **无循环依赖**

---

## 8. 全链路鸡生蛋审计

所有潜在循环点及处理：

| # | 疑似循环 | 处理 |
|---|---|---|
| 1 | methods.md §B.2 SDEdit 需要 Stage C 的 T_k 先验 | **删除**（SCAR v3 不需要） |
| 2 | SCAR 的 mask 需要当前步 Tweedie；Tweedie 来自当前步 v_cfg | 单步内顺序计算，非循环 |
| 3 | Stage C 分割 M_move 需要 T_k；T_k 需要 M_move | EM 交替（上一轮 T_k → 本轮 M_move → 本轮 T_k）；初始值来自静态几何 |
| 4 | 轴约束需要 A*；A* 需要 T_k；T_k 需要轴约束 | EM 交替（上一轮 T_k → 本轮 A* → 本轮轴 → 本轮 T_k） |
| 5 | canonical move 需要 T_k^{-1} backwarp | EM 交替，初始 T_k^(0) 从 anchor PCA + state-0 几何 |
| 6 | Stage E visibility 需要 T_k | 前向依赖（T_k 已 Stage C 产出） |
| 7 | Stage D swept 需要 T_k 和 phi 范围 | 前向依赖 |

**结论**：所有"循环"都是 EM 交替或前向依赖，无真正循环依赖。

---

## 9. 代码改动全景

| 文件 | 状态 | 行动 |
|---|---|---|
| `mine/TRELLIS/trellis/pipelines/samplers/scar.py` | 不存在 | **(DONE 2026-04-17) 新增 `scar.py`**（SCAR v3：Tweedie-space gradient push，λ=α/t，去掉 /K，schedule 改早期 [0.7, 1.0, 1.0, 0.5]）。VGCF 保留为 legacy ablation，结构 90% 一致 |
| `mine/TRELLIS/trellis/pipelines/samplers/bcac.py` | 现有 | (保留) legacy ablation，不激活 |
| `mine/TRELLIS/trellis/pipelines/samplers/vgcf.py` | 现有 | (保留) legacy ablation，不激活 |
| `mine/pipelines/stage_b_vgcf.py` | 现有 | (保留) legacy ablation，不动 |
| `mine/pipelines/stage_b_scar.py` | 不存在 | **(DONE 2026-04-17) 新增**：SCAR 驱动 + ICP 串联 + 可视化 |
| `mine/pipelines/utils/icp.py` | 不存在 | **(DONE 2026-04-17) 新增**：rigid translation-only ICP（compute_translation_offset、apply_translation、align_to_reference） |
| `mine/tests/test_icp.py` | 不存在 | **(DONE 2026-04-17) 新增**：9 个单元测试（零偏移、整数偏移、mask、roundtrip、cap） |
| `mine/tests/test_scar_formula.py` | 不存在 | **(DONE 2026-04-17) 新增**：6 个单元测试（mask 零方差/双峰/α=0/M=0/拉向共识/M² 选择性） |
| `mine/tests/test_scar_sampler.py` | 不存在 | **(DONE 2026-04-17) 新增**：4 个集成测试（形状、baseline 等价、诊断分段、output 差异） |
| `mine/tests/test_stage_b_scar_driver.py` | 不存在 | **(DONE 2026-04-17) 新增**：2 个驱动测试（形状/位移修正） |
| `mine/tests/test_stage_b_e2e_30857.py` | 不存在 | **(DONE 2026-04-17) 新增**：30857 端到端烟雾测试（需 CUDA + TRELLIS 权重 + 输入数据） |
| `mine/scripts/compare_stage_b_samplers.py` | 不存在 | **(DONE 2026-04-17) 新增**：SCAR vs plain vs VGCF vs BCAC 消融对比脚本 |
| `mine/pipelines/sajo/anchors.py` | 只有 state-0 anchor | **新增** `compute_persistent_contact_set(T_k, M_move_canonical, M_base_canonical, K)` |
| `mine/pipelines/sajo/em.py` | 无 canonical 聚合 | **重写外层**：加入 Step 1-5 的 canonical 聚合 / 持续接触 / 硬轴投影 |
| `mine/pipelines/sajo/screw.py` | 无 hard A* projection | **新增** `project_q_to_anchor_line(q, omega, A_star, weights)` |
| `mine/pipelines/stage_c_sajo.py` | 现有 | 更新 driver 以对接新 EM 输出（新增 canonical 几何、A* 的持久化） |
| `mine/pipelines/stage_d_placeholder.py` | placeholder | **重写为 `stage_d_cleanup.py`**：D1-D4 实现 |
| `mine/TRELLIS/trellis/modules/sparse/transformer/` Stage-2 block | 无 MSV-CA | **新增** `MSVCABlock`（替换原 `ModulatedSparseTransformerCrossBlock` 在 MSV-CA pipeline 下的调用） |
| `mine/pipelines/stage_e_msvca.py` | 不存在（当前是 placeholder Stage-2） | **新增**：E.1 可见性 + E.3 MSV-CA + E.4 per-part + E.5 uncertainty |
| `mine/pipelines/stage_f_assemble.py` | 现有 | 对接 Stage E 输出（per-part mesh + 置信度） |
| `mine/configs/v1.yaml` | 现有 | (DONE 2026-04-17 Stage B: 新增 scar 块、默认 sampler=scar) 后续更新 Stage C/D/E 参数 schema |
| `mine/run_v1.py` | 现有 | (DONE 2026-04-17 Stage B: 新增 SCARSampler 分支 + run_scar 调用 + 目录 stage_b_vgcf→stage_b + vgcf_res→stage_b_res) 后续串联 C/D/E/F |

---

## 10. 超参默认

| Stage | 参数 | 值 |
|---|---|---|
| B | K=状态数 | 6 |
| B | Euler 步数 | 12 |
| B | CFG strength | 7.5 |
| B | CFG interval | [0, 1] |
| B | SCAR 早期步 | 0..3 |
| B | alpha schedule | [0.7, 1.0, 1.0, 0.5] |
| B | active_fraction | 0.8 |
| B | tau_percentile | 0.65 |
| B | eta (sigmoid sharpness) | 0.5 |
| B | eps_log | 1e-6 |
| B | ICP max translation | 1.5 voxel |
| C | n_outer (max epoch) | 20 |
| C | n_inner (Adam steps) | 10 |
| C | lr_S | 1e-2 |
| C | lr_phi | 5e-3 |
| C | alpha (anchor weight) | 0.1 |
| C | beta (prior weight) | 1e-4 |
| C | tol (convergence) | 1e-3 |
| C | sigma_b, sigma_m | 0.25, 0.15 |
| C | tau_b, tau_m | 0.05, 0.05 |
| C | anchor min component | 8 voxels |
| C | A* vote threshold | 0.5 |
| C | A* degenerate fallback threshold | < 3 voxels |
| D | swept-volume sample count | 32 |
| D | contact band radius | 2 voxels |
| D | boundary dilation/erosion radius | 1 voxel |
| E | visibility sharpness tau | 5 |
| E | always-occluded threshold (v_max) | 0.05 |
| E | uncertainty threshold (u) | 0.95 |
| F | pybullet 采样数 | 20 |
| F | penetration tolerance | 1e-3 (归一化) |
| F | CoACD threshold | 0.05 |

---

## 11. 消融计划（实验章节骨架）

| 实验 | 对比项 | 主要指标 |
|---|---|---|
| Exp 1 | Stage B：SCAR v3 / Option A / plain / BCAC / VGCF | 跨状态 base IoU、表面 Chamfer |
| Exp 2 | Stage B：with/without ICP | 位置漂移量（rigid alignment error） |
| Exp 3 | Stage C：canonical 聚合 on/off | Stage C EM 收敛速度、M_move 精度 |
| Exp 4 | Stage C：A* 持续接触 on/off（退回 state-0 anchor） | 轴角度/位置误差 |
| Exp 5 | Stage C：硬轴投影 on/off | 同上 |
| Exp 6 | Stage D：冲突清理 on/off | pybullet 验证通过率、per-part Chamfer |
| Exp 7 | Stage E：MSV-CA / TRELLIS 多图 / single-state | held-out open-state PSNR |
| Exp 8 | 端到端：对比 PAct、FreeArt3D、DreamArt、ArtiLatent on PartNet-Mobility | 关节角/位置误差、Chamfer、验证率 |

---

## 12. 开放问题（诚实列出）

1. SCAR v3 未经实验验证，可能在某些物体类型上效果和 Option A 持平或略差 → fallback 到 Option A 是合理选项，**不写入代码兜底**，仅作为 ablation 结论。
2. A\* 持续接触在 `phi_max < 5°` 的小角度下可能接近退化（所有 K 状态的接触集几乎相同），此时 EM 对持续接触的敏感度下降。实验监控 `|A*| / |A^(0)|` 作为退化指标。
3. 刚体 canonical 补全假设 move 在整个 joint 范围内是**单一刚体**。对铰链有次级关节或柔性耦合的物体不适用（超出本文范围）。
4. 带地毯输入假设由手工或 `add_disk.py` 保证；未来 Stage A（Wan2.2）阶段需要独立验证地毯在生成视频中保留。
5. Stage E 的 MSV-CA 实现需要修改 TRELLIS Stage-2 的 transformer block 内部；需要保持原模型权重不变的前提下只改 forward 逻辑。

---

## 13. 与 methods.md / pipeline.md 的差异摘要

| 文件 §节 | 现有描述 | 本 spec | 原因 |
|---|---|---|---|
| methods.md §B.1 (K/V sharing) | steps 4-8, blocks B/4..3B/4 | **替换为 SCAR v3**（Tweedie-space gradient push, steps 0-3, λ=α/t 缩放） | schedule 错位；原 VGCF 机制本身对但力度不足（/K 归一） |
| methods.md §B.2 (SDEdit) | t=0.5 注入 | **删除** | 需要初步 T_k，鸡生蛋 |
| methods.md §B.3 (ICP) | 有 | **保留** | 已验证的简单有效手段 |
| methods.md §C.2-C.4 (anchor) | state-0 接触 + PCA 软惩罚 | **替换为 A\* 持续接触 + 硬投影** | 物理更对，跨状态信息未利用 |
| methods.md §C.5 (EM) | forward warp 数据项，E-step 只更新 M | **扩展为 7 步**：canonical 聚合 + 前投分割 + A\* 更新 + 硬轴投影 + Adam | 未利用刚体 canonical 补全 |
| methods.md §D, §E | 完整描述但代码是 placeholder | **实现如 §5, §6** | 需要落地 |

---

**结束**。请 review 后反馈。
