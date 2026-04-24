# Stage C — SegMatch v3 Spec

> **Date**: 2026-04-22
> **Status**: 设计定稿，待实施
> **Scope**: 单关节 MVP（P=1），架构预留 multi-joint（star kinematic tree）
> **Deadline**: 6/1；MVP 5 天内跑通一个 sample，ablation 5/1 起
> **不替换**: `pipelines/sajo/`（保留作 legacy ablation）
> **不改**: `方案.md` / `贡献和动机.md`（另起 spec，后续再汇入）

---

## 0. 对齐（给审稿人的一句话）

> Stage C 的输入是 Stage B v4.3 的 `(O_stack, z_final, M_attn_64)`；输出是 `(joint_type, axis, T_k, phi_k, canonical_base, canonical_move, contact_region, per_state_assignment)`，喂给 Stage D/E 做 visibility-aware 跨状态纹理回灌（论文主卖点）。Stage C 本身在论文里是支持模块（1-1.5 页 methods，2-3 行 ablation），novelty 只声明 "cross-stage M_attn → graph-cut unary" 一条。

---

## 1. 定位（锁死"Stage C 只做什么"）

### 1.1 Stage B v4.3 已产出（不重复造轮子）

| 产物 | 形状 | 语义 | Stage C 如何用 |
|---|---|---|---|
| `O_stack` | (K=6, 64, 64, 64) binary | 跨 state aligned occupancy | 数据项、graph-cut data term、canonical 聚合 |
| `z_final` | (K=6, 8, 16, 16, 16) float | TRELLIS SS latent（8 通道） | per-voxel descriptor，跨 state matching |
| `M_attn_64` | (64, 64, 64) ∈ [0, 1] | 语义 base/move 判别（v4.3 核心） | graph-cut unary prior，base centroid |
| `P_base_shared_64_filtered` | (64, 64, 64) | 几何+语义融合 base prob | fallback；MVP 不用 |

### 1.2 Stage C 的独特职责（Stage B 做不到的）

1. **关节推断**: joint_type ∈ {revolute, prismatic} + axis `(omega, q)` / `(v_hat)`
2. **Per-state rigid transform T_k**: K 个 SE(3) 变换
3. **Per-state joint param phi_k**: 角度（rev）或距离（pris）
4. **Canonical per-part volume**: `canonical_base`, `canonical_move`（给 Stage D 做纹理回灌）
5. **Contact region**: 连通接触带（Stage D 边界约束）
6. **Per-state assignment**: `(K, 64, 64, 64)` int ∈ {0=base, 1=move}（Stage E 用）

### 1.3 Stage B v4.3 没解决、Stage C 必须处理的三个根因

- **根因 A（结构先验）**: rigid body axiom — 同一 part 共享一个 T_k；接触区-轴几何相容（target.md G3）
- **根因 B（voxelization aliasing）**: 64³ 离散 rigid warp 不精确 → median warp + 可选 128³ upsample
- **根因 C（TRELLIS 生成不一致）**: 各 state DiT 独立 inference → robust statistics (RANSAC/median)

---

## 2. 为何不是 SAJO EM / Structured EM（决策记录）

见 [record/stageC/1.md](1.md)、[2.md](2.md)、[3.md](3.md)、[4.md](4.md) 的 brainstorm 演化，以及 2026-04-22 的对谈判决。核心：

| 方案 | 致命问题 |
|---|---|
| SAJO EM（当前 `pipelines/sajo/`） | variance split 复发 Stage B problem 13；state-0 anchor; warp 数据项方向错；硬编码 single-part |
| Structured EM + graph-cut | EM 初始化敏感不解；per-voxel α 违反 rigid body；多 label α-expansion 慢；未用 z_final |
| SegMatch-Lite（1.txt 原案） | state-0 reference 复发 outlier；window 太小；amb_k 补丁；Kabsch 无 outlier 防御 |
| **SegMatch v3（本 spec）** | 吸收以上所有 principled 修正，避免过度工程 |

---

## 3. I/O 接口

### 3.1 输入（严格）

```python
O_stack: torch.Tensor        # (K, 64, 64, 64) float ∈ [0, 1]（soft）或 binary
z_final: torch.Tensor        # (K, 8, 16, 16, 16) float
M_attn_64: torch.Tensor      # (64, 64, 64) float ∈ [0, 1]
# 全部从 outputs/<sample>_421/stage_b/ 直接读
```

### 3.2 输出（给 Stage D）

```python
@dataclass
class StageCResult:
    joint_type: str                          # "revolute" | "prismatic"
    omega: torch.Tensor                      # (3,) revolute 轴方向；prismatic 里 = v_hat
    q: torch.Tensor                          # (3,) revolute 轴上一点；prismatic 0 向量
    v: torch.Tensor                          # (3,) revolute Plücker moment；prismatic v_hat
    T_k: torch.Tensor                        # (K, 4, 4) SE(3), canonical→state k
    phi_k: torch.Tensor                      # (K,) per-state 角度/距离，phi_0 = 0
    canonical_base: torch.Tensor             # (64, 64, 64) binary
    canonical_move: torch.Tensor             # (64, 64, 64) binary
    contact_region: torch.Tensor             # (64, 64, 64) binary
    per_state_assignment: torch.Tensor       # (K, 64, 64, 64) int8 ∈ {0, 1}
    meta: Dict[str, Any]                     # BIC_rev, BIC_pris, inlier_counts, 消融旋钮
```

### 3.3 世界坐标约定（继承 `pipelines/sajo/warp.py`）

voxel index `(i, j, k) ∈ [0, 63]³` → world point `p = (i, j, k) / 63 - 0.5`，单位立方体 `[-0.5, 0.5]³` 居中于原点。所有 T_k 在世界坐标作用。

---

## 4. 七步算法（MVP 主干）

### C.1 候选分区

```
base_soft(v)   = M_attn_64(v)                                  # 全 voxel soft prior，给 graph-cut unary
move_cand_k    = O_k ∩ (M_attn_64 < τ_move)                    # hard move 候选，限 matching 搜索空间
base_centroid  = weighted centroid of {v : M_attn_64(v) > τ_high} ∩ (mean_k O_k > 0.5)
                  with weights w(v) = M_attn_64(v)              # 用作 canonical frame origin
```

- `τ_move = 0.3`：M_attn < 0.3 → 有把握的 move
- `τ_high = 0.7`：M_attn > 0.7 → 有把握的 base（用于 centroid；不用于 matching）
- **注意**：base 不参与 matching（base 不动），matching 只在 move_cand 之间做

### C.2 Feature upsample

```
F = trilinear_interpolate(z_final, 16³ → 64³)                  # (K, 8, 64, 64, 64)
F_normed = F / ||F||_2 along dim=channel + eps                 # L2 normalize per voxel
```

MVP 只用 `z_final` 的 8 通道（指令 (a)）。若 C.3 inlier 率 < 20% 或 BIC flip，再上 DiT mid-layer hook（~30 行 Stage B 改动，ablation）。

### C.3 Pairwise mutual-best matching

对每对 `(k, k')`，`k < k'`，共 C(K, 2) = 15 组：

```
V_k   = move_cand_k 的 voxel 坐标 (N_k, 3), N_k ≈ 3K-5K
V_k'  = move_cand_{k'}                                          (N_{k'}, 3)

# Feature cosine + spatial gating (全局 NN, 不限 window)
sim_feat(i, j)  = F_normed[k, :, V_k[i]] · F_normed[k', :, V_k'[j]]          # (N_k, N_k')
dist_spat(i,j)  = ||V_k[i] - V_k'[j]||_2 / 63                                # normalize to [0, 1]
cost(i, j)      = (1 - sim_feat(i, j)) + λ_spat · dist_spat(i, j)           # λ_spat = 0.3

# Mutual-best NN
j_best(i) = argmin_j cost(i, j)                                # for each v in V_k
i_best(j) = argmin_i cost(i, j)                                # for each v' in V_k'
mutual    = {(i, j) : j = j_best(i) ∧ i = i_best(j)}

# Lowe ratio test (τ_lowe = 0.88 @ 8-dim, 见 §8.1)
second_best_j(i) = second smallest cost(i, :)
pass_lowe(i, j)  = cost(i, j) / second_best_j(i) < τ_lowe

matches[(k, k')] = [(V_k[i], V_k'[j]) : (i, j) ∈ mutual ∧ pass_lowe(i, j)]
```

**实现**:
- `torch.cdist` + L2 normalize 算 feature sim（GPU，~50 ms per pair）
- 15 对全做，总算力 ~2 秒，可接受（见 §8.2）
- 不引 FAISS 依赖

### C.4 刚体拟合（RANSAC-Kabsch + joint-constrained fit + BIC）

#### C.4.1 Per-pair RANSAC-Kabsch（MVP）

对每组 `matches[(k, k')]`，跑 RANSAC：

```
for iter in 1..RANSAC_ITERS:                                    # RANSAC_ITERS = 200
    sample 4 pairs uniformly at random
    T_hyp = Kabsch_SVD(sampled 4 src-tgt pairs)                # 4×4 SE(3)
    inliers = [(src, tgt) : ||T_hyp · src - tgt||_2 < τ_ransac]
                                                                # τ_ransac = 2/63 (≈ 2 voxel)
    if |inliers| > |best_inliers|:
        best_inliers = inliers
T_{k→k'} = Kabsch_SVD(best_inliers)                             # 最终闭式解
```

#### C.4.2 Gauge: canonical frame

```
T_0 = I (以 state 0 的 move 相位 = phi_0 = 0)
canonical_world_origin = base_centroid   (from C.1)             # 几何 origin = base 加权质心
                                                                # 彻底消 state 0 outlier 对 base 的污染
```

- 注意：`T_0 = I` 是 move 部件的 phase gauge，不是 base 的。Base 在所有 state 都不动，其一致统计量是 K 个 state 的 M_attn-high voxel 加权质心（世界坐标系 origin）
- MVP 里 `T_k = T_{0→k}` 直接取 RANSAC 结果（用 pair (0, k)）
- Ablation：用 15 组 `T_{k→k'}` 做 pose graph 一致化（见 §5.1）

#### C.4.3 Joint-constrained re-fit（revolute / prismatic 各一遍）

已有 unconstrained `T_k` 后，强制到 revolute / prismatic manifold：

**Revolute 分支**:

```
# 参数：omega ∈ S², q ∈ R³（但因 gauge 冗余，取 q ⊥ omega），phi_k ∈ R (K 个)
# 共享 axis: (omega, q)，每个 state 一个角度 phi_k

初始化:
    from T_{0→k}, k=1..K-1:
        log_k = logm(T_{0→k})                                 # se(3), (6,)
        omega_k = log_k[3:] / ||log_k[3:]||                   # rotation part of log
    omega_init = weighted_mean_spherical(omega_k, weights = |angle_k|)
                                                              # 加权球均值
    q_init = mean_k(closest point on axis of T_k)

for outer in 1..MAX_FIT:                                       # MAX_FIT = 30
    # Adam refine (omega, q, phi_k) with manifold projection
    L = Σ_{(k,k')} Σ_{(v,v')∈matches[k,k']} ||exp_se3_rev(omega, q, phi_k'-phi_k)·v - v'||²
    L.backward(); adam.step()
    omega ← omega / ||omega||
    q ← q - (q·omega)·omega            # remove gauge along ω
```

**Prismatic 分支**:

```
# 参数：v_hat ∈ S², t_k ∈ R (K 个)
初始化:
    from T_{0→k}:
        trans_k = translation part of T_{0→k}
    v_hat_init = normalize(mean_k(trans_k))
    t_k_init = trans_k · v_hat_init

for outer in 1..MAX_FIT:
    L = Σ_{(k,k')} Σ_{(v,v')} ||v + (t_k' - t_k)·v_hat - v'||²
    L.backward(); adam.step()
    v_hat ← v_hat / ||v_hat||
```

#### C.4.4 BIC 选 joint type

```
L_rev_final  = Σ_{(k,k')} Σ_{(v,v')} ||T_k_rev(v) - T_k'_rev(v)||² 在 matches 上
L_pris_final = 同上, 用 prismatic T_k
N            = Σ_{(k,k')} |matches[k, k']|            # 总 match pair 数
k_rev        = 4 + K      (omega: 2 dof + q projected: 2 dof + phi_k: K)
k_pris       = 2 + K      (v_hat: 2 dof + t_k: K)
BIC_rev      = 2 L_rev_final + k_rev · log(N)
BIC_pris     = 2 L_pris_final + k_pris · log(N)
joint_type   = argmin(BIC_rev, BIC_pris)
```

**注意（诚实 footnote，见 §8.3）**: 15 对 pairwise 的 residuals 不完全独立（同一 voxel 可出现在多个 pair），BIC 偏乐观。相对 ranking 仍稳，paper methods 里明说。

### C.5 2-label graph-cut（base vs move）

```
# Unary potentials in logit space（数值稳定, §4.2）
logit_attn(v) = log(M_attn_64(v) + ε) - log(1 - M_attn_64(v) + ε)         # ε = 1e-3

U_base(v) = -logit_attn(v)                                                  # high M_attn → low cost for base
U_move(v) = +logit_attn(v) - λ_data · data_term(v)                          # λ_data = 2.0
           # data_term(v) = (1/K) Σ_k O_k(T_k(v))                           # 用 T_k 前向预测
                                                                            # Sigmoid: higher → more move

# Pairwise Potts on 26-neighborhood
P(v, v') = λ_smooth · [label(v) ≠ label(v')]
λ_smooth  = λ_0      (MVP, MVP λ_0 = 1.0)                                   # ablation 里自适应，见 §5.2

# α-expansion graph cut (PyMaxflow, 2-label)
# ≈ 1 秒 @ 64³ 2-label
label(v) ∈ {base, move}                                                     # hard assignment
```

**Gotcha**: 只在 `O_k` 或 `mean_k(O_k)` > threshold 的 voxel 上跑 graph-cut（avoid air voxels）。air voxel 直接标 background 不参与。

### C.6 轴精化（contact principal axis）

```
# 从 C.5 assignment 提取接触带
contact = dilate(base_assigned, 1) ∩ move_assigned          # 26-connectivity dilate

# Contact 的 1-dim principal axis (PCA 最大特征向量)
eigvals, eigvecs = PCA(contact voxel coords, weights = M_attn 邻域)
principal_axis_dir = eigvecs[:, argmax(eigvals)]            # 3-dim unit vector
principal_axis_pos = weighted_centroid(contact voxels)     # 3-dim point

# Adam refine (仅 axis, 不动 assignment)
if joint_type == "revolute":
    # 硬约束: omega 的方向倾向 principal_axis_dir
    #         axis line 穿过 principal_axis_pos
    L_axis = w_dir · (1 - |omega · principal_axis_dir|)
           + w_pass · Σ_{a ∈ contact} m_attn(a) · ||(a - q) × omega||²
                                          # 加权垂距平方
    L_total = L_rigid_fit + λ_axis · L_axis
elif joint_type == "prismatic":
    # v_hat 倾向 principal_axis_dir
    L_axis = (1 - |v_hat · principal_axis_dir|)
    L_total = L_rigid_fit + λ_axis · L_axis

# Optimize (omega, q, phi_k) / (v_hat, t_k) only；assignment frozen
# 10-20 步 Adam 够
```

**物理对齐**: target.md G3 原话 "旋转轴必须穿过 base–move 接触边界附近的离散 anchor set 或其窄邻域"。我们的 contact region 就是 anchor set；principal axis 是它的 1-dim 结构。

### C.7 Canonical aggregation（median warp）

```
# Per-state backwarp 到 canonical
for k in 0..K-1:
    O_k_move = O_k ∩ move_assigned                          # state k 的 move voxel
    back_k   = trilinear_warp(O_k_move, T_k^{-1})          # 回到 canonical

# Median over K (robust to TRELLIS noise, §根因 C)
canonical_move = median_k(back_k) > 0.5                     # binary
canonical_base = median_k(O_k ∩ base_assigned) > 0.5        # base identity warp

# Contact region (Stage D 边界约束)
contact_region = dilate(canonical_base, 1) ∩ canonical_move
```

**Ablation（§5.3）**: `canonical_move_128 = median_k(128³_upsample → trilinear_warp → 64³_downsample)`；减少 aliasing。MVP 用 64³ median 就够。

### C.8 Diagnostic auto-switch gate（Q1 addendum — 运行时 feature fallback）

**动机**: `z_final` (8-dim) 的描述力未经实证（§8.5）。MVP 先跑 z_final，但若实证弱必须自动切到 DiT block 18 hidden (1024-dim) 重跑。**不 a priori 预判，不手动判断** — 三个自动诊断指标 gate。

**诊断指标（C.4 之后计算）**:

```
lowe_pass_rate     = Σ_{(k,k')} |matches[k,k']| / Σ_k |move_cand_k|        # 匹配成功率
ransac_inlier_rate = Σ_{(k,k')} |inliers[k,k']| / Σ_{(k,k')} |matches[k,k']|  # 刚体一致率
bic_margin         = |BIC_rev - BIC_pris| / min(BIC_rev, BIC_pris)          # 类型判别显著性

trigger_hook = (lowe_pass_rate < 0.20)
            ∨ (ransac_inlier_rate < 0.10)
            ∨ (bic_margin < 0.05)
```

**动作**:

```
if trigger_hook:
    # 需要 Stage B 的 hook：加载 DiT block 18 hidden (~30 行 Stage B 改动)
    F = load_dit_hidden_features(stage_b_output, block=18)     # (K, 1024, 64, 64, 64)
    # 重跑 C.3 → C.4 → C.5 → C.6 → C.7
    rerun_pipeline(F)
else:
    # 当前结果 OK, 继续 C.5 / C.6 / C.7
    pass
```

**代码接口**:

- `configs/v1.yaml::stage_c.segmatch.feature_source`: `auto` | `z_final` | `dit_hidden`
- `auto`（默认）触发上述诊断 gate；`z_final` / `dit_hidden` 强制使用单一 source（ablation A6 / A10 用）

**日志**: `outputs/<sample>/stage_c/diagnostics.json`，包含三指标值 + 触发决策 + 重跑次数 + 最终 feature source。

**成本估算**: auto 触发重跑整条 C.3-C.7（~5-10 秒），加载 hidden feature（~1 秒）。若第一次 z_final 就 OK，只多几毫秒算诊断。

**为何选 block 18**: TRELLIS SS flow 24 层；DIFT 经验中后层（~70% 深度）semantic correspondence 最好；M_attn_64 本身就是从 DiT 某 block 的 self-attn 算的，信号有效已证明。block 选择作 ablation A6 扫 {8, 12, 16, 20}。

---

## 5. 三个 ablation 升级（MVP 后补）

### 5.1 Pose graph / Karcher mean on SE(3)

**动机**: MVP 用 pair (0, k) 的 RANSAC-Kabsch 直接得 T_k，等同于把 state 0 当 matching anchor（虽然 canonical origin 已经解耦，但 matching pair 仍偏重 state 0）。Pose graph 用全部 15 对 `T_{k→k'}` 做 SE(3) 流形上的一致性求解。

**算法（Govindu 2004 motion averaging, Ceres / theseus-ai 实现）**:

```
# 目标
min_{T_0, ..., T_{K-1}} Σ_{(k, k')} d_SE3(T_k' · T_k^{-1}, T_{k→k'})²

# Gauge constraint
T_0 = identity rotation, t_0 = base_centroid                # 锚定

# Optimizer: theseus-ai 的 SE3 group + Levenberg-Marquardt
import theseus as th
T_vars = [th.SE3(name=f"T_{k}") for k in range(K)]
for (k, kp), T_rel in relative_Ts.items():
    cf = th.CostFunction(...)        # d_SE3 residual
    objective.add(cf)
theseus_layer = th.TheseusLayer(th.LevenbergMarquardt(objective))
T_consistent, _ = theseus_layer.forward({})
```

**依赖**: `theseus-ai >= 0.2.1`（Meta PyTorch-native，Ubuntu/Linux 原生支持；已确认环境 OK）。

**替代**（若 theseus 安装/API 问题）: 手写 5 行 Karcher mean on SE(3)：

```
T_avg = T_0
for _ in range(5):                                          # iteratively reweighted averaging
    deltas = [logm(T_k_rel · T_avg^{-1}) for T_k_rel in T_k_rel_list]
    mean_delta = mean(deltas)
    T_avg = expm(mean_delta) · T_avg
```

### 5.2 λ_smooth 从 M_attn bimodality 自适应

**动机**: MVP 手调 `λ_smooth = 1.0` 是 magic number。AAAI 审稿人会问"为啥这个值"。

**自适应公式**:

```
# Bimodality index (Sarle's 1936): ρ = (μ₃² + 1) / (μ₄ + 3·(N-1)²/((N-2)·(N-3)))
hist = histogram(M_attn_64.flatten(), bins=50)
ρ    = Sarle_bimodality(hist)                              # ρ ∈ [5/9, 1]; higher = more bimodal

λ_smooth = λ_0 / (ρ - 5/9 + ε)
                                                           # ρ 高（M_attn 很清晰）→ λ 小（少 smoothness）
                                                           # ρ 低（M_attn 灰带多）→ λ 大（强 smoothness）
```

**ablation table**:
- fixed λ_0 = 0.5, 1.0, 2.0
- adaptive λ = λ_0 / (ρ - 5/9 + ε)

### 5.3 128³ upsample warp

**动机**: 64³ aliasing（target.md 根因 B）。

```
for k in 0..K-1:
    O_k_move_64  = O_k ∩ move_assigned
    O_k_move_128 = trilinear_upsample(O_k_move_64, 64→128)
    back_k_128   = trilinear_warp(O_k_move_128, T_k^{-1}, resolution=128)
    back_k_64    = avgpool(back_k_128, 128→64)

canonical_move = median_k(back_k_64) > 0.5
```

**代价**: 8× memory & compute，但只 K=6 次，一次 ~1 秒，可接受。

### 5.4 Seed-based anchored matching（Q2 addendum）

**动机**: state 0 是真实图像（非 Stage A 视频合成），其 move segmentation 更精准；MVP 主干的 pairwise mutual matching 对每个 state 独立做 `O_k ∩ (M_attn<τ_move)` 阈值，没偏重 state 0 的精准度。本 ablation 测是否把 state 0 的高置信 move seg 作 seed 注入 matching 能提升精度。

**算法**:

```
# 从 state 0 高置信 move voxel 作 seed
seed_0 = O_0 ∩ (M_attn_64 < τ_move) ∩ (mean_k O_k > 0.5)     # state-0 高置信真 move

# 对每个 seed 在每 state k (k≠0) 里强制找对应（不要求 mutual-best, 放松 Lowe）
seed_matches = defaultdict(list)
for v_0 in seed_0:
    for k in 1..K-1:
        candidates = move_cand_k    # 不限 window
        costs = cost_feature(v_0, candidates)       # 8/1024-dim cosine + spatial
        top_idx = argmin(costs)
        second_idx = second_argmin(costs)
        if costs[top_idx] / costs[second_idx] < τ_lowe_relaxed:    # τ_lowe_relaxed = 0.95
            seed_matches[(0, k)].append((v_0, candidates[top_idx]))

# 合并进主 matches（union, 不 overwrite）
for (k, k'), pairs in seed_matches.items():
    matches[(k, k')] = matches[(k, k')] + pairs     # RANSAC 后续统一过滤重复
```

**ablation 对比**:
- **A9.a** pure pairwise mutual (MVP 主干)
- **A9.b** pure seed-based from state 0
- **A9.c** pairwise ∪ seed-based (union, 预期最优)

**预期**:
- 若 state 0 真的比其他 state 准：A9.c 在 axis error / 类型判别上有 5-10% 提升
- 若 A9.c 无提升：证明 pairwise mutual 已充分利用了 state 0 信息，seed anchoring 冗余
- 若 A9.c 反而变差：TRELLIS 生成的 state 1-5 里 M_attn<τ_move 区域反而比 state 0 纯（可能因 Stage B v4.3 BMCSA 在 non-state-0 上效果强）→ 这本身是有价值的负面结论

---

## 6. 代码目录（新增，不动 `sajo/`）

```
mine/pipelines/stage_c_segmatch/
├─ __init__.py
├─ config.py             # dataclass: SegMatchHParams
├─ partition.py          # C.1 候选分区 + base_centroid
├─ features.py           # C.2 z_final trilinear upsample + L2 normalize
├─ matching.py           # C.3 pairwise mutual-best NN + Lowe ratio
├─ rigid_fit.py          # C.4 RANSAC-Kabsch + joint-constrained + BIC
├─ pose_graph.py         # (ablation) §5.1 SE(3) consistency via theseus
├─ graph_cut.py          # C.5 2-label α-expansion with logit unary
├─ axis_refine.py        # C.6 principal-axis-constrained Adam
├─ aggregation.py        # C.7 median warp (+ §5.3 128³ upsample ablation)
├─ run_stage_c.py        # 端到端 driver（读 stage_b/ 产物，写 stage_c/ 产物）
└─ tests/
   ├─ test_partition.py            # τ thresholds, centroid
   ├─ test_matching.py             # mutual-best, Lowe ratio
   ├─ test_rigid_fit.py            # Kabsch 恢复已知 T，RANSAC outlier 拒绝
   ├─ test_graph_cut.py            # logit numerical stability, 2-label on toy
   ├─ test_axis_refine.py          # contact principal axis 收敛
   ├─ test_aggregation.py          # median vs OR vs max, aliasing
   ├─ test_pose_graph.py           # (ablation) theseus SE(3) one-step
   └─ test_e2e_30857.py            # 端到端 30857_421 产物
```

**现有代码复用**:
- `pipelines/sajo/warp.py::trilinear_warp` / `batch_trilinear_warp` — 直接 import
- `pipelines/sajo/screw.py::exp_se3` / `project_revolute` / `project_prismatic` — 直接 import
- `pipelines/sajo/bic.py` — BIC 结构复用，但调用点改成 SegMatch 的 residual（不是 SAJO EM 的 L_data）

**新增依赖**:
- `theseus-ai >= 0.2.1`（ablation §5.1，Ubuntu/Linux，已确认 OK）
- `PyMaxflow >= 1.2.13`（C.5 graph-cut，pure Python pip）

---

## 7. 配置 schema（`configs/v1.yaml` 追加）

```yaml
stage_c:
  selector: segmatch           # segmatch | sajo（legacy ablation）
  segmatch:
    # C.1
    tau_move: 0.3
    tau_high: 0.7
    # C.2
    feature_source: auto       # auto | z_final | dit_hidden (§C.8 gate)
    feature_l2norm: true
    dit_block: 18              # used if feature_source=dit_hidden or gate triggers
    # C.3
    lambda_spat: 0.3
    tau_lowe: 0.88             # 8-dim feature 下偏大, §8.1
    # C.4
    ransac_iters: 200
    tau_ransac: 0.0317         # 2/63 voxel, 世界坐标
    max_fit_iters: 30
    adam_lr_axis: 1.0e-2
    adam_lr_phi: 5.0e-3
    # C.5
    lambda_data: 2.0
    lambda_smooth: 1.0
    lambda_smooth_adaptive: false     # ablation §5.2
    logit_eps: 1.0e-3
    # C.6
    lambda_axis: 0.5
    w_axis_dir: 1.0
    w_axis_pass: 0.5
    axis_refine_iters: 20
    # C.7
    warp_resolution: 64               # 64 | 128 (ablation §5.3)
    aggregator: median                # median | max | or (ablation)
    # C.8 auto-switch gate thresholds
    gate_lowe_pass_min: 0.20
    gate_ransac_inlier_min: 0.10
    gate_bic_margin_min: 0.05
    # §5.1 ablation
    use_pose_graph: false
    # §5.4 ablation
    seed_based_matching: false        # false | seed_only | union
```

---

## 8. 风险与 footnote（诚实列出）

### 8.1 Lowe ratio test @ 8-dim（实证调参）

- SS latent 8 通道比 DIFT 1280/SD-DINO 1792 通道密得多，NN 距离分布紧
- 经典 τ_lowe = 0.7-0.8（SIFT 128-dim）在 8-dim 下过度过滤
- MVP 取 τ_lowe = 0.85-0.90，实验扫描
- **ablation**: τ_lowe ∈ {0.75, 0.85, 0.88, 0.92, 0.95}
- Paper footnote 诚实说明

### 8.2 Pairwise matching 算力

- K=6 → 15 pairs
- 每 pair: N_k × N_{k'} ≈ 5K × 5K = 25M cosine sim
- Total: 15 × 25M = 375M float sim + argmin
- **torch.cdist @ GPU**: 8-dim float32，~2 秒总共 ✓ 可接受
- 不引 FAISS（Ubuntu 下 pip 装 faiss-gpu 偶尔有依赖冲突）

### 8.3 BIC i.i.d 假设弱

- 15 pairwise residuals 不独立（一个 voxel 可出现在多 pair 中作不同 role）
- BIC 的 `N = Σ|matches|` 偏乐观（有效样本数 < 表面数）
- **但**: revolute vs prismatic 的相对 ranking 稳定（两者都吃同样的 dependence）
- Paper methods footnote:
  > "Our BIC uses the cumulative residual over all pairwise matches as a likelihood proxy. The matches are not strictly i.i.d. across pairs, so the absolute BIC values may be optimistic; however, the relative ranking (revolute vs prismatic) is robust, which is the only use here."

### 8.4 K=6 RANSAC 统计显著性

- RANSAC 的 inlier count 统计显著性在 sample pool 小时弱（Vidal-Ma-Sastry GPCA 要求 F ≥ 10）
- 每 pair pool 是 ~几千 match pair（N_k × N_{k'} 里 mutual-best 后可能几百到几千），不算极限
- **mitigation**:
  - 低 inlier 率（<5%）时 pair-wise hypothesis 标 degenerate
  - 用 pose graph (§5.1) 把 15 pair 信息融合
  - Fallback: median 聚合所有 pair 的 T 估计（Karcher mean on SE(3)）

### 8.5 z_final 8-dim 描述力未验证（→ auto-gate §C.8 处理）

- SS VAE `lambda_kl=0.001` 近 AE，8-dim 理论上信息密度高，但经验未验证
- **不 a priori 预判，不手动判断**：§C.8 的诊断 auto-switch gate 根据 `lowe_pass_rate` / `ransac_inlier_rate` / `bic_margin` 三指标**自动**触发切 DiT block 18 hidden (1024-dim) 重跑
- Hook 成本: Stage B 加 ~30 行 hook 保存指定 DiT block 的 hidden state，一次性改动
- **已有锚点**: DIFT (1280-dim)、SD-DINO (1792-dim)、Diff3F (distilled on 3D shapes) 都 >> 8 维；更重要的是 **M_attn_64 本身就是从 DiT 某 block 的 self-attn feature cross-state agreement pooling 出来的 scalar**（见 stageb_detail.md §7），证明 DiT hidden 里有可用的 correspondence 信号，只是 8-dim SS latent 是否压缩足够 preserve 这个信号是 empirical 问题
- Hook block 选择作 ablation A6 扫 {8, 12, 16, 20}

### 8.6 Multi-joint 扩展（AAAI 版本外）

- 当前 2-label graph-cut 是 (base, move) 单 part
- Multi-joint 需 (base, move_1, ..., move_P) 多 label + sequential RANSAC 发现 P
- 扩展路径清晰但不在 MVP scope；supplementary 讨论
- Kinematic tree: star only（base 是所有 part 父节点）；嵌套 tree 作 future work

### 8.7 Canonical frame gauge 的隐含偏向

- MVP: `T_0 = I` 意味着 canonical_move 的相位 = state 0 的 move 相位
- state 0 的 move 本身也被 TRELLIS 缩小 bias 污染（同 Stage B 问题）
- **但**: C.7 的 `median_k(back_k)` 实际上是 median over 6 个 state 回 canonical 的 occupancy，state 0 只贡献 1/6，outlier 被 median 吸收
- Pose graph (§5.1) 进一步把 gauge 分布到所有 state，彻底解耦

### 8.8 Graph-cut data term 的鸡生蛋

- C.5 的 `data_term(v) = (1/K) Σ_k O_k(T_k(v))` 需要 T_k
- C.4 在 C.5 之前算出 T_k，所以不是鸡生蛋
- 但 T_k 在 C.6 之后还会精修；迭代的话 C.5 应该重跑
- **MVP**: C.4 → C.5 → C.6 → C.7 单 pass（不迭代）；精修后的 T_k 只用于 C.7 aggregation
- **可选**: C.4 → C.5 → C.6 → C.5' → C.7（迭代 1 次），ablation 测边际收益

---

## 9. 实验计划

### 9.1 MVP phase（4/23 - 4/27, 5 天）

| 日 | 任务 | 产物 |
|---|---|---|
| 4/23 | 骨架 + C.1-C.2 + unit tests | `partition.py`, `features.py`, tests 绿 |
| 4/24 | C.3 matching + C.4 RANSAC-Kabsch + BIC | `matching.py`, `rigid_fit.py`, tests 绿 |
| 4/25 | C.5 graph-cut + C.6 axis refine | `graph_cut.py`, `axis_refine.py`, tests 绿 |
| 4/26 | C.7 + e2e driver | `aggregation.py`, `run_stage_c.py`, toy e2e 绿 |
| 4/27 | 跑 30857, 26525, 7128, 7201 四 sample | output viz，diagnose |

**MVP 成功判据**:
- 4 sample 全跑通不崩
- 3/4 sample：joint_type 正确（human-eye）；轴方向 < 15° 误差（human-eye on viz）
- canonical_move viz 里没有明显 base 侵占或 move 缺失
- `per_state_assignment` 和 M_attn_64 在 high-conf voxel 上一致率 > 95%

### 9.2 Ablation phase（4/28 - 5/10, 13 天）

| ablation | 对比项 | 估计时间 |
|---|---|---|
| A1. τ_lowe 扫参 | 0.75 / 0.85 / 0.88 / 0.92 / 0.95 | 2 天 |
| A2. λ_smooth fixed vs adaptive (§5.2) | 0.5 / 1.0 / 2.0 / adaptive | 1 天 |
| A3. Pose graph on/off (§5.1) | per-pair T_k vs pose-graph T_k | 2 天 |
| A4. Aggregator | median / max / OR | 1 天 |
| A5. Warp resolution (§5.3) | 64³ / 128³ | 1 天 |
| A6. Feature source | z_final (8d) / DiT block {8, 12, 16, 20} hidden (1024d) | 3 天（需 Stage B hook）|
| A7. BIC 相对 ranking 稳定性 | Bootstrap 采样 | 1 天 |
| A8. SegMatch v3 vs SAJO EM | 关节误差、Chamfer | 2 天 |
| A9. Seed-based anchored matching (§5.4) | pure pairwise / pure seed / union | 1 天 |
| A10. Auto-switch gate (§C.8) | auto / fixed z_final / fixed DiT | 1 天 |

### 9.3 Benchmark phase（5/11 - 5/25, 15 天）

- PartNet-Mobility subset: 2 revolute + 2 prismatic 起步（已有 4 sample），扩到 10-20
- PARIS dataset: 2-state articulated objects（我们 K=6 应超越 K=2 baseline）
- **Metrics** (per 样本):
  - `axis_angle_error`: acos(|omega · omega_gt|) in deg
  - `axis_position_error`: dist(axis_line, axis_gt_line) in voxel
  - `joint_param_error`: mean |phi_k - phi_k_gt| / gt_range
  - `per_part_chamfer`: move mesh vs gt mesh
  - `pybullet_validation_rate`: 无 self-penetration rate (from Stage F)
- **Baselines**:
  - SAJO EM（我们的旧版本）
  - PARIS (2-state，仅 0 / K-1)
  - FreeArt3D + NAP 关节拟合（若能跑通）

### 9.4 Paper buffer（5/26 - 6/1, 7 天）

- 实验表格整理
- 消融结论写入 methods 附录
- Failure case 整理（哪些物体失败，为什么）
- Reviewer 问题预演

---

## 10. AAAI 表述（锁死措辞）

### 10.1 Contribution bullet（仅 1 条写进 Contributions 段）

> **Cross-stage semantic prior propagation.** We propose to use the base/move discrimination mask M_attn produced by the preceding 3D diffusion stage as a principled unary prior in a graph-cut formulation for rigid body segmentation. This establishes a structured information channel between the generation stage (Stage B) and the articulation-inference stage (Stage C), avoiding both the "variance-based split" pathology of prior EM-based articulation methods and the re-computation of semantic signals already encoded in the diffusion model.

### 10.2 Method paragraph（Stage C methods 节中，不进 Contributions）

> We use the frozen TRELLIS Structured Latent (SS) as an 8-channel per-voxel 3D descriptor for cross-state voxel correspondence. To our knowledge this is the first application of structured 3D diffusion latents to training-free articulated motion discovery, extending the diffusion-feature correspondence literature (DIFT, SD-DINO, Diff3F) from 2D images / static 3D shapes to the K-state articulated regime.

### 10.3 不吹（全是 standard technique，methods 里 mention 但 **不写 contribution**）

- Pairwise mutual-best NN matching（multi-view feature matching, classic）
- RANSAC-Kabsch（Kabsch 1976, Fischler-Bolles 1981）
- Pose graph averaging on SE(3)（Govindu 2004）
- Median canonical aggregation（robust statistics）
- 2-label α-expansion graph-cut（Boykov-Kolmogorov 2001/2004）
- Contact region principal axis constraint（target.md G3 的几何实现）

### 10.4 Stage C 在 paper 里的比重

- **Methods 节**: 1-1.5 页（占 methods 总长 20%，主力留给 Stage D/E 的纹理回灌）
- **Ablation**: 2-3 行（主要对比 variance split vs M_attn prior；median vs OR；RANSAC vs plain Kabsch）
- **Main table**: Stage C 单独列 axis angle / position error；但整体 pipeline 表里它是中间指标

---

## 11. 与现有代码的关系

| 文件/目录 | 状态 | 动作 |
|---|---|---|
| `pipelines/sajo/` | 现有 | **保留**，legacy ablation；`configs/v1.yaml::stage_c.selector: sajo` 可切回 |
| `pipelines/sajo/warp.py` | 现有 | 不动，`segmatch/` 直接 `from ..sajo.warp import ...` 复用 |
| `pipelines/sajo/screw.py` | 现有 | 不动，`segmatch/` 复用 `exp_se3`, `project_revolute`, `project_prismatic` |
| `pipelines/sajo/bic.py` | 现有 | 结构复用；SegMatch 里的 BIC 直接手写（不调用）避免耦合 |
| `pipelines/stage_c_sajo.py` | 现有 | 保留 legacy driver；新加 `pipelines/stage_c_segmatch.py` driver |
| `pipelines/stage_c_segmatch/` | **新增** | 本 spec 的全部实现 |
| `configs/v1.yaml` | 现有 | 追加 `stage_c.segmatch` 块；default `selector = segmatch` |
| `run_v1.py` / `run_stage_bc.sh` | 现有 | 追加 selector 分支 |
| `record/stageC/1-4.md` | 现有 | 保留，brainstorm 档案 |
| `方案.md` / `贡献和动机.md` | 现有 | **不改**；本 spec 单独 live，待用户汇总时再并 |

---

## 12. 下一步（立即）

- [ ] `pipelines/stage_c_segmatch/` 骨架（目录 + `__init__.py` + `config.py`）
- [ ] `partition.py` + `features.py` + 4 unit tests
- [ ] `matching.py` + 3 unit tests
- [ ] `rigid_fit.py` + RANSAC 恢复已知 T 的 unit test
- [ ] `graph_cut.py` + logit stability test + toy 2-label test
- [ ] `axis_refine.py` + principal-axis 收敛测试
- [ ] `aggregation.py` + median vs OR 对比测试
- [ ] `run_stage_c.py` + 30857 e2e 烟雾测试（加载 stage_b 产物、产 stage_c 产物）

**首个 milestone**: 30857 跑通，`viz/canonical_move.html` 人眼判断 OK，`joint_type` 对。

---

## 13. 附录 A — world coord 转换参考

沿用 [pipelines/sajo/warp.py:36-39](../../pipelines/sajo/warp.py:36)：

```
voxel idx (i, j, k) ∈ [0, 63]^3
world     p = (i, j, k) / 63 - 0.5 ∈ [-0.5, 0.5]^3
grid_sample 约定 (x, y, z) = last-dim 逆序 (W, H, D)，见 warp.py:93-94
```

## 14. 附录 B — 关键超参一览

| stage | 参数 | 默认 | 来源 |
|---|---|---|---|
| C.1 | τ_move | 0.3 | SegMatch 1.txt 原案 |
| C.1 | τ_high | 0.7 | 同上 |
| C.3 | λ_spat | 0.3 | MVP, ablation 扫 |
| C.3 | τ_lowe | 0.88 | 8-dim 下偏大，§8.1 |
| C.4 | RANSAC_ITERS | 200 | 标准值 |
| C.4 | τ_ransac | 2/63 ≈ 0.0317 | 2 voxel 世界坐标 |
| C.4 | max_fit_iters | 30 | Adam 收敛够 |
| C.4 | lr_axis | 1e-2 | sajo/em.py 同 |
| C.4 | lr_phi | 5e-3 | sajo/em.py 同 |
| C.5 | λ_data | 2.0 | MVP, ablation 扫 |
| C.5 | λ_smooth | 1.0 | MVP；§5.2 自适应 |
| C.5 | logit_eps | 1e-3 | 数值稳定 |
| C.6 | λ_axis | 0.5 | MVP |
| C.6 | w_axis_dir | 1.0 | MVP |
| C.6 | w_axis_pass | 0.5 | MVP |
| C.6 | axis_refine_iters | 20 | MVP |
| C.7 | warp_resolution | 64 | MVP；§5.3 上 128 |
| C.7 | aggregator | median | MVP；§5.3 ablation |
| C.8 | gate_lowe_pass_min | 0.20 | §8.5 经验值 |
| C.8 | gate_ransac_inlier_min | 0.10 | §8.5 经验值 |
| C.8 | gate_bic_margin_min | 0.05 | 类型判别稳定阈值 |
| §5.4 | tau_lowe_relaxed | 0.95 | seed-based 放松 Lowe |

---

_End of Stage C SegMatch v3 Spec._
