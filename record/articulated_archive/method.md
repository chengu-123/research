# method.md — 具体方法设计：TRELLIS 内部 RASA + Wan2.2 RFSDS + 跨状态纹理融合

版本：2026-04-26  
范围：本文件给出每个阶段的数学形式、模块接口、优化目标、实现细节、诊断指标与消融项。

---

## 0. 符号

输入：

- `I0`：单张闭态 RGB 图。
- `c_T`：TRELLIS image condition，通常来自 DINO/image encoder。
- `c_W`：Wan2.2 I2V image/text condition。
- `K`：TRELLIS 内部离散状态数，例如 4 或 6。
- `T`：Wan RFSDS 渲染视频帧数，例如 9、17、41。
- `s_k`：state code，`k = 0..K-1`，其中 `s_0` 是 closed state。
- `q_t`：连续时间视频帧的 joint state，`t = 0..T-1`。

TRELLIS SS-DiT：

- `x_τ^k`：第 k 个状态在 SS rectified-flow timestep τ 的 noisy latent。
- `H_l^k ∈ R^{L×C}`：第 l 个 block 的 hidden tokens，`L=16^3=4096`，`C≈1024`。
- `v_θ(x_τ, τ, c_T)`：冻结 TRELLIS SS-DiT 的 velocity prediction。

Canonical structure：

- `B`：canonical base。
- `M`：canonical movable part。
- `p_B(x)`：base soft occupancy / probability。
- `p_M(x)`：move soft occupancy / probability。
- `T(q)`：single-DoF transform。
- `p_k(x)`：第 k 个状态的 soft occupancy。

Texture / SLAT：

- `S_B, S_M`：canonical base/move SLAT features。
- `C_B, C_M`：base/move color / 3DGS appearance variables。
- `D(i)`：canonical surface token / Gaussian i 的可见 donor states。

---

## 1. RASA：RFSDS-optimized Articulated State Adapter

### 1.1 插入位置

RASA 插入冻结 TRELLIS SS-DiT 的 selected transformer blocks。

推荐初始 blocks：

```python
adapter_blocks = [8, 12, 16, 20]
```

或完整消融：

```python
[6, 12, 18]
[8, 12, 16, 20]
[4, 8, 12, 16, 20]
```

插入点优先级：

1. self-attention 后、cross-attention 前；
2. block output residual 处；
3. AdaLN modulation residual。

三者都要消融，但主实现建议先用 self-attention 后的位置。

### 1.2 输入输出接口

输入：

```python
H: Tensor[K, L, C]
state_code: Tensor[K, d_s]
timestep_embed: Tensor[K, d_t]
image_cond_summary: Tensor[d_c] or None
```

输出：

```python
Delta_H: Tensor[K, L, C]
part_logits_16: Tensor[K, L, 3]  # base / move / uncertain
part_slots: Tensor[K, P, C]
joint_params: dict
```

其中 `P=3`：base, move, uncertain。后续多部件扩展可设 `P = 1 + N_move + 1_uncertain`。

### 1.3 模块结构

#### Step 1：state-conditioned tokens

```text
H̃_k = H_k + E_state(s_k) + E_time(τ)
```

`E_state` 让同一输入图产生 closed → open 的状态差异。

#### Step 2：part-slot pooling

使用 learnable part queries：

```text
Q_part = {q_base, q_move, q_uncertain}
```

对每个状态：

```text
P_k = CrossAttn(Q_part, K=H̃_k, V=H̃_k)       # P_k ∈ R^{P×C}
```

得到每个状态的 base/move/uncertain part evidence。

#### Step 3：cross-state part attention

对同一个 part slot，跨 K 个状态做 attention：

```text
P̂_{:,j} = SelfAttn_state(P_{0:K-1,j})
```

目的：让模型在生成过程中理解“同一个部件在不同状态中如何对应”。

#### Step 4：scatter back to voxel tokens

```text
ΔH_k = CrossAttn(Q=H̃_k, K=P̂_k, V=P̂_k)
```

zero-init residual：

```text
H'_k = H_k + α_l * ΔH_k
```

`α_l` 初始化为 0 或非常小，保证初始等价于原 TRELLIS。

#### Step 5：part and joint heads

part logits：

```text
m_k = Linear(H'_k)  # K × L × 3
```

joint head：

```text
z_move = Aggregate(P̂_{:, move}, over states and selected blocks)

joint_type_logit = MLP_type(z_move)
axis_raw        = MLP_axis(z_move)       # R^3
axis            = axis_raw / ||axis_raw||
pivot           = MLP_pivot(z_move)      # R^3, revolute only
range_raw       = MLP_range(z_move)
range           = range_prior * sigmoid_or_softplus(range_raw)
q_k             = MonotonicHead(state_code_k, range)
```

当前 MVP：joint type 可由用户指定或 prompt 指定，joint head 只预测对应类型的参数。

---

## 2. SS-DiT velocity correction：让结构变量参与生成流

RASA 不能只是读 hidden。它必须参与 denoising。

冻结 TRELLIS 给出 velocity：

```text
v = v_TRELLIS(x_τ, τ, c_T)
```

rectified-flow clean prediction：

```text
x0_hat = x_τ - τ * v
```

RASA 从 hidden 预测结构化 clean latent / occupancy：

```text
x0_struct = ComposeSS(RASA(H), part_logits, joint_params, state_code)
```

混合：

```text
λ(τ) = schedule, e.g. small at high noise, larger near mid/low noise
x0_mix = (1 - λ(τ)) * x0_hat + λ(τ) * x0_struct
v_corr = (x_τ - x0_mix) / (τ + ε)
```

采样器使用 `v_corr` 更新 latent。

注意：

- 原始 TRELLIS 权重不变。
- RASA 改的是 velocity field 的新增结构化 residual。
- `λ(τ)` 需要消融：过大破坏 TRELLIS prior，过小 RASA 无效。

---

## 3. Articulated Soft SS Composition

### 3.1 从 token logits 到 64³ soft occupancy

RASA 输出 16³ token logits：

```text
m_B^16, m_M^16, m_U^16
```

可通过以下方式得到 64³ occupancy：

1. 经过 TRELLIS SS decoder 得到 logits；
2. 或 token logits trilinear upsample 到 64³ 作为 gating，再与 SS decoder logits 相乘/相加。

推荐形式：

```text
logit_B_64 = D_SS(z_B) + Up(m_B^16)
logit_M_64 = D_SS(z_M) + Up(m_M^16)
p_B = sigmoid(logit_B_64)
p_M = sigmoid(logit_M_64)
```

其中 `z_B, z_M` 来自 RASA 结构化 clean latent branch。

### 3.2 Single-DoF transform

#### Prismatic

```text
T(q)(x) = x + q * a
```

- `a`：unit axis。
- `q ∈ [0, q_max]`。

#### Revolute

```text
T(q)(x) = R(a, q)(x - o) + o
```

Rodrigues：

```text
R(a,q) = I + sin(q)[a]_x + (1-cos(q))[a]_x^2
```

- `a`：unit axis。
- `o`：pivot/origin。
- `q ∈ [0, θ_max]`。

### 3.3 Occupancy composition

第 k 个状态：

```text
p_k(x) = 1 - (1 - p_B(x)) * (1 - Warp(p_M, q_k)(x))
```

`Warp` 用 differentiable `grid_sample`。

这一步保证：

- base branch 在所有状态共享；
- move branch 只有通过 T(q_k) 变化；
- RFSDS gradient 能回到 part logits 与 joint variables。

### 3.4 Base / move hardening

训练中用 soft `p_B, p_M`。导出前：

```text
B_hard = p_B > τ_B
M_hard = p_M > τ_M
```

然后做：

- connected component filtering；
- remove tiny floating components；
- contact region preservation；
- no overlap or controlled overlap at joint/contact boundary。

---

## 4. Wan2.2 I2V-A14B RFSDS

### 4.1 Locked-camera prompt

正向 prompt 模板：

```text
A locked-off single-camera video. The first frame is exactly the provided image. 
The camera is static, with no zoom, no pan, and no orbit. 
The object body remains stationary. Only the [drawer/door/lid] [slides/rotates] open. 
The material, color, lighting, and texture remain consistent across all frames. 
Newly revealed interior surfaces are coherent with the object material.
```

negative prompt：

```text
camera movement, zoom, pan, orbit, object morphing, changing identity, 
new object, changing background, inconsistent texture, flickering, deformation, 
floating parts, non-rigid motion, body moving
```

### 4.2 渲染视频

Geometry phase：

```text
Render soft occupancy / coarse Gaussian / depth-normal-silhouette video
```

Texture phase：

```text
Render TRELLIS 3DGS / radiance field RGB video
```

固定相机，不优化 camera。

first-frame 处理：

- 推荐 first-frame latent clamp：第一帧 Wan latent 直接绑定输入图 VAE latent。
- 同时保留 photometric / DINO anchor，防止物体 identity 漂移。

### 4.3 RFSDS residual

渲染视频：

```text
V_φ = Render(B_φ, M_φ, T(q), texture_φ)
```

Wan VAE latent：

```text
z = E_WanVAE(V_φ)
```

rectified-flow noising：

```text
z_τ = (1 - τ) z + τ ε
```

冻结 Wan2.2 预测 velocity：

```text
v_hat = Wan2.2(z_τ, τ, image_cond=c_W_img, text_cond=c_W_txt)
```

RF target velocity：

```text
v_target = ε - z
```

RFSDS gradient residual：

```text
g = w(τ) * (v_hat - v_target)
  = w(τ) * (v_hat - ε + z)
```

surrogate loss：

```text
L_RFSDS_sur = < stopgrad(g), z >
```

反传路径：

```text
z → V_φ → renderer → texture / geometry / joint / RASA
```

通常不需要对 Wan2.2 本身求梯度；Wan 作为 frozen teacher forward。Wan VAE encoder 与 renderer 到变量的梯度需要保留。

### 4.4 τ schedule

几何阶段：

```text
τ ∈ high/mid noise range
```

偏向 large motion / layout / trajectory。

纹理阶段：

```text
τ ∈ low noise range
```

偏向 detail / material / temporal consistency。

可以采用 CHORD-style weighted / annealed τ schedule：早期大噪声，后期小噪声。

---

## 5. Losses

总目标：

```text
L = λ_rfsds L_RFSDS
  + λ_first L_first_frame
  + λ_base L_base_static
  + λ_part L_part_separation
  + λ_joint L_joint_prior
  + λ_motion L_motion_lower_bound
  + λ_tex L_texture_anchor
  + λ_reg L_adapter_reg
```

### 5.1 RFSDS loss

由 Section 4 给出，是主监督。

### 5.2 First-frame anchor

```text
L_first = || Render_frame0 - I0 ||_1
        + LPIPS(Render_frame0, I0)
        + DINO_distance(Render_frame0, I0)
```

如果使用 first-frame latent clamp，则这项可降低权重，但不应完全移除。

### 5.3 Base static loss

防止 base 获得 state-dependent residual：

```text
L_base_static = Σ_k || H_base^k - H_base^0 ||^2
```

或直接通过 architecture 禁止 base branch 使用 state-dependent transform。

### 5.4 Part separation

```text
L_overlap = mean(p_B * p_M)
L_entropy = entropy_regularization(part_probs)
L_size = volume_prior(move_volume / object_volume)
```

避免：

- move 吞掉整个物体；
- move 为空；
- base/move 大面积重叠。

### 5.5 Joint prior

```text
axis_unit: ||a|| = 1
q_monotonic: q_{t+1} >= q_t
range_bound: q_max within plausible range
single_dof: no dense deformation
contact_prior: move and base remain physically adjacent near joint/contact
collision_prior: optional sweep collision penalty
```

### 5.6 Motion lower bound

防止 RFSDS collapse 到静态视频：

```text
L_motion_lb = max(0, m_min - mean_pixel_motion_or_q_range)^2
```

MVP 可以直接用 `q_T - q_0`。

### 5.7 Texture losses

state0 visible：

```text
L_state0_tex = photometric / DINO / LPIPS against I0 visible region
```

cross-state texture consistency：

```text
L_tex_cons = Σ_{i,k,k'∈D(i)} || InvWarp(C_{k,i}) - InvWarp(C_{k',i}) ||
```

provenance sparsity：

```text
L_prov = encourage confident donor weights, but not collapse to invalid state
```

### 5.8 Adapter regularization

```text
L_adapter = Σ_l ||ΔH_l||^2 + ||adapter_weights||^2
```

防止新增层破坏 TRELLIS prior。

---

## 6. Optimization Schedule

### Phase 0 — Initialization / diagnostics

```text
iters: 0
run original TRELLIS for closed-state initialization
cache Wan image/text condition
optional Wan I2V sample for prompt verification
```

### Phase 1 — Geometry and trajectory

```text
iters: 0..N_geom, e.g. 1200-1600
optimize: SS-RASA, part logits, joint vars, small SS residual
freeze: texture / SLAT color residual
render: silhouette/depth/normal/simple albedo
τ: high/mid noise
```

Purpose：让 RFSDS 主要修改 segmentation / joint / large geometry。

### Phase 2 — Boundary refinement

```text
iters: N_geom..N_topology, e.g. 200-400
optimize: narrow-band part logits, joint fine-tune
freeze: high-confidence base core and move core
render: better soft occupancy / coarse Gaussian
τ: mid/low noise
```

Purpose：细化 move/base 边界。

### Phase 3 — Topology hardening

```text
soft p_B,p_M → hard coords_B, coords_M
connected components
filter floating noise
fix topology for SLAT
```

No continuous gradient claim through this step.

### Phase 4 — SLAT / texture refinement

```text
iters: N_tex, e.g. 400-800
optimize: SLAT-RASA, texture residual, 3DGS color/SH, donor weights
freeze: main geometry and joint, or use very low LR
render: RGB 3DGS/radiance video
τ: low noise
```

Purpose：优化跨状态可见纹理与新暴露区域。

### Phase 5 — Export

```text
decode mesh
generate URDF
run simulator diagnostics
save provenance
```

---

## 7. Cross-State Visible Texture Fusion

### 7.1 Donor states

对 canonical surface / Gaussian / SLAT token i：

```text
D(i) = { k | i 在 state k 的固定相机视角可见 }
```

可见性由 renderer 给出：depth test、normal facing、alpha contribution。

### 7.2 Provenance classes

```text
P0: state0_visible
P1: open_state_exposed
P2: never_visible
```

训练与报告必须区分三类。

### 7.3 Fusion formula

```text
w_{k,i} = softmax(
    β1 * visibility_{k,i}
  + β2 * view_quality_{k,i}
  + β3 * part_consistency_{k,i}
  + β4 * rfsds_confidence_{k}
  + β5 * state0_priority_{k,i}
)

C_i_can = Σ_{k∈D(i)} w_{k,i} * InvWarp(C_{k,i}, q_k)
S_i_can = Σ_{k∈D(i)} w_{k,i} * InvWarp(S_{k,i}, q_k)
```

### 7.4 Texture variables

可优化：

```text
SLAT feature residual ΔS_i
3DGS color / SH residual ΔC_i
roughness/metallic proxy if available
fusion weights w_{k,i}
```

不可大幅优化：

```text
main geometry
joint axis/range
base/move identity
```

---

## 8. Differentiability Policy

### 8.1 可导

```text
RASA adapters
SS-DiT velocity correction
SS decoder logits
soft occupancy composition
grid_sample warp
soft renderer / 3DGS renderer
Wan VAE encoder to rendered pixels
SLAT features with fixed coords
3DGS / radiance field appearance
```

### 8.2 不声称连续可导

```text
threshold p > τ
argwhere active coords
connected components
mesh simplification
URDF export
hard simulator validation
```

### 8.3 工程策略

- Geometry phase 用 soft SS 避开 `argwhere`。
- Topology hardening 作为离散阶段。
- Texture phase 固定 coords，从而 SLAT/3DGS 可连续优化。
- STE/Gumbel top-k 只作为 ablation，不作为主线依赖。

---

## 9. Implementation Details

### 9.1 Suggested hyperparameters

```yaml
K_states: 4 or 6
T_video_frames: 9 initially, then 17 or 41
adapter_blocks_ss: [8, 12, 16, 20]
adapter_blocks_slat: [8, 12, 16, 20]
part_slots: [base, move, uncertain]
optimizer: AdamW
lr_adapter: 1e-4 to 3e-4
lr_joint: 1e-3
lr_texture: 1e-3 to 5e-3
adapter_weight_decay: 1e-4
rfsds_iters_geom: 1200-1600
rfsds_iters_texture: 400-800
render_res_geom: 256
render_res_texture: 480p-compatible crop/resize
```

### 9.2 Checkpoint outputs

```text
checkpoints/
  iter_0000/
  iter_0200/
  iter_0500/
  iter_1000/
  iter_final/
```

Each checkpoint：

```text
adapter.pt
part_logits.npy
joint.json
rendered_video.mp4
p_base.npy
p_move.npy
rfsds_loss.json
first_frame.png
```

### 9.3 Required diagnostics

```text
1. first frame vs input
2. fixed camera check
3. q_t monotonic curve
4. base/move volume curves
5. axis / pivot trajectory
6. RFSDS residual vs iteration
7. texture donor maps
8. final URDF actuation video
```

---

## 10. Ablations

### 10.1 Adapter location

```text
no RASA
RASA post-self-attn
RASA post-cross-attn
RASA block-output
RASA AdaLN residual
```

### 10.2 RFSDS

```text
no RFSDS
vanilla RFSDS uniform τ
CHORD-style weighted / annealed τ
high-noise only
low-noise only
geometry phase high/mid + texture phase low
```

### 10.3 Structural variables

```text
no explicit base tie
explicit base tie
no joint bottleneck, dense deformation
single-DoF prismatic/revolute
wrong joint type initialized
```

### 10.4 Texture

```text
single closed-state texture only
TRELLIS SLAT texture only
Wan low-noise RFSDS texture
cross-state visible texture fusion
```

### 10.5 Differentiability

```text
soft SS renderer
periodic topology refresh + fixed SLAT coords
straight-through argwhere
Gumbel top-k coords
```

---

## 11. Minimal MVP

MVP 不做完整 URDF 之前，必须先验证：

```text
Input: one cabinet/drawer closed image
Joint type: prismatic manually specified
RASA: SS only, no SLAT texture
Renderer: soft occupancy silhouette/depth/simple gray
Wan: I2V-A14B, locked camera prompt
Optimization: 500-1000 iters
Success:
  - move mask localizes drawer front/body correctly
  - q_t increases monotonically
  - base branch stays static
  - rendered video shows drawer-like motion without camera movement
```

如果 MVP 失败，不要进入 texture/SLAT。

---

## 12. Paper-ready Claims and Non-claims

### 12.1 可以主张

- Frozen TRELLIS + frozen Wan2.2 双 prior。
- TRELLIS 内部新增可优化 articulated adapter。
- 单闭态图输入。
- RFSDS gradient 通过渲染链路优化 part assignment、single-DoF trajectory、texture variables。
- Base/move identity 由内部结构变量绑定。
- Cross-state visible texture fusion。

### 12.2 不应主张

- Wan2.2 直接提供 segmentation label。
- never-visible texture 是真实恢复。
- 官方 TRELLIS `argwhere` topology path 天然端到端可导。
- 不需要 prompt / camera constraint。
- 可以无约束优化原始 TRELLIS attention 权重。

---

## 13. File-level Implementation Plan

建议新增：

```text
pipelines/stage_b_articulated.py
pipelines/rasa/ss_adapter.py
pipelines/rasa/slat_adapter.py
pipelines/rasa/joint_head.py
pipelines/rasa/part_head.py
pipelines/render/soft_voxel_renderer.py
pipelines/render/gaussian_video_renderer.py
pipelines/wan/rfsds.py
pipelines/texture/visible_texture_fusion.py
pipelines/export/urdf_exporter.py
configs/articulated_stage_b.yaml
record/articulated_stage_b/pipeline.md
record/articulated_stage_b/method.md
```

保持旧 `stage_b_scar.py` 作为 baseline 与 initialization infrastructure，不要直接覆盖。

---

## 14. References

- TRELLIS project / paper: https://microsoft.github.io/TRELLIS/ ; https://github.com/microsoft/TRELLIS ; https://arxiv.org/abs/2412.01506
- Wan2.2 official repo / I2V-A14B model card: https://github.com/Wan-Video/Wan2.2 ; https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- CHORD / RFSDS: https://yanzhelyu.github.io/chord/ ; https://arxiv.org/abs/2601.04194
- FreeArt3D: https://arxiv.org/abs/2510.25765
- PAct: https://arxiv.org/abs/2602.14965
