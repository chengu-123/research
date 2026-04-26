# pipeline.md — 单图 TRELLIS 内部可优化部件生成 + Wan2.2 RFSDS 总体流程

版本：2026-04-26  
目标会议标准：AAAI/CVPR 级别，优先保证可证伪、可复现、可消融，而不是堆叠模块。

---

## 0. 核心目标

从**单张闭态输入图像**生成一个可仿真的 articulated 3D asset：

```text
input.png
  → canonical_base.{mesh, slat, gaussian}
  → canonical_move.{mesh, slat, gaussian}
  → joint.json / object.urdf
  → fixed-camera opening video diagnostics
  → texture provenance / cross-state visible texture map
```

当前阶段先限制为：

- 单输入图：只有 closed-state image；没有真实 open-state 图。
- 单可动部件：single-DoF，joint type ∈ {prismatic, revolute}。
- 冻结原始 TRELLIS 与 Wan2.2；只优化我们新加入的网络层与 per-instance 显式变量。
- Wan2.2 使用 I2V-A14B，并通过固定单视角 prompt 约束视频：首帧等于输入图，camera locked，base 不动，只有目标部件运动。

---

## 1. 设计原则

### 1.1 不再走旧 StageB+StageC 后处理路线

旧路线本质是：

```text
TRELLIS / StageB 生成多状态 occupancy
  → 读取 hidden / attention / z_final
  → StageC 后验分割部件
  → 后验拟合 prismatic / revolute 轨迹
```

这不再作为主方法。它可以保留为 baseline / initialization / diagnostic，但不作为论文主线。

新路线是：

```text
single image condition
  → K-state coupled TRELLIS SS-DiT generation
  → inserted cross-state part-aware adapter inside SS-DiT
  → adapter directly emits base/move/trajectory variables
  → differentiable render video
  → Wan2.2 I2V-A14B W/RFSDS gradient
  → update inserted adapter + part logits + joint + SLAT/texture variables
```

核心区别：部件分割和轨迹不是从生成结果中后验拟合出来，而是作为 TRELLIS sparse-structure / SLAT 生成内部的显式变量被 Wan2.2 RFSDS 迭代优化。

### 1.2 Wan2.2 不是 segmentation label teacher，但可以优化 segmentation / trajectory / texture

Wan2.2 不输出 voxel part labels。它提供的是 fixed-camera image-to-video rectified-flow prior。

如果部件分割错误、joint axis 错误、prismatic/revolute 类型错误、open-state 暴露纹理不合理，则我们渲染出来的视频会偏离 Wan2.2 在输入图与 locked-camera prompt 条件下的高概率视频流形。RFSDS residual 可以沿下面路径回传：

```text
Wan2.2 RF residual
  → Wan VAE latent z
  → rendered video pixels
  → differentiable renderer
  → SLAT / Gaussian / texture variables
  → warped base/move geometry
  → part logits and joint variables
  → inserted TRELLIS adapter parameters
```

因此 Wan2.2 的作用是：

- motion prior：什么部件应该运动，如何运动更像真实 opening video；
- fixed-view temporal prior：camera 不动、base 不动、只有 move 部件动；
- appearance / texture prior：后续帧新暴露区域的颜色、材质、细节、时序一致性；
- 不是 explicit part label 或 ground-truth geometry。

### 1.3 “完美 base / only move”必须由内部参数绑定保证

RFSDS 本身不会保证 base bit-identical。必须在我们插入的 TRELLIS 内部结构里保证：

- base branch 是 canonical and shared；
- move branch 是 canonical and warped by a single-DoF transform；
- 每个状态的 SS / SLAT 由同一套 base 与 move 变量组合得到。

这不是 StageC 后处理投影，因为这些变量参与 SS-DiT/SLAT-DiT 的生成和 RFSDS 优化。

---

## 2. 上游模型事实与约束

### 2.1 TRELLIS

TRELLIS 是两阶段 3D-native 生成模型：

1. Sparse Structure stage：生成稀疏结构 / occupancy。
2. SLAT stage：在 active sparse voxels 上生成 Structured LATent，可解码为 mesh、3D Gaussian、radiance field。

TRELLIS 使用 rectified-flow transformers，SLAT 支持 geometry 与 texture/appearance。官方说明强调 SLAT 可以解码为 Radiance Fields、3D Gaussians、meshes，并使用 rectified flow transformers 作为 backbone。

实现约束：

- SS-DiT 的 token 空间是 16³，hidden dim 通常为 1024，与你当前 StageB 保存的 `dit_hidden.pt` 维度一致。
- 官方 `SS decoder → threshold/argwhere → active coords` 是 topology hard selection，不可把它声称为天然端到端可导。
- 在优化 loop 中，应使用 soft SS / fixed topology / periodic topology refresh，而不是宣称硬 argwhere 可连续反传。

### 2.2 Wan2.2 I2V-A14B

Wan2.2 I2V-A14B 是 image-to-video 模型，A14B 系列采用 MoE：高噪声 expert 偏整体 layout / large motion，低噪声 expert 偏细节 / texture refinement。I2V 条件把输入图像经 VAE 编码作为视频生成条件，适合我们用单张闭态图约束首帧。

使用原则：

- 冻结 Wan2.2；不训练 Wan。
- 用 locked-camera prompt 限制输出视频。
- 先用 high/mid noise 优化 geometry / trajectory；后用 low noise 优化 exposed texture 与 temporal detail。

### 2.3 CHORD / RFSDS

CHORD 的关键启发是：用 rectified-flow video model 的 velocity residual 作为 4D representation 的优化梯度。我们借用这个思想，但目标不同：

- CHORD：静态 3D/4D 表示 → 动态场景。
- 本项目：单图 TRELLIS 内部部件变量 + single-DoF joint + SLAT/texture → URDF-ready articulated asset。

---

## 3. 总体 pipeline

## Stage 0 — 输入、条件与初始化

输入：

```text
input.png
motion_prompt，例如：
  "A locked-off single-camera video. The first frame is exactly the provided image. 
   The camera is static. The cabinet body remains stationary. 
   Only the drawer slides open. The material and lighting stay consistent."
joint_type prior: prismatic 或 revolute，可先手动指定
K: discrete TRELLIS states, e.g. 4 或 6
T: rendered video frames for Wan RFSDS, e.g. 9/17/41
```

预处理：

1. 用 TRELLIS image encoder 提取 DINO/image condition。
2. 用 Wan2.2 I2V encoder/cache 得到 image condition 与 text condition。
3. 初始化 K-state latent：同一输入图、shared noise、不同 state code。
4. 初始化 joint：
   - prismatic：axis、range；
   - revolute：axis、pivot/origin、angle range；
   - 状态值 `q_k` 不要全为 0，避免 pivot / range 梯度退化。
5. 可选：先运行 Wan2.2 I2V 采样一条 fixed-camera opening video，仅用于初始化与诊断，不作为 ground truth。

输出：

```text
init/
  cond_trellis.pt
  cond_wan.pt
  state_codes.pt
  joint_init.json
  prompt.txt
```

---

## Stage 1 — TRELLIS SS-DiT 内部 RASA Adapter

新增模块：RASA = RFSDS-optimized Articulated State Adapter。

插入位置：SS-DiT 若干中后层 block 内，优先在 self-attention 后、cross-attention 前，或 block output residual 处做消融。

输入张量：

```text
H_l: K × L × C
K = number of states
L = 4096 tokens for 16³ SS token grid
C = 1024 hidden dim
state_code: K × d_s
image_cond: DINO / TRELLIS condition
```

RASA 输出：

```text
base token features / logits
move token features / logits
uncertain token logits
single-DoF joint variables: type, axis, pivot, range, q_k
adapter residual ΔH_l
```

基本结构：

```text
voxel tokens H_l
  → part-slot pooling: base slot / move slot / uncertain slot
  → cross-state attention over same part slot
  → scatter part information back to voxel tokens
  → zero-init residual adapter ΔH_l
  → part head + joint head
```

关键约束：

- 原始 TRELLIS SS-DiT 权重冻结。
- RASA zero-init，使 iter 0 等价于原 TRELLIS。
- RASA 不是后验 hidden analysis；它直接影响 denoising token dynamics 和结构变量。

输出：

```text
stage1_rasa/
  adapter_state.pt
  part_logits_16.pt
  joint_pred.json
  ss_latents_structured.pt
  diagnostics/part_slots_*.npy
```

---

## Stage 2 — Articulated Soft Sparse Structure Generation

RASA 直接产生 canonical base 与 canonical move 的 soft SS 表示：

```text
p_B: canonical base occupancy probability / logits
p_M: canonical movable part occupancy probability / logits
T(q_k): prismatic or revolute differentiable transform
p_k = union(p_B, warp(p_M, q_k))
```

建议使用 noisy-OR 形式：

```text
p_k(x) = 1 - (1 - p_B(x)) * (1 - Warp(p_M, q_k)(x))
```

注意：这一步是 TRELLIS 内部结构化输出，不是生成后的 StageC 投影。

渲染桥接：

- 在 geometry 阶段，不立即进入 hard `argwhere`。
- 使用 soft occupancy / coarse Gaussian / differentiable voxel splatting 生成固定视角视频。
- 首帧必须和输入图强绑定：推荐 first-frame latent clamp + photometric/DINO anchor。

输出：

```text
stage2_soft_ss/
  p_base_64.npy
  p_move_64.npy
  p_states_64.npy
  rendered_soft_video.mp4
  first_frame_anchor.png
```

---

## Stage 3 — Wan2.2 I2V-A14B RFSDS Geometry / Trajectory Optimization

优化变量：

```text
RASA adapter parameters
part logits / mask logits
joint variables: axis, pivot, q_k, range
small SS residual variables, if used
```

冻结：

```text
TRELLIS original weights
Wan2.2 original weights
input image encoder / text encoder
```

RFSDS loop：

```text
for iter in 1..N_geom:
    1. run K-state TRELLIS SS-DiT with RASA
    2. get p_B, p_M, T(q_k)
    3. render fixed-camera video Vθ
    4. clamp / anchor first frame
    5. encode video by Wan VAE: z = E_wan(Vθ)
    6. sample τ, ε; compute z_τ = (1-τ)z + τε
    7. frozen Wan2.2 predicts velocity v_hat(z_τ, τ, image_cond, text_cond)
    8. RFSDS surrogate gradient updates RASA / masks / joint
```

Schedule：

- high / mid noise: 主要优化 motion、trajectory、large geometry；
- lower noise: 边界细化；
- texture optimization 单独放 Stage 5。

输出：

```text
stage3_rfsds_geom/
  optimized_adapter.pt
  optimized_joint.json
  optimized_p_base.npy
  optimized_p_move.npy
  diagnostics/rfsds_curve.json
  diagnostics/fixed_camera_video_*.mp4
```

---

## Stage 4 — Topology Hardening and TRELLIS SLAT Generation

当 soft SS 稳定后，才 harden topology：

```text
p_B, p_M → threshold / connected components / confidence filtering
  → coords_B, coords_M
```

这一步是离散步骤，不声称其本身可导。之后进入 TRELLIS SLAT：

1. 固定 base/move coords。
2. 用 TRELLIS SLAT-DiT 生成 canonical base/move SLAT。
3. 插入 SLAT-RASA adapter，和 SS-RASA 共享 part-slot identity。
4. 解码为 3D Gaussian / radiance field / mesh。

推荐：

- RFSDS 优化 loop 里优先用 3D Gaussian / radiance field render，因为梯度更平滑。
- 最终 URDF 导出时再用 mesh / FlexiCubes branch 或后处理 mesh extraction。

输出：

```text
stage4_slat/
  coords_base.pt
  coords_move.pt
  slat_base.pt
  slat_move.pt
  gaussian_base.ply
  gaussian_move.ply
  mesh_base_initial.glb
  mesh_move_initial.glb
```

---

## Stage 5 — Cross-State Visible Texture Fusion + Wan Low-Noise Texture Refinement

目标：不要只用单闭态图纹理。要利用：

1. 输入图中可见的 closed-state exterior texture；
2. Wan2.2 fixed-camera opening video 中后续帧暴露的 interior / side / back surfaces；
3. TRELLIS SLAT appearance prior；
4. 多状态 inverse-warp 后的 canonical texture consistency。

Texture provenance 分三类：

```text
state0_visible: 输入图直接可见，强锚定
open_state_exposed: 打开后可见，Wan/RFSDS + SLAT prior 共同优化
never_visible: 完全不可见，只能由 TRELLIS/Wan appearance prior hallucinate，论文中不能声称真实
```

对 canonical surface / Gaussian / SLAT token i，建立 donor set：

```text
D(i) = {state k | surface i 在 state k 的固定相机视角可见}
```

融合：

```text
S_i_can = Σ_k w_{k,i} InvWarp(S_{k,i}, q_k)
C_i_can = Σ_k w_{k,i} C_{k,i}
```

权重来自：

```text
visibility confidence
view angle / projected area
part consistency
RFSDS low-noise residual confidence
state0 anchor priority
```

低噪声 Wan2.2 RFSDS 主要更新：

```text
SLAT texture residual
3DGS color / SH coefficients
fusion weights
small material embeddings
```

禁止在这一阶段大幅改变 geometry / joint。

输出：

```text
stage5_texture/
  textured_gaussian_base.ply
  textured_gaussian_move.ply
  texture_provenance.npy
  donor_weights.npy
  texture_refined_video.mp4
```

---

## Stage 6 — Mesh / URDF Export and Diagnostics

最终导出：

```text
final/
  base_mesh.glb
  move_mesh.glb
  object.urdf
  joint.json
  canonical_base.npy
  canonical_move.npy
  projected_states.npy
  texture_provenance.json
  diagnostics_report.md
```

`joint.json` 格式：

```json
{
  "joint_type": "prismatic | revolute",
  "axis": [x, y, z],
  "origin": [x, y, z],
  "range": [min, max],
  "state_values": [q0, q1, ...],
  "parent": "base",
  "child": "move"
}
```

必须检查：

- base 是否所有状态 bit-identical / slot-identical；
- move 是否只通过 single-DoF transform 变化；
- projected fixed-camera video 是否和 Wan prompt 一致；
- 首帧是否和输入图一致；
- no camera motion / no object morphing / no floating artifacts；
- URDF 能否在 PyBullet / Isaac / MuJoCo 中正常驱动。

---

## 4. 必做 sanity tests

### Test A — RFSDS gradient direction

不是只测正确视频 loss 是否低于错误视频。必须测梯度方向：

```text
wrong mask → RFSDS gradient 是否把 movable logits 推向正确区域
wrong axis → gradient 是否减小 axis angular error
wrong joint type → prismatic/revolute score 是否被修正
wrong exposed texture → low-noise gradient 是否提升 temporal texture consistency
```

### Test B — 固定相机 prompt 有效性

必须验证 Wan2.2 输出：

- 首帧与输入图基本一致；
- 无 camera pan / zoom / orbit；
- base 静止；
- 只有目标 move 部件动。

如果 prompt 不能锁定相机，则 RFSDS 会被 camera motion 吸走。

### Test C — argwhere/topology 断点

验证三种路径：

1. soft SS renderer only；
2. periodic topology refresh + fixed coords；
3. straight-through argwhere / Gumbel top-k，仅作为 ablation。

主线不应依赖不可控的 STE。

### Test D — texture provenance

分开报告：

- state0 visible texture；
- open-state exposed texture；
- never-visible hallucinated texture。

不能把 never-visible hallucination 当作真实重建。

---

## 5. 与现有方法的边界

### 5.1 和 FreeArt3D 的区别

FreeArt3D 依赖多状态图像；本项目是严格单闭态图。我们用 Wan2.2 I2V RFSDS 提供 opening video prior，用 TRELLIS 内部 RASA 生成多状态结构变量。

不能让方法退化为外部 hash-grid composition head + TRELLIS SDS，否则 novelty 会被审稿人归类为 FreeArt3D 变体。

### 5.2 和 PAct 的区别

PAct 是数据集训练 / fine-tune TRELLIS 来学习 articulated part attention。本项目冻结 TRELLIS，只做 per-instance test-time optimization，监督来自 Wan2.2 video prior。

### 5.3 和 CHORD 的区别

CHORD 用 Wan2.2 / RFSDS 优化通用 4D motion representation。本项目要输出 canonical base/move、single-DoF joint、URDF-ready assets，并把梯度接入 TRELLIS SS/SLAT 内部新增层。

### 5.4 和 DreamArt 的区别

DreamArt 更像后处理式 video prior + mesh/trajectory refinement。本项目的核心变量在 TRELLIS 内部生成流中产生，并且使用 SS-RASA 与 SLAT-RASA 共享 part identity。

---

## 6. 风险与对应处理

| 风险 | 具体表现 | 处理 |
|---|---|---|
| Wan 梯度被 camera motion 吸走 | 视频看似合理但相机在动 | locked-camera prompt + negative prompt + first-frame latent clamp |
| RFSDS 只优化纹理不改几何 | part mask / joint 不动 | geometry phase 禁止/冻结 texture；先用 silhouette/depth/normal/simple color |
| base 仍然变化 | 不同状态 base 不一致 | base branch shared by construction；base tokens no state-dependent transform |
| move 消失 | q_k → 0 或 move mask empty | motion lower-bound；q_k 非零初始化；trajectory monotonic prior |
| Wan 让物体变形 | non-rigid morphing | single-DoF transform bottleneck；adapter residual norm regularization |
| hard topology 不可导 | SLAT coords 频繁变化 | soft SS phase + fixed coords SLAT phase；periodic refresh 不进连续梯度 |
| texture hallucination 被过度宣传 | unseen surfaces 看似真实 | provenance map；论文中标注 plausible/inferred texture |

---

## 7. 论文主张

建议主张：

> We introduce a test-time optimizable articulated adapter inside frozen TRELLIS sparse-structure and SLAT generation. Conditioned on a single closed-state image, the adapter jointly produces canonical base, movable part, and a single-DoF trajectory. A locked-camera Wan2.2 I2V-A14B rectified-flow video prior is distilled through RFSDS, so that errors in part assignment, joint motion, and newly exposed texture produce gradients back to the inserted adapters and texture variables, while both TRELLIS and Wan2.2 backbones remain frozen.

中文：

> 我们不是从 TRELLIS 输出后处理出部件，而是在冻结 TRELLIS 的 SS/SLAT 生成内部插入可优化的跨状态部件层；Wan2.2 I2V-A14B 在固定单视角、首帧等于输入图的条件下，通过 RFSDS 同时优化部件分割、single-DoF 轨迹和跨状态可见纹理。

---

## 8. 参考来源

- TRELLIS project / paper: https://microsoft.github.io/TRELLIS/ ; https://github.com/microsoft/TRELLIS ; https://arxiv.org/abs/2412.01506
- Wan2.2 official repo / I2V-A14B model card: https://github.com/Wan-Video/Wan2.2 ; https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- CHORD / RFSDS: https://yanzhelyu.github.io/chord/ ; https://arxiv.org/abs/2601.04194
- FreeArt3D: https://arxiv.org/abs/2510.25765
- PAct: https://arxiv.org/abs/2602.14965
