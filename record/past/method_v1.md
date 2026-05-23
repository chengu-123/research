# CAST-U A'B v1 — Method

> 单图 + Wan2.2 单视角伪视频 → canonical articulated 3D 资产（base mesh + move mesh + single-DoF joint + UV/atlas + URDF），TRELLIS 主干完全冻结。
>
> **核心创新一句话**：把 TRELLIS 的 `SparseStructureFlowModel` 当成一个 grad-enabled 的 *one-step structural refiner*（不再当 sampler 用），让 W-RFSDS 视频蒸馏的梯度通过解析 SE(3) rollout 与可微 Gaussian 渲染，端到端地回到 SS-DiT 内部新增的 zero-init residual adapter，从而在不动 TRELLIS 主干的前提下学习 canonical articulated 几何、part segmentation 与 single-DoF joint 参数。

---

## 0. 目录

1. 目标与边界
2. 与既有方案的对比和裁决（为什么不用 v15 / CAST-U++ / v19.1 / GPT 综合 PDF）
3. 核心设计哲学
4. 变量定义
5. 输入条件构造
6. 一次性 Bootstrap
7. 几何阶段：A'B 主体
8. W-RFSDS 梯度链
9. 损失函数
10. 训练协议
11. 纹理阶段（简述）
12. 导出
13. 终极一句话

---

## 1. 目标与边界

### 1.1 任务定义

**输入**：
- 一张物体闭合状态的 RGB 图像 `I_0`
- 自然语言 prompt（描述铰接部件，例如 "open the drawer slowly, camera locked"）

**输出**：
- `base.glb`：静止部分的 textured mesh
- `move.glb`：可动部分的 textured mesh
- `joint.json`：single-DoF 关节（revolute 或 prismatic）
- `atlas.png` + `texture_provenance.json`：UV 纹理与每个 texel 的来源记录
- `object.urdf`：标准 URDF 资产

### 1.2 硬约束

- TRELLIS 主干（SS-VAE、SS-DiT、SLAT-DiT、D_GS Gaussian decoder、DINOv2）**完全冻结**
- Wan2.2 视频生成模型完全冻结
- 不允许使用 SAM / Grounded-SAM / 外部分割器做 part segmentation 伪标签初始化
- 不允许使用任何外部分割标注
- 每个对象 per-instance optimization（非通用 inference）
- 任务范围限定 single-DoF（revolute 或 prismatic）

### 1.3 类别范围

drawer、cabinet door、microwave / oven、refrigerator、laptop、washing machine 等 single-joint 物体。多关节、复合铰链、带弹性件的对象作为后续工作。

---

## 2. 与既有方案的对比和裁决

本方法在前期四份方案（v15、CAST-U++、CAST-U v19.1、GPT 综合 PDF）的基础上做关键修正。下表说明每个方案保留与放弃的内容、以及放弃的具体技术理由（不是模糊判断）。

### 2.1 v15（canonical-first）

**保留**：
- canonical-first 哲学（最终交付物是单个 canonical asset，K 个状态由解析派生）
- contact-anchor joint constraint 的数学化（softmin 距离形式）
- canonical SLAT/UV donor fusion 思路（多状态可见性加权融合）
- W-RFSDS 高低噪分阶段调度的想法

**放弃**：
- **D1：argwhere 断点处理停留在叙事**。v15 多次提及"绕开 raw argwhere"，但没有具体的可微替代物的形式化定义，也没说梯度通过 dense `SS-VAE-decoder(z_s)` 这个真正可微的桥。
- **D2：Cross-State Canonical Adapter 假设 K-state hidden 物理对齐**。v15 §SS-Adapter 描述每个 state 单独跑 SS-DiT 拿 `h_k`，再做跨 state 聚合。这隐含假设各 state 的 16³ hidden 网格在物理空间对齐。但 TRELLIS 的 bbox 归一化对不同 state 的输入图像给出不同的物理 scale（`s0` 闭合时 bbox 仅是柜体，`s_k` 半开时 bbox 包含外伸抽屉）。该对齐假设在源码层面不成立。
- **D3：UV unwrap 需要 mesh extraction**。最终纹理 atlas 必须先从 occupancy 提 mesh 再 unwrap，这一步是 marching cubes / FlexiCubes 的 argmax，断梯度。v15 写"几何与纹理重叠优化"，但 UV unwrap 一旦执行 geometry 就被锁死，与重叠叙事矛盾。

### 2.2 CAST-U++

**保留**：
- **support superset U 思路**：把可微优化锁在固定 topology 上，避开 `SparseTensor.coords` 不在 autograd 这一硬事实。
- presence gate `g_i` 与 move gate `m_i` 双独立 sigmoid 的设计（softmax 3-way 是冗余的，因为 uncertainty class 等价于 1-g）。
- analytic SE(3) rollout 把 K 状态从 canonical 派生。

**放弃**：
- **D4：outer-loop U refresh 引入周期性离散更新**，本质上是 raw argwhere 的低频版本。每次 refresh 触发 voxel warm-up、参数 remap、optimizer state 重排，工程上脆弱。CAST-U++ 自己也承认这是"必要但危险的外环"。本方法选择**固定 U**：用足够保守的初始 U_0 一次性覆盖所有真实几何候选，全程不刷新。
- **D5：U_0 的"bootstrap union"未明示跨 state 尺度漂移**。多状态独立 TRELLIS 输出再做 union，把每个 state 各自漂移的 voxel 一并塞进去，不是修复尺度。
- **D6：单独作为方法过偏 support/gate，纹理贡献不足**。CAST-U++ 不接 v15 的 canonical SLAT donor fusion，论文很难讲第二条贡献。

### 2.3 CAST-U v19.1

**保留**：
- **BinaryConcrete + STE gate** 的具体公式（前向硬 0/1、反向走 soft sigmoid 梯度），**但只在 D_GS 输出端的 Gaussian opacity 上施加**，不在 SLAT 输入 feats 上施加（理由见 D8）。
- **累积 softplus 单调 φ**：`phi_inc = softplus(delta_phi); phi = cumsum(phi_inc)` 强制 `0 < φ_1 < ... < φ_5`，免去额外单调 loss。
- **move-weighted joint pooling**：用 `m_i` 作为注意力权重池化 hidden，只让"可动 voxel"参与 joint 预测。
- **joint type 软混合**：训练时 `T_k = (1-σ(type))·T_rev + σ(type)·T_pri`，推理时硬切。
- **两阶段时序设计**（geometry 主导→ texture 主导）。

**放弃**：
- **D7：Canonical lift 用 ψ inverse-warp h_k**。v19.1 §5.2 把每个 state 的 `h_k ∈ R^{16×16×16×1024}` 通过 `T_k^{-1}` warp 到 canonical 网格做聚合。这继承了 D2 同样的物理对齐假设错误。
- **D8：STE 应用位置不明确**。v19.1 把 STE gate 作为 SLAT 输入 token 的 mask（`feats_input × g_hard`）；但 SLAT 的 input ResBlock 是个 `SparseResBlock3d` 卷积（`structured_latent_flow.py:129-147`），任何 binary 0/1 mask 经过它立刻被卷成连续值。所谓"STE 恢复 SLAT 训练时离散分布"的论证不成立。本方法**把 BinaryConcreteSTE 的施加点从 SLAT 输入挪到 D_GS 输出端 Gaussian opacity**。
- **D9：周期性 U refresh** 同 D4，放弃。

### 2.4 GPT 综合 PDF（v15 + CAST-U++ + v19.1）

**保留**：
- **三方案职责分离的概念框架**：v15 论文叙事 / CAST-U++ 几何变量定义 / v19.1 训练稳定协议。
- **A'B 主线**：no-grad sampler bootstrap + grad-enabled one-step canonical SS-DiT refiner + learnable z_s_base + SS heads + analytic rollout + W-RFSDS。这是 GPT PDF 真正的创新点。
- 活跃 loss ≤ 4 项的原则，避免一上来开 10 个正则的反模式。
- first_frame_rgb_anchor 损失防止纹理 phase 漂离真实输入。
- Provenance map（每个 texel 标注 source type）保科学诚实。
- W-RFSDS 梯度链显式写出（Wan VAE encode 保 grad、Wan DiT no_grad、renderer 可微）。

**放弃**：
- **D10：carpet 被贬为 "implementation detail"**。GPT PDF §2.1 写"carpet 不是方法主线"。但实测显示 carpet 是 TRELLIS 在单图输入下尺度稳定性的前置条件，不加 carpet `s0` 喂 TRELLIS 输出的体素尺度会显著小于 `s2`（5×4×4 vs 5×5×5 量级）。把 carpet 写在脚注里等于把方法学的前提隐藏。本方法把 carpet 作为 Stage A/B 的明确组件，并设计 U = U_object ∪ U_carpet 的两分集合处理（详见 §4.1.2 和 §6）。
- **D11：伪代码若干技术 bug**。GPT PDF 的伪代码有七处实际 bug（SLAT-DiT 不应进 inner loop / `α_g, α_m` 声明了没用 / `α_m` 不能零初始化 / `deltas` 不能零初始化 / `ψ_param` 被 H_joint 覆盖 / `φ` 索引 off-by-one / `D_GS` 在 K 循环内重复调用）。这些都在本方法的伪代码中修正。
- **D12：U refresh 仍保留**（GPT PDF §7.4），同 D4 放弃。

### 2.5 本方法的最小创新声明

1. **A'B 主线** — 不调 `FlowEulerSampler.sample`，只调一次 `SparseStructureFlowModel.forward` 拿 hidden 和 pred_v。把 SS-DiT 用作 part-aware feature extractor 而非 sampler。
2. **D_GS 输出端 opacity gating** — 把 BinaryConcrete + STE gate 应用在 D_GS 解码后的 Gaussian opacity 上，绕开 SLAT 输入 feats 经 ResBlock 卷积被模糊的问题。
3. **carpet 双集合 U = U_object ∪ U_carpet** — carpet 作为 fixed rendering scaffold 提供尺度锚定但不进入 object 优化主线。所有可学参数（α_g, α_m, ψ）只在 U_object 上定义；contact_anchor / joint loss / URDF export 全部排除 carpet。
4. **canonical 单一 forward + analytic K-state rollout** — SS-DiT 只跑一次 canonical forward（K=1），K 状态完全由解析 SE(3) 派生。避免 v19.1 / GPT PDF 的 K-state inverse-warp lift 物理对齐假设。
5. **三阶段 detach staging** — z_s_base 与 ψ 在同一调度下做 0–5% detach / 5–15% EMA / 15%+ full gradient 的反馈回路打断，统一管理两个潜在震荡源。

---

## 3. 核心设计哲学

### 3.1 Canonical-first

最终交付物是单个 canonical articulated 3D 资产 + 关节参数。K 个状态由解析 SE(3) rollout 派生：

$$O_k(x) = 1 - (1 - B(x))(1 - M(T_k^{-1}(x; \psi, \phi_k)))$$

其中 `B` 是 canonical base occupancy，`M` 是 canonical move occupancy，`T_k` 是 single-DoF SE(3) 变换。**Base 一致性由构造保证（by-construction），不依赖 loss "学" 出来。** 这是相对于"K 状态独立生成 + 后对齐"的范式跃迁。

### 3.2 冻结主干，只学新增层与残差

可学参数限于：
- SS-DiT 内部 block 14/16/18 之后的 zero-init residual adapter
- 三个 head MLP（H_sup / H_part / H_joint）
- 几何 / 部件 / 关节的显式 `nn.Parameter`：`Δz_s, α_g, α_m, ψ_param, delta_phi`

累计参数量约 5–20 MB / 对象。per-instance 优化负担很小，且保留预训练 TRELLIS 强 prior 不被破坏。

### 3.3 W-RFSDS 通过解析路径回到 SS

视频差异 → 渲染差异 → 通过 SE(3) 变换的链式 → 关节参数差异 → 通过 head MLP 链式 → SS-DiT block 14/16/18 之后的 adapter 权重。整条链可微。TRELLIS 主干因 `requires_grad_(False)` 不参与权重更新，但激活仍参与 autograd 图。

### 3.4 离散决策放在 D_GS 输出端

`BinaryConcreteSTE` 把 logit 转为硬 0/1 用作 Gaussian opacity 的乘子。这一位置：
- `diff_gaussian_rasterization` 对任意 `opacity ∈ [0, 1]` 完全可微
- 不经过 SLAT 的 `SparseResBlock3d` 卷积，binary 不被模糊
- forward 硬决策直接匹配导出时的二值化阈值

### 3.5 outer-loop 操作最小化

U 在 bootstrap 后**固定**，不做 refresh。所有几何 / 部件 / 关节优化都在可微 inner-loop 内完成。outer-loop 只剩下温度退火、λ warmup、detach phase 切换——这些都是连续调度，不产生离散跳变。

---

## 4. 变量定义

### 4.1 冻结资产（不参与优化）

| 符号 | 来源 | 形状 / 类型 |
|---|---|---|
| `SS-DiT` | TRELLIS pretrained | `SparseStructureFlowModel`，24 blocks × 1024 hidden × 16 heads，dense 16³ 输入 |
| `SS-VAE-decoder` | TRELLIS pretrained | dense 3D conv decoder 16³→64³，输出 `[1, 1, 64, 64, 64]` occupancy logit |
| `SLAT-DiT` | TRELLIS pretrained | sparse transformer，24 blocks |
| `D_GS` | TRELLIS pretrained | Swin-based decoder，32 Gaussians/voxel |
| `DINOv2` | DINOv2-L | ViT，1024-d patch tokens |
| `Wan2.2 I2V` | Wan-AI/Wan2.2-I2V-A14B | RF MoE，27B params 总 / 14B 激活 |

### 4.2 一次性 Bootstrap 资产（no_grad 计算，存盘）

| 符号 | 形状 | 含义 |
|---|---|---|
| `z_s0` | `[1, 8, 16, 16, 16]` | BMCSA 收敛后的 SS latent（K state 合并版） |
| `z_slat0` | `[\|U\|, 8]` | SLAT bootstrap latent（通过 SLAT sampler 跑一次得到，no_grad） |
| `dit_hidden_cache` | `[1, 4096, 1024] × 3` | 第 14/16/18 block 之后的 cached hidden（仅诊断用） |
| `O_init` | `[1, 1, 64, 64, 64]` | `sigmoid(SS-VAE-decoder(z_s0))` |
| `M_attn_boot` | `[\|U\|]` | BMCSA cross-state cosine 一致性，作 move prior |
| `is_carpet_mask` | `[\|U\|]` bool | FreeArt3D plane fitting 的结果 |
| `U_object` | `[N_obj, 3]` int32 | object 的候选 voxel 集 |
| `U_carpet` | `[N_carpet, 3]` int32 | carpet 的固定 scaffold voxel 集 |
| `ψ_0` | dict | StageC 给出的初始 joint（type, axis, origin, type_logit） |
| `φ_0` | `[6]` | 初始状态进度，`φ_0[0] = 0` |
| `anchors_object` | `[N_a, 3]` | object 上的接触带 anchor |

### 4.3 可学习参数

**P1 几何阶段**：

| 符号 | 形状 | 初始化 | 含义 |
|---|---|---|---|
| `Δz_s` | `[1, 8, 16, 16, 16]` | `zeros_like(z_s0)` | z_s_base 的残差 |
| `α_g` | `[N_obj]` | `logit(O_init[U_object].clamp(ε, 1-ε))` | object voxel 的存在 logit |
| `α_m` | `[N_obj]` | `logit(M_attn_boot[U_object].clamp(ε, 1-ε))` | object voxel 的可动 logit |
| `ψ_param` | `[19]` | `encode_joint(ψ_0)` | joint 显式参数（axis 3 + origin 3 + type 1 + reserve 12） |
| `delta_phi` | `[5]` | `inverse_softplus((φ_0[1:] - φ_0[:-1]).clamp_min(ε))` | 严格正的单调增量 |
| `adapter_{14, 16, 18}` | MLP | 输出 proj zero-init | SS-DiT residual adapter |
| `H_sup, H_part` | MLP | 输出 proj zero-init | support / move residual head |
| `H_joint` | MLP | 输出 proj zero-init | joint residual head |

**P2 纹理阶段**额外可学：

| 符号 | 形状 | 含义 |
|---|---|---|
| `Δz_slat` | `[\|U\|, 8]` | SLAT latent 残差 |
| `D_GS_LoRA` | rank=8 | D_GS 上的 LoRA |
| `Δ A_uv` | atlas 残差 | UV atlas 残差 |
| `donor_weights` | `[N_texel, K]` | donor 融合权重 |

### 4.4 关键初始化的依据

- `α_m` 不能 `zeros`（会让 `m_i ≡ 0.5`，每个 voxel 半 base 半 move，渲染期间产生大量鬼影 Gaussian，梯度 noisy）。改从 BMCSA 的 `M_attn_boot` 用 `logit` 初始化，让 RFSDS 只做小修正。
- `delta_phi` 不能 `zeros`（`softplus(0) ≈ 0.693`，初始 `φ_1` 直接 ~40°，与 BMCSA 的 `φ_0` 几乎不沾边）。改用 `inverse_softplus(φ_0 增量)`，初始 `φ` 严格等于 BMCSA 的 `φ_0`。
- `ψ_param` 是 explicit Parameter，head 输出 `Δψ` 只做 residual：`ψ_pred = project_joint(ψ_param + λ_joint · Δψ)`。早期 `λ_joint = 0` + `H_joint` zero-init 时，`ψ_pred ≡ project_joint(ψ_param)`，纯走 BMCSA 初值。

---

## 5. 输入条件构造

### 5.1 含 carpet / 不含 carpet 的两套数据

| 用途 | 内容 |
|---|---|
| **carpet-aided init** (Stage B 用) | 在 `s_0 ... s_5` 底部合成一个 FreeArt3D 风格的 grounding disk → `s_0_carpet ... s_5_carpet` |
| **clean supervision** (Stage D / F 用) | 原始 Wan2.2 输出 `s_0 ... s_5`（无 carpet） |

`s_0` 本身是用户提供的真实输入图（无 carpet）。`s_1 ... s_5` 是 Wan2.2 I2V 生成的伪开合视频帧。

### 5.2 BMCSA mixed input（仅 Stage B 用）

遵循旧 StageB 的 SCAR Pass-1 symmetric mix，在 SS-VAE latent space 做加权：

$$s_k^{\text{mix}} = 0.3 \cdot s_0^{\text{carpet}} + 0.4 \cdot s_k^{\text{carpet}} + 0.3 \cdot s_5^{\text{carpet}}$$

权重 `(0.3, 0.4, 0.3)` 是旧 StageB 实测稳定的配比。

### 5.3 Canonical image cond（默认配置）

```
cond_can = DINOv2(s_0_carpet)
```

**只用 `s_0` 的 DINO cond 喂 SS-DiT one-step forward。** K-state 信息通过 W-RFSDS 监督和 analytic rollout 注入，不让 SS-DiT hidden 携带 state-specific bias。这一点是相对于"per-state cond_k"的关键差别——后者会让 hidden 在不同 state 下携带不同的 DINO 语义，与"hidden 是 canonical-only"的假设冲突。

---

## 6. 一次性 Bootstrap（Stage B）

完整 `torch.no_grad()` 跑一次，输出 §4.2 列出的所有 bootstrap 资产。

### 6.1 步骤

1. 对 `s_0 ... s_5` 加 carpet → `s_0_carpet ... s_5_carpet`
2. 按 §5.2 在 SS-VAE latent space 构造 mixed input
3. 跑 BMCSA-enabled SS-DiT sampler（24 步去噪），所有 24 block 启用 BMCSA（K/V 跨 state 平均 + M_base soft gate）
4. 用 `forward_hook` 在 block 14/16/18 之后截取 hidden → `dit_hidden_cache`
5. `z_s0 = mean(z_final, dim=K)`
6. `O_init = sigmoid(SS-VAE-decoder(z_s0))`
7. `M_attn_boot = compute_cross_state_token_cosine(z_final)`
8. FreeArt3D plane fitting on `O_init` 底层 z-slice → `is_carpet_mask`
9. 跑 SLAT sampler (no_grad, 一次性) → `z_slat0`
10. StageC joint init：partition → BIC type voting → swept volume carve → axis refine → `ψ_0, φ_0, anchors_object`
11. 构造 `U_object`（保守 dilation）与 `U_carpet`
12. 全部 detach 后写盘

### 6.2 Bootstrap 的角色

旧 BMCSA / StageC 路线**不是最终方法**，只作为初始化器。它解决的是 Q1/Q2/Q3（尺度漂移 + base 一致性 + s0 偏小）。Q4/Q5（SS↔SLAT 梯度断、RFSDS 优化 SS 部件）由 Stage D 的 A'B 主体解决。

---

## 7. 几何阶段：A'B 主体（Stage D）

### 7.1 q_sample 公式（TRELLIS RF）

```python
σ_min = 1e-5    # from trainers/flow_matching/flow_matching.py:62
ε ~ N(0, I)     # noise, shared across K states within an iter
z_t = (1 - t) · z_s_base + (σ_min + (1 - σ_min) · t) · ε
```

**不要写成** `z_t = (1-t)z + tε`（σ_min=0 的简化版）。TRELLIS 训练时用的是带 σ_min 的公式（`flow_matching.py:87`）。

### 7.2 One-step SS-DiT structural refiner

```python
hidden_14, hidden_16, hidden_18, pred_v = SS_DiT_forward_with_adapters(
    x      = z_t,
    t      = 1000.0 * t_ss,    # ★ TRELLIS 内部约定：传入前先 × 1000
    cond   = cond_can,
    adapters = {14: adapter_14, 16: adapter_16, 18: adapter_18}
)
```

**Adapter 注入位置**（block 内）：

```python
# 在 SS-DiT 第 k 个 block 处，原版 forward 为：
h_post = block(h_pre, t_emb, cond)

# 改造为：
h_post = block(h_pre, t_emb, cond)
if k in {14, 16, 18}:
    h_post = h_post + adapter_k(h_post)     # post-adapter residual
    captured[k] = h_post                     # head 输入用 post-adapter
h_pre = h_post  # 下一个 block 输入
```

**关键**：head 必须读 post-adapter hidden，否则 adapter 对 head 无贡献。

**pred_v 仅作为 ablation 的 residual mix**，主版本不用：

```python
# 主版本：
occ_logits = SS_VAE_decoder(z_s_base)   # ★ 用 z_s_base，不用 pred_x0

# 仅 ablation：
pred_x0 = (1 - σ_min) · z_t - (σ_min + (1 - σ_min) · t) · pred_v
z_s_geo = (1 - η) · z_s_base + η · pred_x0   # η ∈ [0, 0.3]
```

### 7.3 几何底座 + residual heads

```python
z_s_base = z_s0 + Δz_s
occ_logits = SS_VAE_decoder(z_s_base)        # [1, 1, 64, 64, 64], 完全可微到 Δz_s

hidden = combine([hidden_14, hidden_16, hidden_18])   # e.g., mean / concat / FiLM

r_i = occ_logits[U_object_i] + α_g[i] + λ_sup  · H_sup(hidden, U_object_i)
b_i =                          α_m[i] + λ_part · H_part(hidden, U_object_i)
```

`λ_sup` 和 `λ_part` 在 `[0, 0.3]` 区间按训练进度 warmup（详见 §10）。早期 `λ = 0` 时，gate 完全依赖 `occ_logits + α` 这一稳定路径，等 hidden 在 adapter 修正后稳定，再让 head 介入。

### 7.4 BinaryConcrete + STE gate

```python
def BinaryConcreteSTE(logit, T):
    u = torch.rand_like(logit)
    g_soft = sigmoid((logit + log(u) - log(1 - u)) / T)
    g_hard = (g_soft > 0.5).float()
    return g_hard - g_soft.detach() + g_soft    # forward hard, backward soft

g_i_object = BinaryConcreteSTE(r_i, T_g)
m_i_object = BinaryConcreteSTE(b_i, T_m)
```

`T_g, T_m` 退火：1.5 → 0.2（geometry phase 内 cosine annealing），P2 固定 0.15。

**核心理由**：gate 不施加在 SLAT 输入 feats（会被 input ResBlock conv 模糊），而是在 §7.9 的 D_GS 输出端 Gaussian opacity 上施加。这一位置 binary forward 直接生效，反向走 sigmoid 软梯度。

### 7.5 拼接 object + carpet

```python
g_full = scatter(g_i_object, U_object) ∪ const_1(U_carpet)
m_full = scatter(m_i_object, U_object) ∪ const_0(U_carpet)
```

carpet voxel：永远 present（g=1）、永远 base（m=0）、无可学参数、不接收梯度。

### 7.6 Joint residual

```python
F_pool = move_weighted_pool(hidden, m_i_object)    # 加权池化 over U_object
Δψ = H_joint(F_pool)                                # zero-init output

ψ_for_warp = stage_detach(ψ_param, phase, mode="joint")
ψ_pred = project_joint(ψ_for_warp + λ_joint · Δψ)
```

`project_joint` 强制结构约束：
- axis 单位归一化
- prismatic direction 单位归一化
- pivot 落在物体 bbox 内
- joint type 通过 softmax / Gumbel-ST 选择
- revolute / prismatic 软混合（训练）/ 硬切（推理）

### 7.7 累积 softplus φ

```python
phi_inc = F.softplus(delta_phi)                                # [5], 严格正
phi = torch.cat([torch.zeros(1), torch.cumsum(phi_inc, 0)])    # [6]
# phi[0] = 0
# phi[k] = φ_1 + ... + φ_k 严格单调增
```

注意长度是 6（含 `phi[0] = 0`），不是 5。

### 7.8 Analytic SE(3) rollout

对每个状态 `k = 0, 1, ..., 5`：

```python
if joint_type == revolute:
    T_k(x) = R(ω, phi[k]) · (x - q) + q       # ω = axis, q = pivot
elif joint_type == prismatic:
    T_k(x) = x + phi[k] · v̂                   # v̂ = unit direction
# 训练时按 type_soft = σ(ψ.type_logit) 软混合两条分支
```

`phi[0] = 0` ⇒ `T_0 = Identity`。

### 7.9 Canonical Gaussian 解码 + warp + opacity gating

**关键工程点**：D_GS 在 K 循环外调用**一次**，K 个状态共享 canonical Gaussians，只 warp 中心和方向。

```python
gauss_can = D_GS(SparseTensor(z_slat0, U))     # ★ once per iter, frozen decoder

for k in 0..5:
    T_k = SE3_rollout(ψ_pred, phi[k])
    
    # 每个 canonical voxel 派生两个 Gaussian 贡献：
    # base 贡献 — 不变 center
    base_gauss_k = gauss_can.clone()
    base_gauss_k.opacity *= g_full * (1 - m_full)
    
    # move 贡献 — warp 后 center / rotation
    move_gauss_k = warp_gaussians(gauss_can, T_k)
    move_gauss_k.opacity *= g_full * m_full
    
    gauss_k = combine(base_gauss_k, move_gauss_k)
    rgb_k = diff_gaussian_rasterize(gauss_k, camera_fixed)
```

`warp_gaussians` 对 move voxel 的 Gaussian：
- `center` ← `T_k @ canonical_center`
- `rotation` ← `R_k @ canonical_rotation`
- `scale, sh` 不变
- base voxel 的 Gaussian 全部保持 canonical 不动

**为什么 opacity 在 D_GS 输出端**（D_GS 是冻结的）：
- 不需要改 D_GS 内部代码
- binary forward 直接生效，不被任何 conv 模糊
- diff_gaussian_rasterize 对 opacity ∈ [0, 1] 完全可微，梯度自然回传到 g, m

---

## 8. W-RFSDS 梯度链

### 8.1 W-RFSDS 形式

```python
def L_WRFSDS(rgb_frames, wan_video, τ_range, text_cond):
    τ = sample(τ_range)
    
    z_θ = wan_vae_encode(rgb_frames)              # ★ grad-enabled
    
    with torch.no_grad():                          # ★ Wan DiT is teacher, no grad
        ε = torch.randn_like(z_θ)
        z_τ = (1 - τ) * z_θ.detach() + τ * ε
        v_pred = wan_dit(z_τ, τ, text_cond)
    
    residual = v_pred - (ε - z_θ.detach())
    w_τ = w_rfsds_weight(τ)
    
    return w_τ * (residual.detach() * z_θ).sum()   # inner product form
```

**三个硬实现点**：
1. `wan_vae_encode` 必须 grad-enabled（否则梯度不流回 `rgb_frames`）
2. `wan_dit` 必须在 `torch.no_grad()` 内（它是 frozen teacher）
3. `residual` 在 `no_grad` 上下文中算好后 detach，SDS 内积形式让梯度从 `z_θ` 流回 `rgb_frames`

### 8.2 完整梯度路径

```
W-RFSDS loss
  ↓
diff_gaussian_rasterize.backward
  ↓
gauss_k.{opacity, center, rotation}
  ├── opacity 链：g_full * (1 - m_full), g_full * m_full
  │     ↓
  │   g_i_object, m_i_object  ✓
  │     ↓ BinaryConcreteSTE backward (soft gradient)
  │   r_i, b_i
  │     ↓
  │   α_g, α_m  ✓
  │   H_sup(hidden), H_part(hidden)  ✓ → hidden → adapter_{14,16,18}  ✓
  │   occ_logits ← SS_VAE_decoder(z_s_base)
  │     ↓ frozen weights backward (激活仍参与图)
  │   z_s_base = z_s0 + Δz_s
  │     ↓
  │   Δz_s  ✓
  │
  ├── center / rotation 链：T_k(ψ_pred, phi[k])
  │     ↓
  │   ψ_pred = project_joint(ψ_param + λ_joint · Δψ)
  │     ↓
  │   ψ_param  ✓
  │   H_joint(hidden, pooled)  ✓ → hidden → adapter  ✓
  │   
  │   phi = cumsum(softplus(delta_phi))
  │     ↓
  │   delta_phi  ✓
```

### 8.3 z_s_base 的两条梯度来源

`Δz_s` 同时从两条路径接收梯度：
1. **Decoder 路径**：`z_s_base → SS_VAE_decoder → occ_logits → r → g → render → loss`。永远开启。
2. **SS-DiT 路径**：`z_s_base → q_sample → z_t → SS-DiT → hidden → heads → m, ψ → render → loss`。仅当 `z_for_q ≠ z_s_base.detach()` 时开启。

第二条路径在前 5% 完全 detach，5-15% 通过 mixing 渐进开启，15% 后全开。理由：早期 `Δz_s`, adapter, heads 同时被 RFSDS 拉极易震荡；让 decoder 路径先稳住 `Δz_s`，再放开 SS-DiT 路径。

---

## 9. 损失函数

### 9.1 P1 几何阶段总损失

```
L_geom =
    λ_rfsds   · L_W-RFSDS(rgb_frames, wan_video_clean, τ_high_mid)
  + λ_first   · ( L1(rgb_0, s_0_clean) + LPIPS(rgb_0, s_0_clean) )
  + λ_contact · L_contact_anchor(ψ_pred, anchors_object)
  + λ_gate    · mean( σ(r)·(1-σ(r)) + σ(b)·(1-σ(b)) )
  + λ_z       · (Δz_s ** 2).mean()
```

**任何时刻活跃 loss ≤ 4 项**。`L_z` 是常规小权重 anchor 损失，不算"活跃"。

可选 stage-gated 损失：
- `L_collision`（仅 transition 阶段开）
- `L_swept_volume`（仅 transition 阶段开）

### 9.2 各项细节

**`L_W-RFSDS`**：见 §8.1。高中噪 `τ ∈ [0.6, 0.9]`（P1），低噪 `τ ∈ [0.1, 0.4]`（P2）。

**`L_first`**：第 0 帧渲染对真实 `s_0_clean`（无 carpet 版）做像素 + LPIPS 一致性。锚定整体外观。

**`L_contact_anchor`**：
- revolute: `L_axis = softmin_a dist(line(ω, q), a)^2`
- prismatic: `L_dir = 1 - cos(v̂, corridor_direction)`

只在 `U_object` 上算（carpet 不参与）。

**`L_gate`**：用 **soft 值**算 entropy 正则。注意不能用 hard 值（hard 值的 g(1-g) ≡ 0，loss 永远 0）。

**`L_z`**：mean-squared，不是 sum，防止 scale 随 |z_s0| 变化。

### 9.3 P2 纹理阶段损失（简述）

```
L_tex =
    λ_rfsds_low · L_W-RFSDS(rgb_frames, wan_video_clean, τ_low)
  + λ_donor    · L_donor_consistency(rgb_frames, A_fused)
  + λ_seam     · L_uv_seam(A_fused)
  + λ_first    · L_first_frame_rgb_anchor(rgb_0, s_0_clean)
  + λ_unseen   · L_unseen_smooth_prior
```

`L_first_frame_rgb_anchor` 防止 low-τ teacher 把材质 / 色彩拉离真实输入。

---

## 10. 训练协议

### 10.1 阶段表

| 阶段 | iter 比例 | `t_ss` | detach 策略 | `λ_sup, λ_part` | `λ_joint` | `T_g, T_m` |
|---|---|---|---|---|---|---|
| Warmup G0 | 0–10% | 固定 0.30 | full detach | 0 | 0 | 1.5 |
| Main G1 | 10–60% | `U(0.25, 0.55)` | 0–5% detach, 5–15% EMA, 15%+ full | 0 → 0.3 | 0 → 0.5 | 1.5 → 0.4 |
| Transition | 60–75% | `U(0.20, 0.40)` | full grad | 0.3 | 0.5 | 0.4 → 0.2 |
| P2 Texture | 75–100% | `U(0.10, 0.40)` | freeze geometry | held | held | 0.15 |

### 10.2 三阶段 detach staging

```python
def stage_detach(tensor, phase, mode):
    f = phase.iter / phase.total
    if f < 0.05:
        return tensor.detach()
    elif f < 0.15:
        if mode == "joint":     # EMA for ψ
            ema_buffer = β · ema_buffer + (1-β) · tensor.detach()
            return ema_buffer
        else:                   # mixing for z_s_base
            ρ = (f - 0.05) / 0.10        # 0 → 1
            return tensor.detach() + ρ * (tensor - tensor.detach())
    else:
        return tensor                    # full grad
```

理由：A'B 有两条反馈回路（`ψ↔m` 和 `Δz_s↔hidden→head`）同时活跃。早期强行 full grad 极易震荡。staged 策略让两条回路分阶段打开。

### 10.3 ε 噪声 K-state 共享

每个 iter 内采样一次 `ε ~ N(0, I)`，6 个 state 的 q_sample 共享同一噪声。理由：如果每个 state 用不同 `ε`，hidden 在 K state 间会带 noise-induced 差异，可能被错误归因到"part 差异"上。

### 10.4 优化器

| 参数组 | optimizer | lr |
|---|---|---|
| `Δz_s` | AdamW | 1e-4 |
| `α_g, α_m` | Adam | 5e-3 |
| `ψ_param` | Adam | 5e-3 |
| `delta_phi` | Adam | 1e-2 |
| `adapter` | AdamW | 1e-4 |
| `H_sup, H_part, H_joint` | AdamW | 5e-4 |

P2 阶段：
- `Δz_slat`: 1e-3
- `D_GS_LoRA`: 1e-4
- `Δ A_uv`: 5e-3

---

## 11. 纹理阶段（简述）

冻结几何：`Δz_s, α_g, α_m, ψ_param, delta_phi, adapter, H_sup, H_part, H_joint`。

可学：`Δz_slat, D_GS_LoRA, Δ A_uv, donor_weights`。

### 11.1 Donor 收集

对每个 canonical surface 点 / texel：

```python
for k in 0..5:
    p_k = T_k(p; ψ_pred, phi[k]) if p is move else p
    u_k = project(p_k, camera)
    if visible(p_k):
        α_k = visibility_k · exp(-β1·angle_k² - β2·depth_k - β3·blur_k)
        donor_k = sample(s_k_clean, u_k)

A_fused(texel) = Σ_k α_k · donor_k / (Σ_k α_k + ε)

provenance(texel):
    if Σ_k α_k > τ_high:           "multi_state_fused"
    elif visibility_0 > τ:         "visible_in_s0"
    elif visibility_open > τ:      "visible_in_open"
    elif geom_completed:           "low_conf_completion"
    else:                          "unobserved"
```

### 11.2 W-RFSDS 切低 τ

`τ ∈ [0.1, 0.4]`，让 Wan teacher 主要修细节而不是整体结构。

---

## 12. 导出

### 12.1 硬阈值

```python
g_object_hard = σ(α_g + occ_logits[U_object] + λ_sup · H_sup) > 0.5
m_object_hard = σ(α_m + λ_part · H_part) > 0.5

base_voxels = U_object[g_object_hard & ~m_object_hard]
move_voxels = U_object[g_object_hard &  m_object_hard]
# carpet 不参与导出
```

### 12.2 Mesh + atlas

```python
base_mesh = mesh_extract(base_voxels, z_slat_final)
move_mesh = mesh_extract(move_voxels, z_slat_final)
base_atlas, move_atlas = uv_unwrap_and_bake(base_mesh, move_mesh, A_fused)
```

### 12.3 Joint + URDF

```python
ψ_hard = harden_joint(ψ_pred)   # type argmax, axis L2-normalize

joint = {
    "type"        : ψ_hard.type,                  # "revolute" or "prismatic"
    "origin"      : ψ_hard.origin.tolist(),
    "axis"        : ψ_hard.axis.tolist(),
    "limit_lower" : 0.0,
    "limit_upper" : phi[5].item(),
    "states"      : phi.tolist()
}

object.urdf = compose_urdf(base.glb, move.glb, joint)
```

### 12.4 Provenance 报告

```python
texture_provenance = {
    "visible_in_s0_ratio"      : ...,
    "multi_state_fused_ratio"  : ...,
    "low_conf_completion_ratio": ...,
    "unobserved_ratio"         : ...,
}
```

---

## 13. 终极一句话

**用 grad-enabled 的 one-step SS-DiT structural refiner 把 W-RFSDS 视频蒸馏的梯度从渲染像素端，经可微 Gaussian 渲染、解析 SE(3) rollout、D_GS 输出端的 BinaryConcrete opacity gate，端到端地回到 SS-DiT block 14/16/18 之后新增的 zero-init residual adapter 和三个 head MLP，从而在冻结 TRELLIS 主干的前提下，在固定的 canonical support `U_object` 上学习 base/move 分割、single-DoF joint 参数和细粒度纹理；carpet 作为 fixed scaffold 提供尺度锚但不进 object 优化主线。**

---

## 附录：与 method.md 配套的 pipeline.md

工程实现规范（stage 顺序、I/O、冻结表、TRELLIS 特定约定、文件结构、sanity check）见 `pipeline.md`。

