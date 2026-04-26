# Stage B — 完整技术档案

> 最后更新：2026-04-18
> 状态：v4.1 已实现，方案 3（attention-driven M_attn）设计中
> 配套文档：
>   - `record/design.md` §3（旧版历史 spec）
>   - `record/redesign.md`（v3 SCAR 详细推演）
>   - `record/2026-04-18-stageb-v3-sdedit-design.md`（v3 SDEdit 完整 spec）
>   - `record/2026-04-18-stageb-v4-bmcsa-design.md`（v4/v4.1 BMCSA 完整 spec，**当前 authoritative**）

---

## 1. Stage B 的定位

**输入**：
- State 0：单张真实带分割图像 `I_0`（主输入，闭合态的 ground truth）
- States 1..K-1：Stage A 的视频扩散从 state 0 生成的合成图像（打开过程）
- DINOv2 patch features `c_0..c_{K-1}`（frozen, ViT-L/14, 每个 state 1374 tokens × 1024 dim）

**输出**：
- K 个 64³ 占据图 `{O_k}`，跨状态 **base 一致**（柜体结构同一），**move 差异**（抽屉/门在各自位置）
- K 个 16³×8 SS latent `{z_k}`，下游 Stage D/E 复用

**核心问题**：TRELLIS SS flow model 以 K-parallel 方式运行时，每个 state 的 DiT forward 是独立的；即使共享初始噪声 + 每个 state 的 c_k 图像条件，最终输出的**柜体几何存在跨状态 drift**，且 state 0 还有额外的 **TRELLIS 闭合态系统性缩小 bias**。Stage B 的任务就是消除这两个问题。

---

## 2. 演进时间线（对照表）

| 版本 | 机制 | 状态 | 主要缺陷 |
|---|---|---|---|
| **v0** (pre-) | Plain K-parallel（纯独立采样）| deprecated | 无跨状态 consistency，base 大幅 drift |
| **v1 VGCF** | Variance-gated consensus force（Tweedie 空间梯度推）| deprecated | 力场方向不可控，时好时坏 |
| **v2 BCAC** | Block-level consensus attention control | deprecated | 仅 DiT 中层 block，作用不充分 |
| **v3a SCAR + push** | Mix（latent 混合）+ Tweedie push（α/t 形式）| legacy ablation | push 不稳（α/t 在 t→0 发散；方向伪 score-form）|
| **v3b SCAR + α·t push** | 同上，α·t 替换 α/t | legacy ablation | push 仍然时好时坏；未根本解决 |
| **v3c SDEdit + augmented intersection** | Pass 1 SCAR → Pass 2 SDEdit with per-state guide | legacy（保留）| 合成 guide 走 encoder 有 OOD 风险；Pass 2 12 步内 K 个 state 仍独立演化 |
| **v4 BMCSA** | 保留 SDEdit 起点 + Pass 2 去噪的 DiT self-attn 做 cross-state blend | **已实现** | base 判定在 long-drawer 平移下有 ~2 voxel 污染（本文档 §6 问题 4）|
| **v4.1 shared P_base** | v4 的 P_base 从 per-target MIN 修复为所有 target 共享 mean(s0..K-1) | legacy | 同 v4 |
| **v4.2 attention M_attn in BMCSA** | 加入 attention-driven M_attn（方案 3）；`M_eff = M_base × M_attn` 只在 BMCSA blend 生效 | 已实现 | guide 里的 P_base 没过滤，起点依然 over-large |
| **v4.3 attention M_attn upstream** | M_attn 上采样到 64³，**直接过滤 P_base_shared**，guide 源头就正确 | **新默认** | — |

---

## 3. 当前架构（v4.1）—— Pass 1 + Pass 2 + BMCSA

### 3.1 Pass 1：SCAR + symmetric mix，push 关闭

```
输入: K=6 共享噪声 eps（shape (1, 8, 16, 16, 16)，repeat K 份），per-state c_k
采样器: SCARSampler（inherits FlowEulerGuidanceIntervalSampler）
步数: 25（TRELLIS-image-large 默认）
配置:
  extreme_mix_mode: symmetric
  mix_weights: (0.3, 0.4, 0.3)            # (w_first=s_0锚, w_self=自留, w_last=s_{K-1}锚)
  mix_steps: 8                             # 前 8 步做 latent 混合
  alpha_peak: 0.0                          # push 关闭
```

**Mix formula**（steps 0..7）：对每个 state k，`z_t^(k) ← 0.3·z_t^(0) + 0.4·z_t^(k) + 0.3·z_t^(K-1)`

- Mix 在 8 步后自动停用，后 17 步纯 K-parallel plain sampling
- 产物：`z_final: (K, 8, 16, 16, 16)` 和 decode 后的 `soft_p1, binary_p1: (K, 64, 64, 64)`

**为什么选 symmetric 而不是 mean_of_middles**：
- mean_of_middles 每步把 K 个 latent 归到同一个值，过度同质化 → "states 2,3 相同"问题
- symmetric 每个 state 保留 40% 自己，更好的 per-state 差异保留

### 3.2 P_base_shared 构造（v4.1 修复后）

```python
# 所有 K 个 state mean（包括 state 0，让 TRELLIS 缩小 bias 被稀释到 1/K）
P_base_shared = mean_{k=0..K-1} P^(k)                # (64, 64, 64)
```

**关键修复**（v4.1 vs v4-draft）：早期我让每个 target 独立算 `min_{k != target} P^(k)`，结果每个 state 的 P_base 不同，state 5 的 P_base 被 state 0 的缩小 bias 污染。v4.1 改为**共享 mean over all K**。

**为什么 mean 不用 min**：
- min 对单一离群 state（= state 0 shrunk）过度敏感，会拖垮 base 识别
- mean 下 state 0 贡献 1/K = 17%，即使 state 0=0.2 state 1-5=0.9，mean=0.78 仍然 > 0.5 阈值，voxel 还在 base

### 3.3 Per-state z_guide_k 构造

对每个 state k：

```python
# P_excl: state k 独占 voxel（其他 state 都没有的那部分）
# max 用的是所有 non-target state，不受 state 0 污染（空贡献不 dominate max）
P_excl_k = ReLU(P^(k) - max_{j != k} P^(j))          # (64, 64, 64)

# P_guide: base 部分共享 + state 独占部分自加
P_guide_k = max(P_base_shared, P_excl_k)             # (64, 64, 64)

# 二值化 + encode 回 SS latent
O_guide_k = (P_guide_k > 0.5).float()
z_guide_k = SparseStructureEncoder(O_guide_k.unsqueeze(0).unsqueeze(0))  # (1, 8, 16, 16, 16)
```

- 所有 K 个 z_guide_k 的**base 部分相同**（都来自共享 P_base_shared）
- 差异只在 P_excl_k 补充的 voxel 上

### 3.4 Pass 2 SDEdit 起点

```python
# Pass 2 起点: per-state noisy guide
# shared_noise = Pass 1 的那份 eps（repeat K 份）
t_star = 0.5
x_{t*}^(k) = (1-t*) · z_guide_k + (σ_min + (1-σ_min) · t*) · shared_noise
```

- 所有 6 个 state 共享同一份 `eps`（保持 K-parallel 随机家族）
- Guide 在 t=0 clean，加噪到 t=0.5 水平
- Pass 2 从这个起点去噪 12 步（**固定 12 步**，不按 `ceil(25 × t*)` 变化）

### 3.5 Pass 2 去噪：BMCSA (Base-Masked Cross-State Attention)

每步 DiT forward 里，**每个 self-attn 层**都做 BMCSA 插入：

```
原 self-attn:    h = self_attn(x)

BMCSA 修改后:
    y_self = self_attn(x)                              # 每个 state 独立 self-attn
    y_shared = self_attn(x, share_kv_across_batch=True)  # Q 每个 state 自己，K/V 跨 batch mean
    M_base = (1, L=4096, 1) from Pass 1 P_base_shared 下采样到 token 分辨率
    eff_M = clamp(bmcsa_strength * M_base, 0, 1)
    h = (1 - eff_M) * y_self + eff_M * y_shared
```

- **全 24 个 block** 都开 BMCSA（default）
- Cross-attn 不变（每个 state 依然 attend 各自的 c_k 图像特征）
- M_base 空间对应：P_base_shared @ 64³ → avg_pool3d(kernel=4, stride=4) → 16³ → flatten to (1, 4096, 1)
  - 因 TRELLIS-image-large `resolution=16, patch_size=1`，token count = 16³ = 4096（不是 8³）

### 3.6 Pass 2 输出替换

```
# Pass 2 完成后:
z_final_v4 = Pass 2 的 z_final（12 步去噪后）
# BMCSA mode 替换 ALL K 个 state
```

解码 → `O_stack` → `remove_disk` → 保存。

### 3.7 实现位置

| 模块 | 改动 |
|---|---|
| `mine/TRELLIS/trellis/modules/attention/modules.py::MultiHeadAttention.forward` | 新加 `share_kv_across_batch` kwarg |
| `mine/TRELLIS/trellis/modules/transformer/modulated.py::ModulatedTransformerCrossBlock._forward` | 加 BMCSA 分支 |
| `mine/TRELLIS/trellis/models/sparse_structure_flow.py::SparseStructureFlowModel.forward` | 加 `**kwargs` + `block_idx` 透传 |
| `mine/pipelines/stage_b_scar.py` | 主 driver、guide builder、M_base 构造、Pass 2 sampler、viz |
| `mine/configs/v1.yaml::stage_b_sdedit` | BMCSA 配置 key |

---

## 4. 各阶段产物（可视化诊断）

```
<output_dir>/stage_b/
├─ z_final.pt                           # 最终 SS latent
├─ O_stack.npy / _soft.npy              # 最终 K 个 occupancy（Pass 2 后）
├─ O_stack_pass1.npy / _pass1_soft.npy  # Pass 1 对照（未经 Pass 2）
├─ sdedit_report.json                   # mode / t_star / pass2_steps / voxel_deltas
│
├─ viz/
│   ├─ O_stack.html / O_k_{00..05}.html # 最终 K 个 state 的 html viz
│   ├─ guide/
│   │   ├─ states_1_to_Km1_overlap_*.html  # s1-K-1 soft intersection（参考）
│   │   ├─ state{k}_P_base.html              # v4.1 后所有 k 相同（= P_base_shared）
│   │   ├─ state{k}_P_excl.html              # 每个 k 不同
│   │   ├─ state{k}_P_guide.html             # = max(P_base, P_excl_k)
│   │   └─ state{k}_O_guide_bin.html
│   ├─ bmcsa/
│   │   ├─ P_base_shared_64.html             # mean over K 的原始概率场
│   │   ├─ M_base_64.html                    # sigmoid 后的 soft gate
│   │   └─ M_base_64.npy
│   └─ scar_diagnostics.html                 # Pass 1 每步 mask / push_norm 曲线
│
├─ per_step/                                 # Pass 1 25 步演化
│   └─ O_step_00.html .. O_step_24.html
│
└─ pass2_per_step/                           # Pass 2 12 步演化
    ├─ O_step_00.html                        # t*=0.5 noisy starting point
    └─ O_step_11.html                        # t=0 最终（应和 O_k_*.html 一致）
```

---

## 5. 累计碰到的问题和解决（完整历史）

### 问题 1: Tweedie push (α/t) 在 t→0 发散

- **症状**：push 强度在采样末期爆炸，破坏 state 0 形状
- **根因**：`push ∝ α/t`，t 向 0 时 1/t → ∞；虽然 α 也衰减，但 α/t 整体不受控
- **修复**：换成 `α·t`（σ_t normalization from Kynkäänniemi 2024 arxiv:2404.07724）
- **结果**：发散消除，但 push 本身"时好时坏"的问题仍在（见问题 2）
- **档案**：scar.py `apply_scar_gradient_push`

### 问题 2: Push 时好时坏（fundamental），mix 被 push 拖累

- **症状**：有的数据 push 后 state 0 对齐好，有的反而漂得更厉害
- **根因**：push 朝 cross-state consensus 拉，consensus 本身可能被 state 0 outlier 污染（自指）；而且 push 的力场 direction 和 DiT 原生 denoising direction 不一致
- **修复**：**push 完全禁用**（alpha_peak=0.0），改走 mix + Pass 2 架构
- **档案**：configs/v1.yaml

### 问题 3: mean_of_middles mix 过度同质化

- **症状**：state 1-4 之间相似度过高，几乎融合成一个 state
- **根因**：mean_of_middles 每步把 K 个 latent 归到 0.3·s0 + 0.4·mean(s1..K-2) + 0.3·s_{K-1}，所有 K 个 state 拿到同一个 mixed；per-state 差异靠后续 DiT 重建
- **修复**：换回 `symmetric` 模式：`mixed^(k) = 0.3·s_0 + 0.4·s_k + 0.3·s_{K-1}`，每 state 保留 40% 自己
- **档案**：v1.yaml scar.extreme_mix_mode

### 问题 4: Pass 2 SDEdit 起点构造里 z_guide per-target 不一致（v4-draft bug）

- **症状**：viz 里每个 state 的 `state{k}_P_base.html` 长得都不一样
- **根因**：guide builder 用 `min_{j != target} P^(j)` 算 per-target base，每个 target 的 reference pool 不同 → 不同 P_base。**state 5 的 P_base 池含 state 0，被 TRELLIS 缩小污染**
- **修复**：v4.1 改成**所有 target 共享 `mean_{k=0..K-1} P^(k)`**，guide builder 接受 `P_base_shared` 参数
- **档案**：_build_augmented_intersection_guide, run_scar bmcsa branch

### 问题 5: Pass 2 t_star=0.5 时 SDEdit 步数被 `ceil(25 * 0.5)` 绑定为 13

- **症状**：调 t_star 时步数也变化，难以控制 variable
- **修复**：加 `pass2_steps: 12` config，固定步数，**t_star 和步数解耦**
- **档案**：_sdedit_refine_k6_bmcsa

### 问题 6: DiT `patch_size=2` 的错误假设 → token 数 8³=512 错算

- **症状**：BMCSA hook 里 `M_base` shape 是 (1, 512, 1)，但 attention 实际输入是 (K, 4096, D)，broadcast 失败
- **根因**：TRELLIS-image-large 实际 `patch_size=1`，token count = 16³ = 4096
- **修复**：从 flow_model 属性自动推 `token_resolution = resolution / patch_size = 16`，config default 改 16
- **档案**：_compute_M_base_tokenspace, run_scar bmcsa branch

### 问题 7: SparseStructureEncoder 不在默认 pipe.models 里

- **症状**：`pipe.models['sparse_structure_encoder']` KeyError
- **根因**：TRELLIS 官方 pipeline.json 只打包 inference 必需的 decoder + flow，encoder 没进去
- **修复**：`pipelines/recon.py` 里新加 `_load_sparse_structure_encoder()`，从 `JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.{json,safetensors}` 手动拉 encoder
- **档案**：recon.py `build_trellis_pipeline`

### 问题 8: `_sdedit_refine_k6_bmcsa` 里 `edict` 未 import

- **症状**：`NameError: name 'edict' is not defined`
- **根因**：我写新函数时忘 copy sibling 函数里的 `from easydict import EasyDict as edict` local import
- **修复**：加回 local import
- **档案**：_sdedit_refine_k6_bmcsa

### 问题 9: Viz 里 P_base 和 M_base 用的 exclude_state_0 不一致

- **症状**：Pass 2 用了 `exclude_state_0=False`（all K mean）但 viz 里算 P_base 用的是 `exclude_state_0=True`（s1-5 mean）
- **修复**：viz 里改用同一个 `_compute_P_base_shared` 辅助函数，保证一致
- **档案**：run_scar viz section

---

## 6. 当前已知的限制 / 未解决问题

### 问题 10（severe / theoretical）: 长抽屉 + 小平移距离下 trajectory 中段被误判为 base

**具体场景**：prismatic 关节（抽屉平移），drawer body 长度 L 大于 state 平移 shift × (K-1) 的某个阈值，导致"多个 state 的 drawer 在同一 voxel 重叠"。

**例**：L=5 voxel，shift=1/state，K=6。
```
voxel 4: 被 s0, s1, s2, s3, s4 占据 (5/6) → mean > 0.5 → 误判 base
voxel 5: 被 s1, s2, s3, s4, s5 占据 (5/6) → 同上误判
```

**BMCSA 下的后果**：
- State 0 在 voxel 5（不是 state 0 drawer 的位置）被 BMCSA 强拉去占据 → **额外多出不该有的 voxel**
- State 5 在 voxel 4（不是 state 5 drawer 的位置）同理

**严重程度**：污染范围极小（~2 voxel），视觉和下游 Stage C 应该都感知不到，目前**实际输出是对的**。

**为什么 revolute 不出现这个问题**：旋转关节下相邻 state 的 move 部分空间覆盖几乎互斥，trajectory voxel 最多被 1-2 个 state 覆盖，mean ≤ 2/6 < 0.5，不触发 base 阈值。

**已探索的修复方向（均未实现）**：
- Connected component filter：drawer 和 cabinet 在 slot 处连通，无法切分 → **否决**
- Shift-pattern detection：沿轴检测 k-shift 相关 neighbor pattern → 可行但 ~50 行代码
- Variance threshold：利用 TRELLIS 软/硬 0 区分 → 实证依赖
- **Attention map agreement（方案 3）**：DiT self-attn features 在 cabinet voxel 上跨状态相似，在 drawer trajectory voxel 上有显著差异 → **正在设计实现**

**方案 3 的核心洞察**：DiT 在 cabinet voxel 上的 hidden feature 跨状态是相似的语义（"柜体边界"），而在 drawer voxel 上 state 0 是"空气"、其他是"抽屉"，跨 state feature 显著不同。用这个 feature-level 跨 state agreement 作为 **动态 M_attn**，和几何 M_base 相乘，修正污染 voxel。

**延迟或处理决定**：当前先接受 2 voxel 污染；实现方案 3 看能不能在 Stage B 内闭环解决；如果 Stage C 的 canonical EM 能鲁棒处理，就作为 backup。

### 问题 11 (mild / P_excl 设计缺陷)

**描述**：在平移下 `P_excl_k = ReLU(P^(k) - max_{j!=k} P^(j))` 只能捕捉**每个 state 相对其他 state 新增的 voxel**（比如 state 1 的 drawer 只有最外那个 voxel 是 P_excl_1 非零；中间状态 state 2 P_excl 可能几乎全为 0 因为被 state 1 和 state 3 的 drawer 完全覆盖）。

**为什么目前结果还是对的**：Pass 2 去噪阶段 BMCSA 只在 M_base 高处 blend，中间 state 的 drawer 靠 c_k conditioning + DiT 重生，不依赖 P_excl_k。

**不需要修复**：属于 graceful degradation，机制本身没错。

### 问题 13 (发现于 v4.2 实验): guide 里 P_base 过大，侵占 move voxel（**所有 sample 都有，26525 极端**）

**实验场景**（2026-04-18, v4.2 运行）：
- 2 平移: `30857`, `26525`
- 2 旋转: `7128`, `7201`
- 结果目录：`outputs/experiment/<sample>_ditnew/`

**视觉结果**：
- 30857 (平移): 好
- 26525 (平移): **state 0,1,2 近乎完全相同**（严重 failure）
- 7128 (旋转): 好
- 7201 (旋转): 好

**诊断（用户观察）**：
打开 `viz/guide/state{k}_O_guide_bin.html` 能看到：
- 所有 state 的 `O_guide_bin` 中，**base 部分（shared across k）已经占据了 drawer 该在的区域**
- 26525 case 特别严重，因为平移 trajectory 非常长 + 平移量小 → mean over K 在 trajectory 中段非常高 (5/6 或 6/6) → 误入 base
- 其他 sample 虽然也有 over-large base，但没极端到视觉 collapse

**根本原因（数据流分析）**：

```
z_final ─────────────────────────────────────────┐
                                                  ↓
                                           M_attn_16 (1, 4096, 1) 16³
                                                  │
                                                  └─► 只在 BMCSA blend 阶段和 M_base_16 相乘
                                                      （Pass 2 DiT 每层 self-attn 内部）

soft_p1 ─► P_base_shared_64 ─► P_guide_k ─► z_guide_k ─► Pass 2 起点
              (64³ mean)        (max with         (encoder)
                                 per-state
                                 P_excl_k)
                  ↑
           **这一步完全不受 M_attn 影响**
```

- M_attn 在 16³ token 空间，只在 BMCSA attention 层过滤 blend 权重
- guide 构造在 64³ 空间，**P_base_shared_64 没有任何 semantic 过滤**
- z_guide_k 从 over-large P_base + 微小 P_excl_k 编码出来
- Pass 2 从 z_guide_k + noise 去噪 12 步，但**起点的几何偏差已经进入 latent trajectory**
- 在 26525 极端情形下，6 个 state 的 z_guide_k 几乎相同（base 主导，P_excl 微小）→ Pass 2 的 12 步无法"把起点错误拉回来"→ s0-2 收敛到同一 output

**为什么旋转不触发**：
旋转的 trajectory 空间覆盖互斥（旋转角度 sweep 快），mean over K 在 trajectory voxel 上 ≤ 2/6 = 0.33 < 0.5 → voxel 不入 base → guide 的 base 部分不含 drawer → 起点正确。

**修复方案（v4.3）**：**把 M_attn 上采样到 64³ 并前置到 guide 构造**。

```python
# v4.3 新增步骤
M_attn_16 = compute_from(z_final)            # (1, 4096, 1) @ 16³
M_attn_16_spatial = M_attn_16.view(1, 1, 16, 16, 16)
M_attn_64 = F.interpolate(M_attn_16_spatial, size=(64, 64, 64), mode='trilinear')
M_attn_64 = M_attn_64.squeeze()              # (64, 64, 64)

# 过滤 P_base 源头
P_base_shared_64_filtered = P_base_shared_64 * M_attn_64

# 下游所有用 P_base 的地方都用 filtered 版本：
#   - _build_augmented_intersection_guide 的 P_base_shared 参数
#   - _compute_M_base_tokenspace 的 P_base_shared 参数
```

**为什么前置（guide）而不是 downstream（BMCSA）**：

1. **源头修正**：guide 错了，Pass 2 的 12 步去噪就算有 BMCSA 也拉不回（26525 case 证明）
2. **z_guide_k 编码正确的几何**：Pass 2 起点的 latent 代表真正的 "shared base + per-state excl"，不会把 K 个 state 起点拉平
3. **单点真相**：整个 Pass 2 流程里只有**一个 M_base 概念**（已融合几何+语义），代码路径更清晰
4. **CVPR/AAAI story 更干净**："几何一致"和"语义一致"**融合在同一个 M_base 里**，不是两套独立 mask 叠加

BMCSA block 里不再需要额外应用 M_attn（M_base_16 已经 encode 了 M_attn 的效果）。旧的 BMCSA M_attn kwarg 保留作 ablation 开关。

**v4.3 预期修复**：
- 26525: drawer 中段 voxel 被 M_attn_64 过滤掉 → 不入 P_base → guide 只含真正柜体 → state 0-5 起点的 excl 差异恢复 → output 分离
- 30857/7128/7201: 原本就好；v4.3 不破坏它们（M_attn 对这些 case 的 drawer 区域仍然低，非 drawer 区域仍然高）

### 问题 12 (未验证): TRELLIS 软/硬空输出分布未验证

**场景**：决定方案 1（软 state-0 veto）是否可行的关键实验。

**需要**：对一个已知有 shrunk shell 的样本，采样 100 个 shell voxel 和 100 个 drawer trail voxel，画 P^(0) 分布。如果两组分布**不重叠**（shell 在 0.1-0.4，trail 在 0.0-0.05），方案 1 可直接 work。

**未执行**：用户选择直接上方案 3，跳过方案 1 的可行性检验。

---

## 7. 方案 3（v4.2 已实现 + v4.3 修复）

**方向**：Attention-Map-Driven Base/Move Discrimination

**核心算法**：
1. Pass 1 结束后（或在 Pass 1 的某个中间 step），从 DiT 的某个中间 block 抽取**per-state 的 spatial token hidden features** h_k ∈ R^{L × D}
2. 对每个 token 位置 ℓ，计算跨 state 的 feature 余弦相似度 `sim(ℓ) = mean_{k != k'} cos(h_k(ℓ), h_k'(ℓ))`
3. 高 sim → cabinet-like（跨 state 语义一致），低 sim → move-like（跨 state 语义不同）
4. `M_attn(ℓ) = sigmoid((sim(ℓ) - thresh) / τ)` 作为 attention-driven base mask
5. 最终 BMCSA 用 `M_combined = M_base × M_attn` 作为 blend 权重（几何 AND 语义都同意才 blend）

**为什么这能修问题 10**：
- Shrunk shell voxel：state 0 的 feature 和 s1-5 的 feature 都处于"柜体边界"语义邻域，sim 高，M_attn 高 → 保留为 base → BMCSA 正常拉 state 0 ✓
- Drawer trail voxel：state 0 feature ="air/air"，s1-5 feature ="drawer material"，sim 低，M_attn 低 → 从 base 中排除 → BMCSA 不拉 state 0 ✓

**文献锚点**：
- Prompt-to-Prompt (Hertz et al. ICLR 2023, arXiv:2208.01626): attention maps 可以语义编辑
- MasaCtrl (Cao et al. ICCV 2023, arXiv:2304.08465): mutual self-attention across ref/target, training-free
- TokenFlow (Geyer et al. ICLR 2024, arXiv:2307.10373): cross-frame token matching for video consistency

**v4.2 实现**（已完成）：
- `_compute_M_attn_tokenspace(z_final, threshold=0.7, tau=0.05)` 返回 (1, 4096, 1) 在 16³ token 空间
- `ModulatedTransformerCrossBlock._forward` 里 `M = M_base × M_attn` 当 M_attn in kwargs
- Config `attn_m_enabled=true, attn_m_threshold=0.7, attn_m_tau=0.05`

**v4.2 实验结果**：
- 4 样本：3/4 视觉良好（30857, 7128, 7201），1/4 失败（26525，s0-2 collapse）
- 失败分析见 §6 问题 13
- viz 诊断在所有 case 都发现 `guide_bin` 的 base 占据 move voxel（严重度不同）

**v4.3 实施**（进行中）：上采样 M_attn 到 64³ 并前置到 guide 构造。见问题 13。

---

## 8. 架构决策总结（供 paper methods 复用）

1. **两阶段架构**：Pass 1 做粗对齐（mix），Pass 2 做精修（SDEdit + BMCSA）—— 和 SDEdit (Meng ICLR 2022) + MVDream (Shi ICLR 2024) 家族对齐
2. **Pass 1 shared noise 在 Pass 2 继承**：K-parallel 的"同一随机家族"不变量不破坏
3. **Per-state z_guide_k = shared base + state-specific excl**：每个 state 的 Pass 2 起点含自己的 drawer hint
4. **BMCSA 是 post-attention blend，不是 K/V concat**：遵循 MorphAny3D (Sun et al. 2026, arXiv:2601.00204) §Moving Cross-Attention 的设计原理（pre-attention KV concat 会引入 OOD attention pattern）
5. **Source edit + kwargs，不 hook 不 monkey-patch**：遵循 MorphAny3D / MasaCtrl 的惯例
6. **Encoder 专门加载**：不依赖默认 pipeline.json
7. **`exclude_state_0: false`（mean over all K）**：1/K 稀释 state 0 bias，不排除其贡献

---

## 9. 下一步

1. **实现方案 3**：attention feature agreement 作为动态 M_attn
2. **跑实验**对比 v4.1 与 v4.2：在 PARIS / PartNet-Mobility 典型 articulation 上看 state 0 scale 修复程度
3. **Stage C 集成验证**：看 v4.1 输出是否足够干净让 Stage C canonical EM 起作用
4. **Failure case 整理**：paper §限制 准备写 "long-drawer-short-shift" pathological case

---

_End of stageb_detail.md_
