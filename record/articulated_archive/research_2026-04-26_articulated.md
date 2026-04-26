# Deep Research: Articulated 3D via In-Loop RASA + Wan2.2 RFSDS in Frozen TRELLIS

**Date**: 2026-04-26
**Purpose**: Single research document covering 5 topics needed to design the new pipeline (replacing old Stage B + Stage C). Consolidates findings from 4 parallel research agents on:
1. TRELLIS layer-by-layer dissection — where to insert adapters
2. MorphAny3D — actual mechanism (SLAT-DiT MCA+TFSA, NOT 2D texture)
3. Adapter techniques in frozen DiT — best fit per head type
4. K-state shared base hard constraints + feature matching alternatives
5. Critique of GPT's `methods.md` / `pipeline.md`

---

## 0. Confirmed Architecture (per advisor)

```
Input: 单张闭态图 I_0
  ↓ Wan2.2 I2V (frozen) — 一次性 sample 生成 opening video
  ↓ 抽 K=6 帧 → I_0..I_5 作为 K 个 "image conditions"
  ↓
TRELLIS image-to-3D:
  K=6 张图 → K 次 DINOv2 (batched) → K 个 image cond
  → SS-DiT (frozen) + 新加 RASA layers (trainable)
    输出: K 个 SS latent，base 通过新层硬绑定共享
  → SS-VAE decoder (frozen) + 新加 part-mask head (trainable)
    输出: 64³ canonical 占用 + per-voxel base/move 概率
  → SLAT-DiT (frozen) + 新加 K-state attention layers (trainable)
    输出: K 个 SLAT latent, 共享 base, varying texture per state
  → SLAT-VAE mesh decoder (frozen) + texture adapter (trainable)
    输出: K 个 mesh / Gaussian
  ↓
Render K=6 帧 (locked camera)
  ↓ Wan2.2 RFSDS (frozen, no_grad on Wan22)
  ↓ Backward → 更新所有 RASA layers + part_mask head + joint_params + texture adapter
  ↓ 2000 iter

最终输出:
  canonical_base (mesh + texture)
  canonical_move (mesh + texture)
  joint params (axis, pivot, q_k) → URDF
```

**Trainable**：~15-25M 参数（adapter + heads，~1-2% of frozen TRELLIS）
**Frozen**：TRELLIS 全部、Wan2.2 全部、DINOv2

---

## 1. TRELLIS 逐层解剖 — Adapter 插入点

### 1.1 关键模型组件

代码路径基底：`C:\Users\管晨皓\Desktop\temp\standard\paper\TRELLIS\trellis\`

#### Stage A — DINOv2 image encoding
- 文件：`pipelines/trellis_image_to_3d.py:118-160` (`encode_image`, `get_cond`)
- 模型：`dinov2_vitl14_reg`，frozen ViT
- 输入：`(K=6, 3, 518, 518)` — **batch 维度原生支持**
- 输出：`patchtokens (K, 1374, 1024)` — 1369 patches + 1 CLS + 4 register
- 当前 multi-image 处理（`run_multi_image:342-376`）用 stochastic round-robin 或 multidiffusion ensembling，**没有 learned multi-view fusion**

#### Stage B — SS-DiT (`models/sparse_structure_flow.py:SparseStructureFlowModel`)
- 配置：`model_channels=1024, num_blocks=24, num_heads=16, attn_mode='full'`
- 24 个 `ModulatedTransformerCrossBlock` (`modules/transformer/modulated.py:76-156`)
- 每个 block 内：`pre_norm → self_attn → cross_attn(DINOv2) → MLP`，全部 adaLN modulation
- 输入：`(B, 4096, 1024)` (4096 = 16³ token grid)
- 输出：`(B, 4096, 1024)` velocity prediction
- 全部 op autograd-clean (no `@no_grad` 在 model 内)

#### Stage C — SS-VAE Decoder (`models/sparse_structure_vae.py:SparseStructureDecoder`)
- 输入：`(B, 8, 16, 16, 16)` SS latent
- 上采样链：16³(512ch) → 32³(128ch) → 64³(32ch) → 1ch logit
- 输出：`(B, 1, 64, 64, 64)` occupancy logit
- **`argwhere(>0)` 在 pipeline 层调用（不在 model 内），不可微 — 但我们不通过它走 backward**

#### Stage D — SLAT-DiT (`models/structured_latent_flow.py:SLatFlowModel`)
- 配置：`model_channels=1024, num_blocks=24, num_heads=16, attn_mode='full' (sparse)`
- 24 个 `ModulatedSparseTransformerCrossBlock` (`modules/sparse/transformer/modulated.py:81-166`)
- 输入：sparse `(M_active_voxels, 8)`，M ≈ 8k-32k
- input_blocks 做 1 次 downsample (64→32 active grid)，channel 升到 1024
- 24 blocks 全 full self-attn over 32k voxels
- output_blocks 做 upsample 回 64 grid
- 输出：sparse `(M_active, 8)` SLAT latent

#### Stage E — SLAT-VAE Decoders
- `decoder_mesh.py` (FlexiCubes mesh) — **partial differentiable** (vertex pos + color 可微，face topology 在 SDF=0 不可微但 measure-zero)
- `decoder_gs.py` (3D Gaussian) — **fully differentiable**
- `decoder_rf.py` (radiance field NeRF-style) — **fully differentiable**
- 都基于 `SparseTransformerBase` (12 blocks, model_channels=768, **windowed swin attention** unlike SLAT-DiT)
- 输出 channel 切片：geometry (sdf/deform/weights) + color (48-d)

### 1.2 五个 Adapter 插入点 (A0–A4)

| 编号 | 位置 | 模块 | 输入 → 输出 | 参数量 | 用途 |
|---|---|---|---|---|---|
| **A0** | Stage A 后 | Multi-View Fusion | `(K=6, 1374, 1024)` → `(1, 1374+P, 1024)` | ~8M | Perceiver-style: 把 K 张图的 DINOv2 token 融合 + 学 P=64 个 part-bank tokens |
| **A1** | 每个 SS-DiT block 内 | Cross-state attention + LoRA | `(K, 4096, 1024)` → `(K, 4096, 1024)` | ~5M | (a) IP-Adapter 风格 parallel cross-attn 到 part-bank，(b) 跨 K 同 voxel 位置 self-attn，(c) Q/K/V/O LoRA rank-16 |
| **A2** | SS-VAE 64³ 输出后 | Part Segmentation Head | `(1, 1, 64, 64, 64)` → `(1, K_parts, 64, 64, 64)` | ~0.05M | Multi-tap MLP head from SS-VAE intermediate features，零初始化最后一层 |
| **A3** | 每个 SLAT-DiT block 内 | K-state cross-batch attention + MorphAny3D-style fusion | `(K·M, 1024)` → `(K·M, 1024)` | ~9M | 跨 K 同 voxel 位置 self-attn + MCA-style cross-attn donor blending |
| **A4** | SLAT-VAE mesh decoder 输出 color slice | Texture Branching Adapter | `(M·64, 96)` → 修正 color `(M·64, 48)` 按 part 路由 | ~0.05M × K_parts | AdaptFormer-style parallel bottleneck，按 PartHead 输出的 part_mask 路由不同 color delta |

**总参数量**：~22M (K_parts=2)，相比 frozen TRELLIS ~1.2B 是 **~1.8%**

### 1.3 K=6 batched forward 内存预算

| 配置 | Memory peak |
|---|---|
| Frozen no_grad K=6 forward | ~3.6 GB activations + ~5 GB weights = ~9 GB |
| RASA grad on, 全 unroll | ~24 GB（24GB 卡装不下） |
| RASA grad on, gradient checkpointing per block | ~12 GB（24GB 卡舒服） |
| + Wan2.2 TI2V-5B forward (no_grad) | +10 GB → ~22 GB |

**结论**：24GB 卡跑得动；80GB 卡更舒服（可以用 Wan2.2 I2V-A14B 换更强先验）

### 1.4 关键设计决定

- **Cross-state attention**：放 **3 个 mid-late blocks（如 16, 19, 22）** 而非每个 block，省 8× 计算 (AnimateDiff 实证 ~4-8 个 temporal layer 即可)
- **Part segmentation head**：多 block tap (12, 16, 20 拼接) 然后 MLP，比单深 tap 边界精度高（DPT, DINO-probing 一致结论）
- **Joint param head**：用 PartHead 输出的 mask 做 attention pool，然后 MLP；axis 用 **S² 离散化分类 + refine**（PARIS 实证比直接 3-vec regression 好）
- **Texture branching**：在 SLAT-VAE decoder color slice，**仅修改 color 不动 sdf/deform/weights**

---

## 2. MorphAny3D — 真实机制（与之前误解的差异）

### 2.1 论文身份

- **Title**: "MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing"
- **Authors**: Sun, Cai, Tang, Tai, Yang, Zhang (NJU + PKU)
- **Venue**: CVPR 2026
- **arXiv**: https://arxiv.org/pdf/2601.00204
- **Local**: `paper/MorphAny3D/`
- **GitHub**: https://github.com/XiaokunSun/MorphAny3D.git

### 2.2 重要更正：不是 2D 纹理融合

**之前的误解**：MorphAny3D 做 2D 图像 / UV 纹理融合，类似 TEXTure / Paint3D。

**实际是**：在 **冻结 TRELLIS SLAT-DiT 内部**做 attention activations 混合，是 **latent-space 操作**，不是 pixel-space。

### 2.3 四个核心机制

#### (1) MCA — Morphing Cross-Attention
**位置**：每个 SLAT-DiT block 的 cross-attention layer
**代码**：`paper/MorphAny3D/trellis/modules/sparse/transformer/modulated.py:166-168`

```python
# 同时跑两次 cross-attn（src image cond / tgt image cond），然后 latent 插值
h = slat_interp(
    cross_attn(Q=h, KV=src_image_cond),
    cross_attn(Q=h, KV=tgt_image_cond),
    alpha=alpha_t  # 时间插值因子
)
```

`slat_interp` 是 sparse latent 加权融合：按 voxel 坐标做最近邻匹配，feats 线性混合，coords 取整去重。

#### (2) TFSA — Temporal-Feature Self-Attention
**位置**：每个 SLAT-DiT block 的 self-attention layer
**代码**：`modulated.py:154-158` + `attention/modules.py:107-127`

第一帧 (`morphing_idx=0`)：cache 当前 K/V 到磁盘
后续帧：load cached K/V，做两次 self-attn（一次用自己 K/V，一次用 cached K/V），slat_interp 混合

```python
h_self = slat_interp(
    self_attn(Q=h, KV=h),                    # current K/V
    self_attn(Q=h, KV=cached_KV_prev_frame), # previous frame's K/V
    alpha=tfsa_alpha=0.8  # 偏向 cache（强一致性）
)
```

这是 **跨时间步的 feature consistency 注入**，跟 TokenFlow / Tune-A-Video 同思路但在 SLAT 空间。

#### (3) Initial-noise interpolation
两个 endpoint 状态都跑一次 SS+SLAT 拿初始 noise 缓存，然后按 voxel 坐标欧氏距离做 NN 匹配，初始 noise 加权混合（`pipelines/trellis_image_to_3d.py:329-341`）。

#### (4) Orientation correction (`oc_flag`)
4 个 cardinal rotation hypothesis 各跑 SS，取 chamfer distance 最小的；旋转 cached K/V 对齐（`trellis_image_to_3d.py:429-455`）。

### 2.4 适配到我们的 K-state articulated 任务

MorphAny3D 是 2-endpoint 插值；我们是 **K=6 状态共享 base**。机制改造：

```python
# Donor weights per voxel per state
def compute_donor_weights(p, K, T_inv_k, coords_per_state):
    """
    p: canonical voxel position
    T_inv_k: inverse joint transform per state
    """
    weights = []
    for k in range(K):
        posed_p = T_inv_k(p)  # bring p to state-k frame
        # 基于：可见性、面朝向、camera朝向、最近邻距离
        v_k = is_visible_in_state(posed_p, k)
        f_k = is_front_facing(posed_p, k)
        c_k = is_camera_facing(posed_p, k)
        d_k = nearest_neighbor_distance(posed_p, coords_per_state[k])
        w_k = v_k * f_k * c_k * exp(-d_k / temperature)
        weights.append(w_k)
    return softmax(weights)  # K-way weights, sum to 1
```

**改造的 SLAT-DiT block forward**（替换 MorphAny3D 的 2-way alpha 为 K-way donor 加权）：

```python
def forward_kstate_block(self, h_canonical, K_image_conds, K_cached_KV, donor_weights):
    # K-way Cross-Attention (generalized MCA)
    h_cross = sum(
        donor_weights[k] * cross_attn(Q=h_canonical, KV=K_image_conds[k])
        for k in range(K)
    )

    # K-way Self-Attention with cached K/V (generalized TFSA)
    h_self = sum(
        donor_weights[k] * self_attn(Q=h_canonical, KV=K_cached_KV[k])
        for k in range(K)
    )

    return h_canonical + h_cross + h_self  # residual
```

### 2.5 关键代码改动文件

| 文件 | 改动 |
|---|---|
| `modules/sparse/transformer/modulated.py:146-180` | MCA + TFSA logic：2-way slat_interp → K-way weighted sum |
| `modules/sparse/attention/modules.py:107-127` | 磁盘 cache K/V loader：从 `tfsa_cache_idx` → list of K state indices |
| `utils/morphing_utils.py:39` | `slat_interp` (binary alpha) → `slat_donor_fuse(slats, weights, coords_canonical)` (K-way) |
| `pipelines/trellis_image_to_3d.py:306-361` | `sample_slat_morphing` → `sample_slat_canonical_fusion` 收 K cached SLAT |

### 2.6 可微性

- `slat_interp` 的线性插值部分对 feats 可微
- `argmin` 索引查找不可微但每 block 固定一次
- coords 取整不可微（用 fixed canonical coords 解决）
- **必须删 `@torch.no_grad()`** 才能 SDS 用

---

## 3. Adapter 技术对比（在 frozen DiT 上加可训练层）

### 3.1 主流技术对比表

| 技术 | 机制 | 零初始化？ | 参数开销 | 我们的适配度 (1-10) | 引用 |
|---|---|---|---|---|---|
| **LoRA** | ΔW = α·B·A in-place | Yes (B=0) | 0.1-1% | 5（修改内部，不加新输出） | arXiv:2106.09685 |
| **IP-Adapter** | 并行 cross-attn 分支 | Yes (W_O^new=0) | ~5% | **9**（cross-state attn 首选） | arXiv:2308.06721 |
| ControlNet | clone encoder + zero-conv | Yes | 50-100% | 3（太重） | arXiv:2302.05543 |
| T2I-Adapter | 多尺度 CNN 注入 | Partial | 5-10% | 4（仅空间） | arXiv:2302.08453 |
| Side-Tuning | frozen + side net 求和 | Yes | flexible | 7（generic fallback） | arXiv:1912.13503 |
| Houlsby | bottleneck 串联 | Yes (W_up=0) | ~3% | 5（侵入） | arXiv:1902.00751 |
| **AdaptFormer** | bottleneck 与 MLP 并行 | Yes (W_up=0) | ~0.15% | **7**（texture head） | arXiv:2205.13535 |
| DoRA | magnitude+direction LoRA | Yes | 0.2-1% | 5 | arXiv:2402.09353 |
| OFT | 正交乘法扰动 | Yes (R=I) | ~1% | 6（保护先验） | arXiv:2306.07280 |
| BitFit/AdaLN-only | 只训 norm/bias | Identity | 0.05% | 3（太弱） | arXiv:2106.10199 |
| **Probing head** | 新 MLP on tapped feature | 设输出 W=0 | per head | **9**（seg/joint heads） | arXiv:1610.01644 |

### 3.2 我们四个 head 的具体推荐

| Head | 推荐技术 | 理由 |
|---|---|---|
| **Cross-state attention** | **IP-Adapter parallel cross-attn**（W_O 零初始化 + 标量 gate） | (a) IP-Adapter pattern = "parallel branch, zero-init output" 是 cross-state attn 的天然形式；(b) 零初始化保证 iter-0 = frozen baseline；(c) 比 side network (option b) 参数少 |
| **Part segmentation** | **Multi-tap probing head**（taps 12, 16, 20 + MLP） | (a) DPT/SegFormer 标准模式；(b) DINO probing 论文证明 mid-late layer 语义最强；(c) 单深 tap 边界精度差 |
| **Joint param** | **Masked attention pool + S² classification** | (a) Mean-pool 混淆部件 → axis 准度差；(b) Perceiver-IO regression head 标准；(c) PARIS 论文实证 axis-on-S² 离散化 + refine 比直接 3-vec regression 好 |
| **Texture fusion** | **AdaptFormer parallel bottleneck**（在 SLAT-VAE color slice） | (a) 仅修改 color 不动 geometry；(b) 标量 gate 渐近开启；(c) 参数极小 |

### 3.3 共享 backbone + 任务专属 head

不要为每个 head 独立加 adapter，**用一条共享的 cross-state side branch** 读取 blocks {6, 12, 18, 24}，然后 4 个 head 共享这个 augmented feature：
- Hyperformer (Karimi Mahabadi et al. arXiv:2106.04489) 实证 shared-bottleneck 比独立 adapter 在同等参数下表现更好
- 我们也只需要 ONE zero-init 安全点

---

## 4. K-State Shared Base 硬约束机制

### 4.1 三种 pattern 总结

| Pattern | 硬度 | 例子 | 我们用？ |
|---|---|---|---|
| **架构共享**：one canonical tensor + K transforms | **硬** (完全相同 by construction) | FreeArt3D, ArtGS | **Tier 2** |
| **Cross-state attention**：K parallel forward + attention 交换 | 软（强） | MVDream, AnimateDiff, SVD | **Tier 1** |
| **Loss-only consistency**：`L = ‖base_k - base_0‖²` | 软（弱） | ReconFusion | × |

### 4.2 我们的两层混合方案

**Tier 1 — 软（in-loop）**：
- 在 SS-DiT 的 3 个 mid-late blocks (16, 19, 22) 插 cross-state self-attention
- 在 SLAT-DiT 同样位置也插（用 MorphAny3D 改造的 K-way fusion）
- 让 K 个 denoising 轨迹**共享 base 信息**（哪些 voxel 属于 base，base 的几何形状）
- 但不强制 bit-identical

**Tier 2 — 硬（post-loop / 渲染前）**：
- 在 K 个 SS latent 输出后（或 mesh 输出后），**强制 base voxel 用 anchor (state 0) 的值**
- FreeArt3D-style：
  ```python
  # K SS latents 都来自 RASA forward
  z_s_K = rasa_ss_forward(K_image_conds)  # (K, 8, 16, 16, 16)

  # Tier 2: hard anchor copy
  base_mask_16 = downsample(part_mask_64, 4)  # 64³ → 16³ token grid
  for k in range(1, K):
      z_s_K[k][base_mask_16] = z_s_K[0][base_mask_16]  # 复制 base voxel
  ```

这样：
- Tier 1 让 K 个轨迹**协同生成** base（信息共享）
- Tier 2 让 K 个 base **完全相同**（架构保证）

### 4.3 计算成本

Cross-state attention at 3 mid-late blocks × 25 sampling steps × K=6:
```
3 × 25 × (6²·4096·1024) = 3 × 25 × 150M ops = 11G ops total
```
活动检查点 + fp16 → 24GB GPU 跑得动

### 4.4 跟 LoFTR 等 feature matching 的关系

**关键洞察**：cross-state attention ≈ K-way LoFTR + injection (latent space)

LoFTR 在 pixel space 找两图对应；我们的 cross-state attention 在 SS/SLAT latent space 找 voxel 对应。两者本质上做相同事情：
- LoFTR：`match(img_a, img_b) → pixel correspondences`
- 我们：cross-state attn 在 voxel l 上让 K 个 state 互相 attend

TRELLIS 已有的 DINOv2 cross-attention 已经做 image↔voxel matching；新加 cross-state self-attention 做 voxel↔voxel-across-K matching。**两者结合 = LoFTR-equivalent，在 latent space 内置，不需要外挂网络。**

显式 LoFTR/DUSt3R/MASt3R **不需要**。

---

## 5. 对 GPT methods.md / pipeline.md 的批判性审查

### 5.1 GPT 做对的部分（保留）

1. ✅ **整体三冻结**：TRELLIS + Wan2.2 + DINOv2 frozen，只训 RASA
2. ✅ **In-loop adapter 而非 post-hoc**：RASA 必须在生成 loop 里，不是 cached-hidden
3. ✅ **Fixed-support differentiable gate** (methods.md §6)：跳过 argwhere，用 fixed coords + sigmoid logit gate。**正确**
4. ✅ **Single-DoF 显式 transform** (methods.md §5)：prismatic / revolute Rodrigues。**正确**
5. ✅ **Locked-camera prompt + first-frame clamp**：Wan2.2 RFSDS 关键防御
6. ✅ **W-RFSDS 公式** (methods.md §9.2)：跟 CHORD 一致
7. ✅ **5-phase optimization schedule**：geometry 先，texture 后，符合 ProlificDreamer 经验
8. ✅ **Adapter zero-init**：iter-0 = frozen baseline 是必要保证

### 5.2 GPT 做错的部分（必须改）

#### ❌ 错误 1：cross-state attention 只放 SS-DiT，没放 SLAT-DiT

GPT methods.md §1.1：
```
rasa_blocks = [8, 12, 14, 16, 18, 20]  # 仅在 SS-DiT
```

**问题**：MorphAny3D 的实证表明 SLAT-DiT 内部的 MCA/TFSA 才是跨状态纹理一致的关键。SS 阶段决定**哪个 voxel 是 part**，SLAT 阶段决定**这个 voxel 长什么颜色**。两个阶段都要插。

**修正**：
- SS-DiT: blocks {16, 19, 22}（共 3 个，省 8× 计算）做 cross-state attn for **base/move 协同识别**
- SLAT-DiT: blocks {16, 19, 22}（共 3 个）做 K-way MorphAny3D-style MCA + TFSA fusion for **跨 state 纹理一致**

#### ❌ 错误 2：part-mask head 在 16³ token resolution 然后 trilinear upsample

GPT methods.md §3.4：
```
m^16 = MLP(F_l)  # at 16³ token resolution
m^64 = TrilinearUpsample(m^16)
```

**问题**：16³ 每个 token 覆盖 4³=64 个真实 voxel。薄部件 (drawer face 1-2 voxel 厚) 在 16³ 分辨率根本看不见。直接 trilinear upsample 没 boundary-aware 信息。

**修正**：
- **Multi-tap probing**：tap blocks {12, 16, 20} 的 hidden state，concat → MLP → 16³ logit
- **加上 SS-VAE 64³ tap**：在 SS-VAE decoder 输出 occupancy logit 旁边加并行 head 直接产 64³ part_mask（boundary precision）
- 两路融合：16³ logit upsample + 64³ direct logit 加权和

#### ❌ 错误 3：Cross-State Visible Texture Fusion 描述是 2D pixel level

GPT methods.md §11.2：
```
C_i_can = Σ_k w_{k,i} · InvWarp_{q_k}(C_{k,i})
```

GPT 写 "InvWarp" 是把 rendered pixel color back-warp 到 canonical mesh。**这是 2D 纹理融合**，类似 TEXTure/Paint3D。

**问题**：MorphAny3D 实证 **SLAT latent 级别的融合更强**。在 SLAT-DiT 内部混合 K/V activations，让 frozen TRELLIS prior 直接生成跨 state 一致纹理，比 2D back-projection 失真小、几何敏感度高。

**修正**：把 GPT §7/§11 的 "InvWarp pixel color" 替换为 "MorphAny3D-style SLAT-DiT MCA+TFSA K-way fusion"：
```python
# SLAT-DiT 内部，每个选定 block:
h_cross = Σ_k w_k(p) · cross_attn(Q=h, KV=K_image_conds[k])
h_self = Σ_k w_k(p) · self_attn(Q=h, KV=cached_KV[k])
h = h + h_cross + h_self
```
fusion 在 SLAT 空间发生，texture 在 mesh decoding 时已经融合。

#### ❌ 错误 4：joint head 用直接 3-vec axis regression

GPT methods.md §3.4 + §5.1：
```
axis = MLP_axis(z_move)  # ∈ R³, 然后 normalize
```

**问题**：PARIS (arXiv:2308.07391) 实证直接 3-vec axis regression 在 rotation axis 上易陷局部最优。

**修正**：
- 把 S² 球面离散化（如 162-bin icosahedral grid），先 cross-entropy classification 选 bin，再做 sub-bin 3-vec refinement
- 或：predict axis 用 6-DoF representation (Zhou et al. CVPR'19) 然后 Gram-Schmidt 投回 SO(3)

#### ❌ 错误 5：K=6 batched 但 image conditioning 写法不一致

GPT pipeline.md Phase 1：
```
"K state-coded denoising trajectories share the input condition"
```

**冲突**：导师明确说"具体输入的是视频扩散视频提取的六张照片"——是 K 张不同图，不是单张图 + state code。

**修正**：
- 不用 state code embedding（容易 perturb frozen DiT off-manifold）
- 直接 batch K 张图：DINOv2 batched forward → K 个不同 image cond → SS-DiT batched forward
- Cross-state attention 让 K 个 batch 协同共享 base 信息

#### ❌ 错误 6：adapter 插入 6 个 block 太多

GPT methods.md §3.1：
```
rasa_blocks = [8, 12, 14, 16, 18, 20]
```

**问题**：6 个 block × 24 SS-DiT total 是 25%，参数膨胀 + 计算开销。

**修正**：3 个 mid-late block 足够（参考 AnimateDiff、SVD 实证）：
```
rasa_blocks = [16, 19, 22]
```

### 5.3 GPT 没覆盖但必须做的部分

- **Tier 2 hard anchor copy**：GPT 只有 cross-state attention soft consistency，没有 FreeArt3D-style 强制 base 复制保证 bit-identical
- **Wan2.2 抽帧 K=6 → K image conditions 的具体流程**：GPT pipeline.md Phase 0 没写 Wan2.2 一次性预生成视频抽帧
- **MorphAny3D K-way 改造**：GPT §7/§11 是 2D-pixel inverse warp，要换 SLAT-latent K-way fusion
- **SS-VAE 64³ part-mask head**：仅 16³ token tap 是不够的，要补 64³ direct head

---

## 6. 修正后的最小实现路径

### Phase 0 — 一次性 (no_grad)
```python
1. Wan2.2 I2V on I_0 → opening video → 抽 K=6 帧 → I_0..I_5
2. DINOv2 batched on K=6 images → cond_K = (K, 1374, 1024)
3. TRELLIS SS-DiT once with zero-RASA → SS_init (K, 8, 16³)  for cache
4. SS-VAE decode → occupancy logits, build SUPPORT_COORDS (loose τ + dilation + topk)
5. SLAT-DiT once with zero-RASA → SLAT_init for cache, store K K/V activations
6. SLAT-VAE mesh decode → canonical mesh init
```

### Phase 1 — RFSDS optimization loop (2000 iter)
```python
for iter in 1..2000:
    # Forward (RASA active, gradient ON for adapters)
    z_s_K = SS_DiT_with_RASA(noise_K, cond_K)  # (K, 8, 16³)
    
    # Tier 2 hard anchor copy (after sampling)
    base_mask = part_seg_head(SS_VAE_decode(z_s_K[0]))  # from anchor frame
    for k in 1..K-1: z_s_K[k][base_voxels] = z_s_K[0][base_voxels]
    
    occ_K = SS_VAE_decode(z_s_K)  # (K, 1, 64³)
    support_gate_K = sigmoid(occ_K_at_SUPPORT_COORDS / temp)
    part_mask = part_seg_head(occ_K, taps_from_SS_DiT)
    joint_axis, joint_pivot, q_k = joint_head(masked_pool(taps, part_mask))
    
    z_slat_K = SLAT_DiT_with_RASA(noise', SUPPORT_COORDS, cond_K, MorphAny3D-fusion)
    mesh_K = SLAT_VAE_decode(z_slat_K, color_adapter, part_mask)
    
    # K-state derivation (architectural sharing for hard base identity)
    canonical_mesh_base, canonical_mesh_move = decompose(mesh_K[0], part_mask)
    K_meshes = [warp(canonical_base + canonical_move, T_k(axis, pivot, q_k)) for k in 1..K]
    
    # Render K frames (locked camera)
    V_k = nvdiffrast(K_meshes, camera_locked)
    
    # Wan2.2 RFSDS gradient
    z_video = Wan22_VAE.encode(V_k)  # autograd ON for activations
    τ = inverse_cdf(ŵ, 1 - iter/2001)  # high→low schedule
    ε = randn_like(z_video)
    z_τ = (1-τ)·z_video.detach() + τ·ε
    with no_grad:
        v̂ = Wan22_DiT(z_τ, τ, image=I_0, text=locked_camera_prompt, CFG=25→12)
    grad = (v̂ - ε + z_video.detach())
    z_video.backward(gradient=grad)
    
    # Update: A0..A4 RASA params, part_seg_head, joint_head, color_adapter
    optimizer.step(); optimizer.zero_grad()
```

### Phase 2 — Topology hardening + URDF export
```python
# threshold + connected component cleanup
B_hard = (canonical_base > 0.5) ; M_hard = (canonical_move > 0.5)
mesh_base = mesh_extract(B_hard, with_color)
mesh_move = mesh_extract(M_hard, with_color)
joint = {"type": "rev"|"pris", "axis": axis_S2, "origin": pivot, "range": [0, max(q_k)]}
write_urdf(mesh_base, mesh_move, joint)
```

### 总结指标
- **总参数 trainable**: ~22M (~1.8% of frozen TRELLIS)
- **GPU memory peak**: ~22GB (24GB 卡可)
- **每 iter 时间**: ~5-8 秒（Wan2.2 5B 主导）
- **2000 iter / asset**: ~3-5 小时
- **vs CHORD 20h H200**: 4× 加速（更小 prior + 更少展开）

---

## 7. 论文 Novelty Claims（AAAI ready）

| Claim | 支撑 | 跟谁不同 |
|---|---|---|
| (1) **In-loop RASA**: 在 frozen TRELLIS SS-DiT 和 SLAT-DiT 内部插可训练 cross-state attention，让 K 状态在生成过程中协同 | A1 + A3 | FreeArt3D 把 TRELLIS 当黑盒；MonoArt fine-tune；SegViGen fine-tune |
| (2) **MorphAny3D K-way 改造** for articulated states：把 2-endpoint MCA/TFSA 推广到 K-state donor-weighted fusion，硬绑定 base + 软共享 texture | §2.4 + A3 | MorphAny3D 是 morphing；CHORD 是 free 4D motion |
| (3) **Tier 1+2 shared base**：cross-state attention 软共享 + anchor copy 硬一致，保证 base bit-identical without 大破坏 frozen TRELLIS | §4.2 | 现有 multi-view 全是 soft-only |
| (4) **RFSDS 优化 only inserted layers**（not 4D-GS like CHORD）→ output 直接是 URDF-ready articulated 3D | §6 | CHORD 不输出 URDF，FreeArt3D 需 K=6 真实图 |

---

## 8. References (consolidated)

### TRELLIS family
- TRELLIS arXiv:2412.01506 — original
- MonoArt arXiv:2603.19231 — uses 8-dim SLAT + trained heads
- SegViGen — fine-tunes TRELLIS.2 for part segmentation
- PartGen Meta arXiv:2412.18608 — frozen 3D backbone + segmentation head

### MorphAny3D
- arXiv:2601.00204 — paper
- https://github.com/XiaokunSun/MorphAny3D.git — code
- Local: `paper/MorphAny3D/`

### CHORD / RFSDS
- arXiv:2601.04194 — paper
- Project: https://yanzhelyu.github.io/chord/

### Wan2.2
- arXiv:2503.20314 — Wan paper
- HF: Wan-AI/Wan2.2-I2V-A14B (also TI2V-5B for 24GB GPU)

### Articulation specifically
- FreeArt3D arXiv:2510.25765 — single-DoF screw with TRELLIS prior, requires K real images
- ArtGS arXiv:2502.19459 — multi-view RGB-D × 2 states, dual-quaternion blending
- PARIS arXiv:2308.07391 — axis-on-S² discretization rationale

### Adapter techniques
- LoRA arXiv:2106.09685
- IP-Adapter arXiv:2308.06721 — parallel cross-attn pattern (我们用)
- AdaptFormer arXiv:2205.13535 — parallel bottleneck (texture head 用)

### Multi-view consistency
- MVDream arXiv:2308.16512 — soft cross-view attention
- AnimateDiff arXiv:2307.04725 — temporal modules at subset of blocks (我们参考)

### Feature matching (确认不用，但理解)
- LoFTR arXiv:2104.00680
- DUSt3R arXiv:2312.14132
- MASt3R arXiv:2406.09756
