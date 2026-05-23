# CAST-U A'B v2 — Method

> 单图 + Wan2.2 单视角伪视频 → canonical articulated 3D 资产（base mesh + move mesh + single-DoF joint + UV/atlas + URDF），TRELLIS 主干完全冻结。
>
> **核心创新一句话**：把 TRELLIS 的 `SparseStructureFlowModel` 当成一个 grad-enabled 的 *one-step structural refiner*（不当 sampler 用），让 W-RFSDS 视频蒸馏的梯度通过解析 SE(3) rollout 与可微 Gaussian 渲染，端到端地回到 SS-DiT 内部新增的 zero-init residual adapter，从而在不动 TRELLIS 主干的前提下学习 canonical articulated 几何、part segmentation 与 single-DoF joint 参数。
>
> **本版与 v1 的差异**：v1 有 16 处 TRELLIS 代码接口 bug（来自对源码的错误假设）和 5 个未拍板的设计点。v2 把所有接口与 TRELLIS / Wan2.2 实际源码对齐，并固化所有设计决策。所有声明均有 `file:line` 级别的源码核验。

---

## 0. 目录

1. 目标与边界
2. 与既有方案的对比和裁决
3. 核心设计哲学
4. 变量定义
5. 输入条件构造
6. 一次性 Bootstrap
7. 几何阶段：A'B 主体
8. W-RFSDS 梯度链与监督形式
9. 损失函数
10. 训练协议
11. 纹理阶段
12. 导出
13. 与 CHORD 的关系与差异
14. 终极一句话

---

## 1. 目标与边界

### 1.1 任务定义

**输入**：
- 一张物体闭合状态的 RGB 图像 `s_0_clean`
- 自然语言 prompt（描述铰接部件）

**输出**：
- `base.glb`：静止部分 textured mesh
- `move.glb`：可动部分 textured mesh
- `joint.json`：single-DoF 关节
- `atlas.png` + `texture_provenance.json`
- `object.urdf`

### 1.2 硬约束

- TRELLIS 主干（SS-VAE / SS-DiT / SLAT-DiT / D_GS / DINOv2）**全程冻结**
- Wan2.2 全程冻结
- 不使用外部分割器伪标签
- per-instance optimization
- single-DoF（revolute 或 prismatic）

### 1.3 帧数与分辨率

- **F = 21**（Wan2.2 `F % 4 == 1` 约束下的合法值，21 = 4·5+1，给出 6 个 latent frame）
- **分辨率 512 × 288**（保持 16:9，比 Wan 默认 832×464 节约 ~3.6× 显存）

---

## 2. 与既有方案的对比和裁决

| 方案 | 保留 | 放弃 |
|---|---|---|
| **v15 (canonical-first)** | canonical-first 哲学；contact-anchor joint；canonical UV donor fusion；W-RFSDS 高低噪分工 | (D1) argwhere 断点只停留在叙事；(D2) 跨 state hidden 物理对齐假设错；(D3) UV unwrap 是 argmax 断梯度 |
| **CAST-U++** | support superset U 思路；presence/move 双 sigmoid；analytic SE(3) rollout | (D4) outer-loop U refresh 不稳定；(D5) U_0 union 未明示尺度漂移；(D6) 纹理贡献不足 |
| **CAST-U v19.1** | BinaryConcrete + STE；累积 softplus 单调 φ；move-weighted joint pooling；joint type 软混合 | (D7) canonical lift K-state hidden 物理对齐假设错；(D8) STE 应用位置错（应在 D_GS opacity 端，非 SLAT 输入端）；(D9) U refresh 同 D4 |
| **GPT 综合 PDF** | 三方案职责分离框架；Stage A–J 表；活跃 loss ≤ 4 项；first_frame_rgb_anchor；A'B 主线（one-step grad-enabled refiner）；provenance map | (D10) carpet 被贬为脚注（实测它是 TRELLIS 尺度稳定性的硬前提）；(D11) 伪代码 7 处接口 bug（SLAT-DiT inner loop / α_g/α_m 双重计数 / α_m & deltas zero-init / ψ_param 被覆盖 / D_GS 重复调用 / SS_DiT subclass / opacity 改 `_opacity`）；(D12) U refresh 保留 |
| **CHORD (本方法的 W-RFSDS 直接参考)** | W-RFSDS 公式、τ schedule、Wan2.2 14B I2V 作为 teacher、render-then-VAE-encode、SDS chain rule（DiT stop-grad，VAE encode 保 grad）、canonical asset + analytic SE(3) deformation 范式 | (D-CHORD) CHORD 的 3D-GS 资产来自给定 mesh 转换（无 SS↔SLAT 梯度断），我们必须从 TRELLIS 单图 pipeline 造出来；CHORD 是 scene-level free-form deformation，我们是 single-DoF articulated rigid；CHORD 纯 SDS prior 无 reconstruction，我们加 Wan VAE latent + RGB reconstruction 作为补充监督 |

### 2.1 本方法的最小创新

1. **A'B 主线**：no-grad sampler bootstrap + grad-enabled one-step canonical SS-DiT structural refiner + learnable z_s_base + SS heads + analytic rollout + W-RFSDS
2. **D_GS 输出端 opacity gating**：BinaryConcrete + STE gate 应用在 `get_opacity` 后的 Gaussian opacity 上，避开 SLAT 输入 ResBlock 卷积模糊 binary 的问题
3. **canonical 单一 forward + analytic K-state rollout**：SS-DiT 只跑一次 canonical forward（K=1），21 帧由 6 个学习 φ 值插值 + 解析 SE(3) 派生
4. **W-RFSDS = SDS prior + Wan VAE latent reconstruction 混合**：CHORD 是纯 SDS prior，我们加入 wan_video_target 的 latent 重建项（按阶段权重 schedule，几何阶段重 SDS、纹理阶段重 recon）
5. **三阶段 detach staging**：z_s_base 与 ψ 在 0–5% / 5–15% / 15%+ 调度下统一管理两个反馈回路

### 2.2 carpet 的最终定位

**carpet 仅作为 Stage B Bootstrap 内的 TRELLIS 尺度稳定 scaffold，在 Stage D 之后完全不出现**：
- Stage A 用 `s_0_clean` 喂 Wan2.2 → 生成 21 帧 clean 视频
- Stage B 把 carpet 加到从 Wan 输出抽取的 6 帧上 → `s_k_carpet`（仅供 BMCSA 输入），跑完 BMCSA 后用 FreeArt3D plane fit 检测 carpet voxels，从 `U_object` 中**剥离**
- Stage D / F：渲染 `U_object`（无 carpet），W-RFSDS 监督 clean Wan 视频，`L_first` 锚 `s_0_clean`

论文表述："Carpet voxels are detected via FreeArt3D-style plane fitting after BMCSA bootstrap and excluded from the canonical support set before geometry optimization. They serve only as TRELLIS scale stabilizer during initialization and are not part of the proposed method."

---

## 3. 核心设计哲学

### 3.1 Canonical-first

最终交付物是单个 canonical articulated 3D 资产 + 关节参数。F=21 帧由 6 个学习 φ 值在时间维度插值，再通过解析 SE(3) rollout 派生：

$$O_t(x) = 1 - (1 - B(x))(1 - M(T_t^{-1}(x; \psi, \phi_t)))$$

其中 $B$ 是 canonical base occupancy，$M$ 是 canonical move occupancy，$T_t = T(\psi, \phi_t)$ 是从 6 个语义状态插值得到的 single-DoF SE(3) 变换。Base 一致性是 by-construction。

### 3.2 冻结主干，只学新增层与残差

可学参数限于：
- SS-DiT block 14/16/18 之后的 zero-init residual adapter
- 三个 head MLP（H_sup / H_part / H_joint）
- 显式 `nn.Parameter`：`Δz_s, α_g, α_m, ψ_param, delta_phi`

累计参数量约 5–20 MB / 对象。

### 3.3 W-RFSDS 通过解析路径回到 SS

视频差异 → 渲染差异 → Gaussian center / opacity 链 → analytic SE(3) → joint head + part head → SS-DiT hidden → adapter weights。整条链可微。TRELLIS 主干因 `requires_grad_(False)` 不参与权重更新，但激活仍参与 autograd。

### 3.4 离散决策放在 D_GS 输出端

`BinaryConcreteSTE` 把 logit 转 hard 0/1 用作 Gaussian opacity 的乘子。这一位置：
- `diff_gaussian_rasterization` 对任意 `opacity ∈ [0,1]` 完全可微
- 不经过 SLAT 的 `SparseResBlock3d` 卷积
- forward 硬决策直接匹配导出时的二值化

### 3.5 outer-loop 操作最小化

`U_object` 在 bootstrap 后**固定不变**，不做 refresh。所有几何 / 部件 / 关节优化都在可微 inner-loop 内完成。

---

## 4. 变量定义

### 4.1 冻结资产（不参与优化）

| 符号 | 来源 / file:line | 形状 |
|---|---|---|
| `SS-DiT` | `models/sparse_structure_flow.py::SparseStructureFlowModel` | `resolution=16, patch_size=1, model_channels=1024, num_blocks=24, num_heads=16` |
| `SS-VAE-decoder` | `models/sparse_structure_vae.py::SparseStructureDecoder` | 16³→64³ dense conv decoder，输出 `[B,1,64³]` logit |
| `SLAT-DiT` | `models/structured_latent_flow.py::SLatFlowModel` | sparse transformer |
| `D_GS` | `models/structured_latent_vae/decoder_gs.py::SLatGaussianDecoder` | Swin-based, **32 Gaussians/voxel**，`forward(x: SparseTensor) → List[Gaussian]` |
| `DINOv2` | DINOv2-L | 1024-d patch tokens |
| `Wan22 VAE` | `Wan-AI/Wan2.2-I2V-A14B/vae2_1.py::WanVAE_` | 时间下采 4×（`vae_stride = (4, 8, 8)`） |
| `Wan22 DiT` | `Wan-AI/Wan2.2-I2V-A14B/model.py::WanModel` | MoE 27B / 14B 激活 |

### 4.2 一次性 Bootstrap 资产（no_grad 计算，存盘）

| 符号 | 形状 | 含义 |
|---|---|---|
| `z_s0` | `[1, 8, 16, 16, 16]` | BMCSA 收敛后 SS latent（K 个状态合并） |
| **`z_slat0`** | `[N_obj, 8]` post-norm | SLAT bootstrap latent，**已经过 `* std + mean` 后归一化**（decoder 期望的输入分布） |
| `dit_hidden_cache` | `[1, 4096, 1024] × 3` | block 14/16/18 之后 hidden（仅诊断用） |
| `O_init` | `[1, 1, 64, 64, 64]` | `sigmoid(SS-VAE-decoder(z_s0))` |
| `M_attn_boot` | `[N_obj]` | BMCSA cross-state token cosine（move prior） |
| `is_carpet_mask` | `[N_full]` bool | FreeArt3D plane fitting 结果 |
| `U_object` | `[N_obj, 3]` int32 | object 候选 voxel 集（已去掉 carpet） |
| `ψ_0` | dict | StageC joint 初值 |
| `φ_0` | `[6]` | 6 个状态进度（`φ_0[0]=0`） |
| `anchors_object` | `[N_a, 3]` | 接触带 anchor |
| `z_wan_target` | `[1, C_lat, 6, 36, 64]` | Wan VAE encoder 编码后的 21-帧 clean Wan 视频 latent（avoid 每 iter 重算） |

### 4.3 可学习参数

**P1 几何**：

| 符号 | 形状 | 初始化 | 含义 |
|---|---|---|---|
| `Δz_s` | `[1, 8, 16, 16, 16]` | zeros | z_s_base 残差 |
| `α_g` | `[N_obj]` | **zeros**（residual to `occ_logits`） | object 存在 logit 残差 |
| `α_m` | `[N_obj]` | `logit(M_attn_boot.clamp(0.05, 0.95))` | object 可动 logit |
| `ψ_param` | `[19]` | `encode_joint(ψ_0)` | joint 显式参数 |
| `delta_phi` | `[5]` | `inverse_softplus(φ_0[1:]-φ_0[:-1])` | 严格正单调增量 |
| `adapter_{14,16,18}` | MLP | output proj zero-init | SS-DiT residual adapter |
| `H_sup, H_part` | MLP | output proj zero-init | support/move residual head |
| `H_joint` | MLP | output proj zero-init | joint residual head |

**P2 纹理**额外可学（**几何全部冻结**）：

| 符号 | 形状 | 含义 |
|---|---|---|
| `Δ_features_dc` | `[N_gauss, 1, 3]` | per-Gaussian SH₀ DC 颜色残差，zero-init |
| `donor_weights` | `[N_texel, K=6]` | donor 融合权重 |

**P2 不再优化 `Δz_s, α_g, α_m, ψ_param, delta_phi, adapter, H_*`。**

### 4.4 关键初始化的依据

- `α_g = 0` (residual): 避免与 `occ_logits` 双重计数。初始 `r_i = occ_logits_i + 0 + 0 = logit(O_init_i)`
- `α_m = logit(M_attn_boot.clamp(0.05, 0.95))`: 防止初始 `m_i = 0.5` 导致每个 voxel 半 base 半 move 鬼影
- `delta_phi = inverse_softplus(φ_0 增量)`: 让初始 `φ = φ_0`，不让 `softplus(0)=0.693` 制造意外大初值
- `ψ_param = encode_joint(ψ_0)`: `ψ_pred = project_joint(ψ_param + λ_joint · Δψ)`，head zero-init 时 `ψ_pred ≡ project_joint(ψ_param)`

---

## 5. 输入条件构造

### 5.1 Stage A 与 Stage B 的数据分离

| 用途 | 内容 | carpet | 来源 |
|---|---|---|---|
| **W-RFSDS supervision** | `s_0_clean ... s_20_clean`（21 帧 clean Wan 视频） | 无 | Wan2.2(`s_0_clean`, prompt, F=21) |
| **L_first anchor** | `s_0_clean` | 无 | 用户输入 |
| **BMCSA bootstrap** | 6 个抽样状态 `s_k_carpet` (k=0..5) | 有 | 从 21 帧 clean Wan 输出抽 6 帧 `+ add_carpet` |

### 5.2 BMCSA 的 latent-space SCAR mix（**不是图像编码混合**）

旧 StageB 的 "mixed input" 实际是**采样过程中的 K-parallel sampler 之间的 latent mix**，不是 "image → SS-VAE latent → mix"（TRELLIS 没有 image → SS-VAE 路径）。

正确机制：

```
K=6 parallel SS samplers，每个 state 一条 noise track:
    z_t^(0), z_t^(1), ..., z_t^(5)

DINOv2 cond per state:
    cond_k = DINOv2(s_k_carpet) for k=0..5

Sampler 步骤:
    Pass 1 (steps 0..7):
        for each step t:
            z_t = SS_DiT_step(z_t, t, cond_k for each k)
            # K-parallel batch=6
            
            # SCAR symmetric mix:
            for k in 1..4:
                z_t^(k) ← 0.3·z_t^(0) + 0.4·z_t^(k) + 0.3·z_t^(5)
            # k=0 和 k=5 不参与 mix
    
    Pass 2 (steps 8..24):
        启用所有 24 个 block 的 BMCSA (K/V 跨 state 平均)
        不再 SCAR mix
        可选 SDEdit from t*=0.5
```

权重 `(0.3, 0.4, 0.3)` 来自旧 StageB 实测稳定配比。

### 5.3 Canonical image cond（Stage D 用）

```
cond_can = DINOv2(s_0_carpet)
```

**Stage D one-step SS-DiT forward 只用 s_0_carpet 的 DINO cond**（K_DiT=1）。21 个渲染帧的运动信息**完全靠 analytic SE(3) rollout 注入**，不让 SS-DiT hidden 携带 state-specific bias。

---

## 6. 一次性 Bootstrap（Stage B）

完整 `torch.no_grad()` 跑一次，输出 §4.2 所有 bootstrap 资产。

### 6.1 步骤

```
1. Wan2.2(s_0_clean, prompt, F=21, 512×288) → 21 帧 clean 视频
2. 从 21 帧抽 6 状态（均匀 [0, 4, 8, 12, 16, 20]）→ s_0_clean..s_5_clean
3. 给抽样的 6 帧加 carpet → s_0_carpet..s_5_carpet
4. K=6 parallel SS sampler:
   - DINOv2 cond per state: cond_k = DINOv2(s_k_carpet)
   - Pass 1 steps 0..7: SCAR symmetric latent mix
   - Pass 2 steps 8..24: BMCSA on all 24 blocks
   - forward_hook block 14/16/18 → dit_hidden_cache
5. z_s0 = mean(z_final, dim=K_batch)
6. O_init = sigmoid(SS-VAE-decoder(z_s0))
7. M_attn_boot = compute_cross_state_token_cosine(z_final)
8. FreeArt3D plane fit on O_init's bottom z-slice → is_carpet_mask
9. 构造 U_object:
   raw = (O_init * (~is_carpet_mask) > 0.3) ∪ corridor ∪ anchor_band ∪ uncertain_shell
   U_object = dilate(raw, 1 voxel)
10. SLAT sampler on U_object coords (no_grad):
    z_slat_raw = slat_sampler.sample(z_s0, U_object_with_batch_col).samples
    z_slat0 = z_slat_raw * slat_std + slat_mean   # ★ post-norm cache
11. StageC joint init: partition → BIC type voting → swept volume carve → axis refine
    → ψ_0, φ_0, anchors_object
12. Wan VAE 编码 clean 21-帧视频（缓存，避免每 iter 重算）:
    z_wan_target = Wan22_VAE_encoder(s_0_clean..s_20_clean).detach()
13. 全部 detach 写盘
```

### 6.2 SparseTensor coords 格式（关键）

TRELLIS 的 `SparseTensor` 要求 coords 是 `[N, 4]`：第 0 列是 batch index。所有构造 SparseTensor 的地方都要加 batch 列：

```python
U_object_xyz = bootstrap.U_object               # [N_obj, 3]
batch_col = torch.zeros(N_obj, 1, dtype=torch.int32)
U_object_with_batch = torch.cat([batch_col, U_object_xyz], dim=-1)   # [N_obj, 4]

sparse_in = SparseTensor(feats=z_slat0, coords=U_object_with_batch)
```

证据：`pipelines/trellis_image_to_3d.py:191` 用 `argwhere(...)[:,[0,2,3,4]]` 构造 4 列 coords；`modules/sparse/basic.py:107-127` 用 `coords[:,0]` 作 batch index。

---

## 7. 几何阶段：A'B 主体（Stage D）

### 7.1 RF q_sample（TRELLIS 公式）

```python
σ_min = 1e-5    # 来自 trainers/flow_matching/flow_matching.py:62
ε = torch.randn_like(z_s_base)
z_t = (1 - t) * z_s_base + (σ_min + (1 - σ_min) * t) * ε
```

**不要写 `z_t = (1-t)·z + t·ε`**（σ_min=0 简化版）。

### 7.2 One-step SS-DiT structural refiner（composition wrapper，**不 subclass**）

```python
class SS_DiT_WithAdapters(nn.Module):
    """
    Composition wrapper, NOT subclassing SparseStructureFlowModel.
    Avoids the unsafe `super().__init__(base.config)` pattern (config不存在).
    """
    def __init__(self, base_ss_dit, adapters: dict):
        super().__init__()
        self.base = base_ss_dit                    # 持有引用，不复制权重
        self.adapters = nn.ModuleDict(adapters)    # {'14': ..., '16': ..., '18': ...}
        for p in self.base.parameters():
            p.requires_grad_(False)
    
    def forward_capture(self, x, t_raw, cond):
        """
        x:     [1, 8, 16, 16, 16]
        t_raw: scalar in [0, 1]
        cond:  [1, N_dino, 1024]
        Returns: pred_v, captured_hidden_at_14_16_18
        """
        t_model = torch.tensor([1000.0 * t_raw], device=x.device)   # ★ TRELLIS × 1000 约定
        
        # 手动复刻 base.forward 的 block loop（无法走 self.base(x, t, cond) 因为要 capture）
        h = self.base.input_layer(x.view(1, 8, -1).permute(0, 2, 1))   # patchify, [1, 4096, 1024]
        h = h + self.base.pos_emb[None]
        t_emb = self.base.t_embedder(t_model)
        if self.base.share_mod:
            t_emb = self.base.adaLN_modulation(t_emb)
        h = h.type(self.base.dtype)
        t_emb = t_emb.type(self.base.dtype)
        cond = cond.type(self.base.dtype)
        
        captured = {}
        for k, block in enumerate(self.base.blocks):
            h = block(h, t_emb, cond)
            key = str(k)
            if key in self.adapters:
                h = h + self.adapters[key](h)     # post-block residual
                captured[k] = h                   # ★ post-adapter hidden
        
        h = h.type(x.dtype)
        pred_v = self.base.out_layer(h).permute(0, 2, 1).view(1, 8, 16, 16, 16)
        return pred_v, captured
```

**关键**：
- `t_raw * 1000` 是 TRELLIS 内部约定（`flow_euler.py:38-42`）
- adapter 在 block 输出之后注入（post-block residual）
- head 读 post-adapter hidden，否则 adapter 对 head 无贡献
- adapter 输出 proj zero-init 让 step 0 时 `adapter(h) ≡ 0`，退化为 vanilla SS-DiT

### 7.3 Hidden → U 坐标采样（trilinear + PE + occ_logits）

SS-DiT hidden grid = 16³（`patch_size=1` 验证自 `ss_flow_img_dit_L_16l8_fp16.json`）。`U_object` 在 64³。需要显式坐标映射：

```python
def sample_hidden_at_U(hidden_14, hidden_16, hidden_18, U_object_xyz, occ_logits):
    """
    hidden_*: [1, 4096, 1024]
    U_object_xyz: [N, 3] in [0, 64)
    occ_logits: [1, 1, 64, 64, 64]
    Returns: feat per voxel [N, feat_dim]
    """
    grid_res = round(hidden_14.shape[1] ** (1/3))    # = 16; never hardcode
    
    # Normalize U coords to grid_sample's [-1, 1]
    coord_norm = (U_object_xyz.float() / 63.0) * 2.0 - 1.0    # [N, 3]
    
    sampled = []
    for h in [hidden_14, hidden_16, hidden_18]:
        h_grid = h.view(1, 1024, grid_res, grid_res, grid_res)
        f = F.grid_sample(
            h_grid,
            coord_norm.view(1, -1, 1, 1, 3),
            mode='bilinear',            # 5D bilinear = trilinear
            align_corners=True,
            padding_mode='border',
        ).squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)    # [N, 1024]
        sampled.append(f)
    
    # Fourier positional encoding
    pe = fourier_pe(U_object_xyz.float() / 63.0, num_freqs=6)    # [N, 36]
    
    # Local occ logit per voxel (already on 64³ grid → direct index)
    occ_per_voxel = occ_logits.view(-1)[U_object_flat_idx].unsqueeze(-1)    # [N, 1]
    
    return torch.cat([*sampled, pe, occ_per_voxel], dim=-1)
    # 总维度: 3·1024 + 36 + 1 = 3109
```

**为什么 trilinear 单独不够 + 需要 PE**：16³ token 网格的一个 token 服务 64 个 64³ voxel（4×4×4 sub-cube）。trilinear 只在 token 边界提供 smooth interp，sub-cube 内部 voxel 之间区别度低。Fourier PE 给每个 voxel 一个 fine-grained 位置签名，让 head MLP 能在 sub-cube 内做 part 分割。

### 7.4 几何底座 + residual heads（α_g 零初始化避免双重计数）

```python
z_s_base = bootstrap.z_s0 + Δz_s             # [1, 8, 16, 16, 16]
occ_logits = ss_vae_decoder(z_s_base)        # [1, 1, 64, 64, 64], 可微 to Δz_s

feat = sample_hidden_at_U(
    hidden_14, hidden_16, hidden_18,
    bootstrap.U_object_xyz,
    occ_logits,
)    # [N_obj, 3109]

# r_i (presence logit) — α_g zero-init residual, occ_logits provides the prior
r = occ_logits.view(-1)[U_object_flat_idx] + α_g + λ_sup * H_sup(feat).squeeze(-1)
# r_i = logit(O_init_i) + 0 + 0 在 step 0 时 — 单次 prior

# b_i (move logit) — α_m initialized from M_attn_boot
b = α_m + λ_part * H_part(feat).squeeze(-1)
# b_i = logit(M_attn_boot_i) + 0 在 step 0 时
```

H_sup, H_part 的 output projection zero-init，让 step 0 时 `λ_sup·H_sup ≡ 0`，logits 纯由 BMCSA prior 决定。

### 7.5 BinaryConcrete + STE gate

```python
def BinaryConcreteSTE(logit, T):
    u = torch.rand_like(logit)
    g_soft = sigmoid((logit + torch.log(u) - torch.log(1 - u)) / T)
    g_hard = (g_soft > 0.5).float()
    return g_hard - g_soft.detach() + g_soft     # forward hard, backward soft

g_obj = BinaryConcreteSTE(r, T_g)        # [N_obj]
m_obj = BinaryConcreteSTE(b, T_m)        # [N_obj]
```

`T_g, T_m`：warmup 1.5，main phase 退到 0.4，transition 退到 0.2，texture 固定 0.15。

### 7.6 Joint residual（ψ_param 不被 H_joint 覆盖）

```python
# move-weighted attention pool
weights = m_obj / (m_obj.sum() + 1e-6)              # [N_obj]
F_pool = (weights.unsqueeze(-1) * feat).sum(dim=0)  # [3109]

Δψ = H_joint(F_pool)                                # [19], zero-init output

# Staged detach for joint feedback loop
ψ_for_warp = stage_detach(ψ_param, phase, mode="joint")

# ★ Residual around ψ_param, not overwrite
ψ_pred = project_joint(ψ_for_warp + λ_joint * Δψ)
```

`project_joint` 强制结构约束：
- axis L2 归一化
- prismatic direction L2 归一化
- pivot 落在 bbox 内
- joint type via `type_soft = sigmoid(ψ.type_logit)` 软标量（训练阶段两路 render soft blend，推理阶段 hard 切分）

### 7.7 累积 softplus φ（长度 6）

```python
phi_inc = F.softplus(delta_phi)                                   # [5], 严格正
phi = torch.cat([
    torch.zeros(1, device=phi_inc.device),
    torch.cumsum(phi_inc, dim=0)
])                                                                 # [6]
# phi[0] = 0; phi[k] = sum_{j=1..k} phi_inc[j-1]; 严格单调增
```

### 7.8 Analytic SE(3) rollout（21 帧插值 + 两路 render 软混合）

6 个语义 φ 通过线性插值映到 21 个渲染时刻：

```python
phi_render_indices = torch.linspace(0, 5, 21)              # [21] float in [0, 5]
# Linear interpolation through 6 control points
phi_render = interp_1d(phi, phi_render_indices)            # [21]
# phi_render[0] = 0, phi_render[20] = phi[5]

# Generate two-branch SE(3) per render frame
T_revolute = [SE3_revolute(ψ_pred.axis, ψ_pred.origin, phi_render[t]) for t in range(21)]
T_prismatic = [SE3_prismatic(ψ_pred.axis, phi_render[t]) for t in range(21)]
```

**Joint type soft blend 在 RENDER 端，不在 SE(3) 矩阵端**：

```python
type_soft = sigmoid(ψ_pred.type_logit)

rgb_revolute = render_with_warp(gauss_can, T_revolute, g_full, m_full)    # [21, 3, H, W]
rgb_prismatic = render_with_warp(gauss_can, T_prismatic, g_full, m_full)

rgb_frames = (1 - type_soft) * rgb_revolute + type_soft * rgb_prismatic   # [21, 3, H, W]
```

理由：在 SE(3) 矩阵层面线性混合（`T = (1-p)·T_rev + p·T_pri`）数学上不是 rigid transform。在 render 输出 RGB 上软混合是合法的（两个独立的 rigid transform 各自渲染，输出像素层面 blend）。

代价：每 iter 渲染量翻倍 = 42 次 rasterize（21 帧 × 2 分支）。在 H800 上仍可接受。

### 7.9 Canonical Gaussian 解码 + warp + opacity gating（D_GS 输出端）

```python
# ★ K loop 外，每 iter 一次
sparse_in = SparseTensor(
    feats = bootstrap.z_slat0,                 # post-norm
    coords = bootstrap.U_object_with_batch_col # [N_obj, 4]
)
gauss_can = d_gs(sparse_in)[0]                 # ★ 取 [0]，因为返回 List[Gaussian]

# 提取 activated opacity（★ get_opacity，不是 _opacity）
opacity_canon = gauss_can.get_opacity          # sigmoid(_opacity + opacity_bias), [N_gauss, 1]
# N_gauss = N_obj * 32 (D_GS 每 voxel 32 Gauss)

xyz_canon = gauss_can.get_xyz                  # [N_gauss, 3]
rot_canon = gauss_can.get_rotation             # [N_gauss, 4] quaternion
scale_canon = gauss_can.get_scaling            # [N_gauss, 3]
sh_canon = gauss_can._features_dc              # [N_gauss, 1, 3] SH₀ DC

# Per-voxel gate → per-Gaussian gate (each voxel has 32 Gauss)
parent_idx = bootstrap.gaussian_parent_voxel_idx   # [N_gauss], maps gauss_idx → U_object_idx
g_per_gauss = g_obj[parent_idx]                    # [N_gauss]
m_per_gauss = m_obj[parent_idx]                    # [N_gauss]
```

**渲染单个 branch 的 21 帧（base + warped move 双贡献）**：

```python
def render_with_warp(gauss, T_list, g_per_gauss, m_per_gauss):
    rgbs = []
    for t in range(21):
        T_t = T_list[t]   # SE(3) matrix [4, 4]
        
        # Base 贡献：位置不变，opacity 乘 g·(1-m)
        means_base    = xyz_canon
        opacity_base  = opacity_canon.squeeze(-1) * g_per_gauss * (1 - m_per_gauss)
        rotation_base = rot_canon
        
        # Move 贡献：位置 warp，opacity 乘 g·m
        means_move    = (T_t[:3, :3] @ xyz_canon.T).T + T_t[:3, 3]
        opacity_move  = opacity_canon.squeeze(-1) * g_per_gauss * m_per_gauss
        rotation_move = quat_mul(R_to_quat(T_t[:3, :3]), rot_canon)
        
        # Concat to 2*N_gauss
        means_all    = torch.cat([means_base, means_move], dim=0)
        opacity_all  = torch.cat([opacity_base, opacity_move], dim=0).unsqueeze(-1)
        rotation_all = torch.cat([rotation_base, rotation_move], dim=0)
        scale_all    = torch.cat([scale_canon, scale_canon], dim=0)
        sh_all       = torch.cat([sh_canon, sh_canon], dim=0)
        
        rgb_t = diff_gaussian_rasterize(
            means3D    = means_all,
            opacities  = opacity_all,
            rotations  = rotation_all,
            scales     = scale_all,
            shs        = sh_all,
            raster_settings = settings_locked_camera,
        )
        rgbs.append(rgb_t)
    return torch.stack(rgbs)
```

**关键设计**：
- 一套 canonical Gaussians，派生 base 与 move 两份 opacity-gated 贡献
- base 贡献 means 不变；move 贡献 means 被 `T_t` warp
- BinaryConcrete STE 的 `g_per_gauss, m_per_gauss` 直接乘 `get_opacity`（post-sigmoid），**不改 `_opacity`**（pre-sigmoid logit）
- 每帧 rasterize 输入 2×N_gauss = 64×N_obj Gaussian
- 不 clone Gaussian 对象（Gaussian 类没有 `.clone()` 方法），直接在 rasterizer 输入端构造 arrays

---

## 8. W-RFSDS 梯度链与监督形式

### 8.1 监督混合：SDS prior + Wan VAE latent reconstruction

不像 CHORD 的纯 SDS prior，我们加入 latent reconstruction，让 wan_video_target 真正进入 loss。

```python
def W_RFSDS_prior(rgb_frames, image_cond, text_cond, τ):
    """CHORD-style SDS prior. Pure score distillation, no target alignment."""
    z_θ = wan22_vae_encoder(rgb_frames)              # ★ grad-enabled
    
    with torch.no_grad():
        ε = torch.randn_like(z_θ)
        z_τ = (1 - τ) * z_θ.detach() + τ * ε
        v_pred = wan22_dit(z_τ, τ, text_cond, image_cond)
    
    # CHORD Eq. 3: residual = v_pred - ε + z
    residual = v_pred - ε + z_θ.detach()
    
    return (residual.detach() * z_θ).sum() / z_θ.numel()


def L_latent_rec(rgb_frames, z_wan_target_cached):
    """Wan VAE latent reconstruction against cached target."""
    z_render = wan22_vae_encoder(rgb_frames)         # grad-enabled, recomputed each iter
    return ((z_render - z_wan_target_cached.detach()) ** 2).mean()


def L_rgb_rec(rgb_frames, wan_video_target):
    """Direct RGB + LPIPS supervision."""
    return l1_loss(rgb_frames, wan_video_target) + lpips_loss(rgb_frames, wan_video_target)
```

### 8.2 完整梯度路径

```
loss = λ_sds · L_sds + λ_lat · L_latent_rec + λ_rgb · L_rgb_rec
       + λ_first · L_first(rgb_frames[0], s_0_clean)
       + λ_contact · L_contact_anchor(ψ_pred, anchors_object)
       + λ_gate · L_gate_entropy(soft_r, soft_b)
       + λ_z · (Δz_s**2).mean()

loss.backward()

梯度路径:
  ┌─ rasterizer.backward
  │     ↓
  │   Gaussian {means, opacity, rotation, sh}
  │     ├── opacity 链 → g_per_gauss, m_per_gauss
  │     │     ↓ BinaryConcreteSTE backward (soft sigmoid grad)
  │     │   r, b
  │     │     ↓
  │     │   α_g, α_m  ✓
  │     │   H_sup(feat), H_part(feat)  ✓
  │     │     ↓
  │     │   feat = trilinear(hidden) + PE + occ_logits
  │     │     ↓
  │     │   hidden_14/16/18  ─→ adapter_14/16/18  ✓
  │     │   occ_logits → SS_VAE_decoder(z_s_base) → z_s_base → Δz_s  ✓
  │     │
  │     ├── means_move 链 → T_k → ψ_pred → ψ_param + λ_joint · H_joint(F_pool)  ✓
  │     │                  → phi_render → phi → cumsum(softplus(delta_phi))  ✓
  │     │
  │     └── sh_all (P2 only) → features_dc + Δ_features_dc  ✓ (P2)
  │
  ├─ wan22_vae_encoder.backward
  │     ↓
  │   rgb_frames (grad-enabled flow back to Gaussians)
  │
  └─ Wan22 DiT: no_grad (teacher only, no权重 update)
```

### 8.3 τ 调度（结合 Wan2.2 MoE 切换）

Wan2.2 I2V-A14B 是 SNR-switched MoE：高 τ 用 high-noise expert（偏整体 layout），低 τ 用 low-noise expert（偏细节）。我们利用这一点分阶段：

```
Geometry phase (G0 + G1): τ_sds ∈ [0.6, 0.9]    高噪 → high-noise expert → geometry/motion 监督
Transition:               τ_sds ∈ [0.4, 0.6]    中噪
Texture phase (T):        τ_sds ∈ [0.1, 0.4]    低噪 → low-noise expert → 细节监督
```

τ 在区间内每 iter 采样。

---

## 9. 损失函数

### 9.1 P1 几何阶段总损失

$$
\mathcal{L}_\text{geom} = \lambda_\text{sds} \cdot \mathcal{L}_\text{SDS} + \lambda_\text{lat} \cdot \mathcal{L}_\text{latent-rec} + \lambda_\text{rgb} \cdot \mathcal{L}_\text{rgb-rec} + \lambda_\text{first} \cdot \mathcal{L}_\text{first} + \lambda_\text{contact} \cdot \mathcal{L}_\text{contact} + \lambda_\text{gate} \cdot \mathcal{L}_\text{gate} + \lambda_z \cdot \mathcal{L}_z
$$

**活跃项 ≤ 4 项原则**：geometry phase 主要项是 `L_SDS, L_first, L_contact, L_gate` 四项；`L_latent_rec` 弱权重；`L_rgb_rec` 几乎为 0；`L_z` 是 anchor 不算"活跃"。

### 9.2 各项

- **L_SDS** = `W_RFSDS_prior(rgb_frames, cond_can, text_cond, τ_high_mid)` — geometry phase 主信号
- **L_latent_rec** = `‖wan_vae(rgb_frames) - z_wan_target_cached‖²` / numel
- **L_rgb_rec** = `l1(rgb_frames, wan_video_target) + lpips(rgb_frames, wan_video_target)`
- **L_first** = `l1(rgb_frames[0], s_0_clean) + lpips(rgb_frames[0], s_0_clean)`
- **L_contact**：
  - revolute: `softmin_a dist(line(ω, q), a)² over a ∈ anchors_object`
  - prismatic: `1 - cos(v̂, corridor_direction)`
  - 只在 U_object 上算（不涉及 carpet）
- **L_gate** = `mean(σ(r)·(1-σ(r)) + σ(b)·(1-σ(b)))` — **用 soft 值**，因为 hard 值 g(1-g) ≡ 0
- **L_z** = `(Δz_s ** 2).mean()` — 防止 Δz_s 漂离 z_s0

### 9.3 P2 纹理阶段总损失

$$
\mathcal{L}_\text{tex} = \lambda_\text{sds-low} \cdot \mathcal{L}_\text{SDS}^\text{low-τ} + \lambda_\text{lat-tex} \cdot \mathcal{L}_\text{latent-rec} + \lambda_\text{rgb-tex} \cdot \mathcal{L}_\text{rgb-rec} + \lambda_\text{first-tex} \cdot \mathcal{L}_\text{first} + \lambda_\text{color} \cdot \mathcal{L}_\text{color-smooth}
$$

P2 阶段 `L_latent_rec, L_rgb_rec` 权重大幅提高（geometry 已锁，wan_video 的纹理细节安全可用）；`L_SDS` 在 low τ。

---

## 10. 训练协议

### 10.1 阶段表

| 阶段 | iter 比例 | `t_ss` (q_sample) | `τ_sds` (W-RFSDS) | detach (z_s & ψ) | `λ_sup, λ_part, λ_joint` | `T_g, T_m` | `λ_sds, λ_lat, λ_rgb` |
|---|---|---|---|---|---|---|---|
| Warmup G0 | 0–10% | 固定 0.30 | 固定 0.85 | full detach | 0, 0, 0 | 1.5 | 1.0, 0.0, 0.0 |
| Main G1 | 10–60% | `U(0.25, 0.55)` | `U(0.6, 0.9)` | 0–5% detach, 5–15% EMA, 15%+ full | 0→0.3, 0→0.3, 0→0.5 | 1.5→0.4 | 1.0, 0.1, 0.0 |
| Transition | 60–75% | `U(0.20, 0.40)` | `U(0.4, 0.6)` | full grad | 0.3, 0.3, 0.5 | 0.4→0.2 | 0.5, 0.5, 0.1 |
| P2 Texture | 75–100% | — (Δz_s frozen) | `U(0.1, 0.4)` | n/a (几何冻结) | (held) | 0.15 | 0.2, 1.0, 1.0 |

### 10.2 Staged detach

```python
def stage_detach(tensor, global_iter, total_iters, mode, ema_buf=None):
    """
    使用全局进度 global_iter / total_iters, NOT per-phase iter.
    avoids re-triggering detach in transition / texture phases.
    """
    f = global_iter / total_iters
    if f < 0.05:
        return tensor.detach()
    elif f < 0.15:
        if mode == "joint":
            ema_buf.mul_(0.95).add_(0.05 * tensor.detach())
            return ema_buf.clone()
        else:
            ρ = (f - 0.05) / 0.10
            return tensor.detach() + ρ * (tensor - tensor.detach())
    else:
        return tensor
```

### 10.3 ε 噪声共享

每个 iter 采样一次 `ε ~ N(0, I)`，21 个渲染帧的 q_sample 共享同一噪声（仅一次 SS-DiT forward，不是 21 次）。Wan VAE 的内部 noise 由 Wan 自己 sample（无关）。

### 10.4 优化器（参数组分别学习率）

| 参数组 | lr |
|---|---:|
| `Δz_s` | 1e-4 |
| `α_g, α_m` | 5e-3 |
| `ψ_param` | 5e-3 |
| `delta_phi` | 1e-2 |
| `adapter_{14,16,18}` | 1e-4 |
| `H_sup, H_part, H_joint` | 5e-4 |

P2:
| `Δ_features_dc` | 1e-3 |
| `donor_weights` | 5e-3 |

全部 AdamW（β=(0.9, 0.999)，weight_decay=0.01）。

---

## 11. 纹理阶段（P2）

### 11.1 几何全部冻结

```python
# 从 P1 末尾固化所有几何状态
gauss_can_p1 = d_gs(SparseTensor(z_slat0, U_object_with_batch))[0]
xyz_frozen      = gauss_can_p1.get_xyz.detach()
scale_frozen    = gauss_can_p1.get_scaling.detach()
rotation_frozen = gauss_can_p1.get_rotation.detach()
opacity_canon_frozen = gauss_can_p1.get_opacity.detach()
features_dc_base    = gauss_can_p1._features_dc.detach()

g_frozen = g_obj_p1.detach()    # P1 末尾 hard gate
m_frozen = m_obj_p1.detach()
ψ_pred_frozen = ψ_pred_p1.detach()
phi_frozen = phi_p1.detach()
```

### 11.2 P2 唯一可学：per-Gaussian color residual

```python
Δ_features_dc = nn.Parameter(torch.zeros_like(features_dc_base))
```

每 iter：

```python
features_dc_optimized = features_dc_base + Δ_features_dc   # ★ 必须真进 render

# Render with frozen geometry + optimized color
for t in range(21):
    T_t = SE3_rollout(ψ_pred_frozen, phi_render[t])
    rgb_t = render_with_warp(
        means=xyz_frozen, opacity=opacity_canon_frozen,
        rotation=rotation_frozen, scale=scale_frozen,
        sh=features_dc_optimized,    # ★ learnable color
        ...
    )

L_tex.backward()    # 梯度只到 Δ_features_dc, 不到几何变量
```

### 11.3 Donor 收集（可选）

为每个 canonical surface 点收集 K=6 状态下的可见性、视角、深度一致性、纹理质量，加权融合 donor color 到 canonical atlas。`donor_weights` 是可学的 `[N_texel, K]`，用于 L_donor_consistency。

### 11.4 P2 不动 Δz_slat / D_GS LoRA

理由：D_GS 输出包含 `xyz, scale, rotation, opacity, features_dc`。修改 Δz_slat 或 D_GS LoRA 会同时改 geometry channels，破坏 P1 已收敛的 part/joint。`Δ_features_dc` 只动 color channel，几何 100% 冻结。

---

## 12. 导出

### 12.1 硬阈值

```python
r_final = occ_logits[U_object] + α_g + λ_sup·H_sup_final
b_final = α_m + λ_part·H_part_final

g_hard = (sigmoid(r_final) > 0.5)
m_hard = (sigmoid(b_final) > 0.5)

base_voxels = U_object[g_hard & ~m_hard]
move_voxels = U_object[g_hard &  m_hard]
# carpet 不参与 export（U_object 已剥离 carpet）
```

### 12.2 Mesh + atlas

```python
# subset feats: 必须按 voxel 索引对齐
base_idx = torch.searchsorted(U_object, base_voxels)
move_idx = torch.searchsorted(U_object, move_voxels)

z_slat_final = z_slat0 + 0    # P1 阶段 Δz_slat 不存在 (P2 也冻结 z_slat)

sparse_base = SparseTensor(
    feats = z_slat_final[base_idx],
    coords = add_batch_col(base_voxels)
)
sparse_move = SparseTensor(
    feats = z_slat_final[move_idx],
    coords = add_batch_col(move_voxels)
)

# 推荐：先 decode 整个 mesh，按 voxel 归属切分 triangle（避免边界 artifact）
mesh_full = d_mesh(SparseTensor(z_slat_final, add_batch_col(U_object)))[0]
base_tri_mask = assign_triangle_to_voxel_group(mesh_full, base_voxels)
move_tri_mask = assign_triangle_to_voxel_group(mesh_full, move_voxels)
mesh_base = extract_submesh(mesh_full, base_tri_mask)
mesh_move = extract_submesh(mesh_full, move_tri_mask)

# UV unwrap + atlas bake
base_atlas, move_atlas = uv_unwrap_and_bake(
    mesh_base, mesh_move,
    A_fused=donor_atlas, provenance=provenance_map
)
```

### 12.3 Joint + URDF

```python
ψ_hard = harden_joint(ψ_pred_final)   # type argmax, axis normalize

joint = {
    "type":        ψ_hard.type,                  # "revolute" or "prismatic"
    "origin":      ψ_hard.origin.tolist(),
    "axis":        ψ_hard.axis.tolist(),
    "limit_lower": 0.0,
    "limit_upper": float(phi[5]),
    "states":      phi.tolist(),
}

object.urdf = build_urdf(base.glb, move.glb, joint)
```

---

## 13. 与 CHORD 的关系与差异

| 维度 | CHORD | 本方法 (A'B v2) |
|---|---|---|
| 任务 | scene-level free-form 4D deformation | single-DoF articulated rigid 重建 |
| 3D 资产来源 | 给定 mesh → 转 3D-GS（已是 canonical） | 单图 → TRELLIS SS+SLAT bootstrap（需绕 argwhere 断点） |
| Teacher | Wan2.2 14B I2V | 同 |
| 渲染 → 监督 | render 3D-GS → gsplat → Wan VAE encode (grad) → Wan DiT (no_grad) → W-RFSDS | 同 |
| 帧数 | F=41 | F=21 |
| 分辨率 | 832×464 | 512×288 |
| 运动表示 | 控制点 + LBS + Fenwick tree per-frame SE(3) | 全局 single-DoF SE(3) + 6 个 φ + 21 帧插值 |
| W-RFSDS 监督形式 | 纯 SDS prior (无 wan_video_target) | SDS prior + Wan VAE latent reconstruction + RGB rec 混合 |
| τ 采样 | 确定性 annealing（按 CDF） | 区间随机采样 + 阶段切换 |
| SS↔SLAT 梯度断 | 不存在（3D-GS 端到端可微） | **本方法的核心创新**：通过 D_GS opacity gating + dense SS-VAE decoder logit 桥 + adapter at SS-DiT block 14/16/18，把 W-RFSDS 梯度引回 SS 内部 |
| Part / joint 分割 | 不做 | BinaryConcrete STE gate + analytic single-DoF |

**本方法的真正创新点不在 W-RFSDS（CHORD 已经做好），而在于"如何在冻结 TRELLIS 主干前提下，让 W-RFSDS 梯度回到 SS 内部新增 adapter"**。

---

## 14. 终极一句话

**用 grad-enabled 的 one-step SS-DiT structural refiner 把 W-RFSDS 视频蒸馏的梯度（SDS prior + Wan VAE latent reconstruction 混合）从渲染像素端，经可微 Gaussian 渲染（base + warped move 双贡献）、解析 SE(3) rollout（6 个学习 φ 插值到 21 帧）、D_GS 输出端的 BinaryConcrete opacity gate（不改 `_opacity`，乘 `get_opacity`）、SS-VAE decoder 的 dense occupancy logit 桥与 trilinear+PE hidden 坐标映射，端到端地回到 SS-DiT block 14/16/18 之后新增的 zero-init residual adapter 和三个 head MLP；在冻结 TRELLIS 主干的前提下，在固定的 canonical support U_object 上学习 base/move 分割、single-DoF joint 参数；carpet 仅作 Stage B 内的 scale-stabilizing scaffold，不进入 Stage D 之后的优化主线；纹理阶段冻结所有几何变量，只学 per-Gaussian SH₀ DC color residual 与 donor fusion。**
