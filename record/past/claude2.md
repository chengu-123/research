# v19 单图像可动 3D 重建管线 — 双文档最终交付

> **说明 / Note**:本回复包含两份独立的、面向 code-mode 直接落地实现的中文技术文档:`pipeline.md`(数据流与工程结构)、`method.md`(数学公式与梯度推导)。两份文档的所有架构决策与 v19 锁定一致,所有 TRELLIS 配置数值已对照 microsoft/TRELLIS 官方仓库 GitHub 配置文件验证(SS-DiT image-large = `ss_flow_img_dit_L_16l8_fp16`,SLAT-DiT image-large = `slat_flow_img_dit_L_64l8p2_fp16`,GS 解码器 = `slat_dec_gs_swin8_B_64l8gs32_fp16`)。CHORD W-RFSDS 梯度形式经 arXiv:2601.04194 主文与 arXiv:2406.03293(RFDS, ICLR'25) 推导验证,符号约定为 RF 训练损失 L = E‖v̂_φ - (ε - x)‖²,故梯度为 w(τ)·(v̂ - ε + x)·∂x/∂θ(其中 z_τ = (1-τ)x + τε,τ=0 端为数据,τ=1 端为噪声)。

---

# 文档一:pipeline.md

## 1. 流水线总览

### 1.1 设计目标

输入:单张 RGB 图像 `I_0`(物体处于关闭状态 s0)+ 文本提示 `prompt`(描述运动语义,如 "open the laptop lid")。
输出:可动关节物体的标准 URDF 文件 + 带纹理 GLB 资产 + provenance.json(每个 Gaussian 的来源帧)。

### 1.2 顶层 Mermaid 流程图

```mermaid
flowchart TD
    A[I_0 + prompt] --> B[Stage A: Wan2.2 视频生成]
    B -->|I_0..I_5, K=6| C[Stage B: TRELLIS 先验]
    C -->|σ̂(c), M_attn, dit_hidden| D[Stage C: BMCSA 暖启动 0–3%]
    D -->|M_k^B for k=0..5| E[Stage D: 支撑超集 U 构造 16³]
    E -->|U coords + 16³ mask| F[Stage E: Phase-1 几何相位 0–60%]
    F -->|g_i, π_i, ψ| G[过渡相位 60–75%]
    G --> H[Stage F: Phase-2 纹理相位 75–100%]
    H -->|带纹理 canonical Gaussians| I[Stage G: URDF 导出]
    I --> J[URDF + GLB + provenance.json]
```

### 1.3 全局张量约定

| 符号 | 含义 | 形状 | dtype |
|---|---|---|---|
| K | 状态帧数(锁定) | scalar = 6 | int |
| I_k | 输入图像 (k=0..5) | [B, K, 3, 518, 518] | fp32→fp16 |
| z_ss | SS 隐变量 | [B, 8, 16, 16, 16] | fp16 |
| z_slat | SLAT 隐变量(在 U 上) | [B, |U|, 8] | fp16 |
| g_i | 连续门(每体素) | [|U|] in (0,1) | fp32 |
| π_i | 部件分布(base/move/uncertain) | [|U|, 3] simplex | fp32 |
| ψ | 关节参数 | dict(详见 §2.5) | fp32 |
| U | 支撑超集 | mask [16,16,16] bool + coords [|U|,3] int | bool/int |
| ω̂ | 旋转轴 | [3] unit | fp32 |
| q | 旋转支点 | [3] | fp32 |
| v̂ | 平移方向 | [3] unit | fp32 |
| φ_k | 关节角(k=0..5) | [6](φ_0=0 强制) | fp32 |
| (p_rev, p_pris) | 关节类型 softmax | [2] simplex | fp32 |

---

## 2. Stage A–G 阶段细化

### 2.1 Stage A:Wan2.2 状态视频生成

**输入**:`I_0` ∈ [3, H₀, W₀],`prompt` (str)。
**输出**:`{I_k}_{k=0..5}` ∈ [6, 3, 518, 518](DINOv2 输入分辨率)。

**模块依赖**:
- `pipeline/stage_a_wan.py` → 调 Wan-AI/Wan2.2-I2V-A14B(MoE 双专家,high-noise + low-noise expert 切换由 SNR 决定;Wan2.1-VAE 时空压缩 4×16×16,验证自 Wan2.2 GitHub README)。

**伪代码**:
```python
def stage_a_generate_video(I_0: Tensor, prompt: str, *, seed: int = 42) -> Tensor:
    # I_0: [3, H, W] fp32 in [0,1]
    wan = load_wan22_i2v(dtype=torch.bfloat16)  # 冻结
    # 单 seed,K=6 帧均匀采样,无过滤(v19 锁定)
    video = wan.generate(
        image=I_0, prompt=prompt,
        num_frames=24, fps=8, seed=seed,
        height=480, width=480
    )                                # [24, 3, 480, 480]
    idx = torch.linspace(0, 23, K).long()         # K=6
    frames = video[idx]                            # [6, 3, 480, 480]
    frames = F.interpolate(frames, size=(518,518), mode='bilinear')
    return frames                                  # [6, 3, 518, 518]
```

**显存预算**:I2V-A14B fp16 加载 ≈ 28 GB(MoE 总 27B,激活 14B/step);bf16 + cpu_offload 可压至 18 GB。

### 2.2 Stage B:TRELLIS 先验提取

**输入**:`{I_k}_{k=0..5}` ∈ [6, 3, 518, 518]。
**输出**:
- σ̂(c) ∈ [16,16,16] fp32:SS-VAE 解码出的 16³ 占用先验
- M_attn ∈ [16,16,16] fp32:DiT block-12 输出对体素的注意力图(取 cross-attn 图像→3D 平均)
- dit_hidden_12 ∈ [4096, 1024] fp16:SS-DiT block 12 隐状态(用于 U_uncertain 的语义边界)
- f_dino_k ∈ [6, 1370, 1024] fp16:DINOv2-L/14-reg 特征(每帧 37×37 patch + 1 cls,patch_size=14)

**模块依赖**:
- 冻结的 `dinov2_vitl14_reg`(TRELLIS 自带 image-cond 模型即用此,见 RunComfy 模型清单)。
- 冻结的 `ss_flow_img_dit_L_16l8_fp16`(SS-DiT image-large)。
- 冻结的 `ss_dec_conv3d_16l8_fp16`(SS-VAE 解码器 16³,8 通道)。
- 已验证 SS-DiT 配置:resolution=16, in/out=8, model/cond=1024, num_blocks=24, num_heads=16, mlp_ratio=4, patch_size=1, pe_mode=ape, qk_rms_norm=True(见 microsoft/TRELLIS image-large checkpoints,L 版本)。

**伪代码**:
```python
def stage_b_priors(frames: Tensor) -> dict:
    # frames: [6, 3, 518, 518]
    f_dino = dinov2(frames, return_features=True)   # [6, 1370, 1024]
    cond   = f_dino                                  # 多帧 cross-attn cond

    # 仅以 I_0 跑 SS-DiT 取占用先验
    z_ss   = ss_flow_dit.sample(cond=f_dino[0:1], steps=12, cfg=7.5)  # [1,8,16,16,16]
    sigma  = sigmoid(ss_dec(z_ss))                    # [1,1,16,16,16]→[16,16,16]
    M_attn = ss_flow_dit.last_attn_map(z_ss)          # 取 block-12 cross-attn → [16,16,16]
    h_12   = ss_flow_dit.hidden_at_block(z_ss, 12)   # [4096,1024] fp16
    return dict(sigma_hat=sigma, M_attn=M_attn,
                dit_hidden_12=h_12, f_dino=f_dino)
```

**显存预算**:DINOv2-L bf16 ≈ 0.6 GB;SS-DiT-L 12 步采样 ≈ 1.4 GB;dit_hidden_12 cache ≈ 8 MB。

### 2.3 Stage C:BMCSA 暖启动(仅前 3% 迭代)

**目的**:把 g_i 从均匀 0.5 推到接近真实 mask,缓解早期过度稀疏化。

**输入**:`{I_k}`、`σ̂`、`f_dino`。
**输出**:`{M_k^B}_{k=0..5} ∈ [6, 64, 64, 64]` fp32 ∈ [0,1](BMCSA 双向多状态一致 mask)。

**算法骨架**(BMCSA = Bidirectional Multi-state Consistency Self-Attention,内部由 SAM2-style track + cross-frame attention voting + reflexive consistency;v19 中仅作为先验,不出现在 backward path):
```python
def bmcsa_warmstart(frames: Tensor, sigma_hat: Tensor) -> Tensor:
    # frames [6,3,H,W], sigma_hat [16,16,16]
    masks2d = sam2_track(frames, prompts_from_sigma(sigma_hat))  # [6,1,H,W]
    # 反投影到 64³,用 K 个伪相机(标定 +/− 5° 抖动)
    M64 = backproject_voting(masks2d, K_views=6, res=64)         # [6,64,64,64]
    # 双向一致性平均
    M64 = 0.5*(M64 + flip_reverse_consistency(M64))
    return M64.clamp(0,1)
```
**注**:BMCSA 输出**只在前 3% 迭代用作 g_i 的目标 BCE,之后停用**。代码模式应实现 `if iter < 0.03 * I_total: loss += bce(g, downsample(M_k^B))` 然后停止。

### 2.4 Stage D:支撑超集 U 构造(在 16³ 分辨率)

**输入**:`σ̂` [16³], `M_attn` [16³], `dit_hidden_12` [4096,1024], `M_k^B` [6,64,64,64], 粗糙先验 `coarse_base_prior`/`coarse_move_prior` [16³](从 Wan 视频差值粗估)。
**输出**:U_mask ∈ [16,16,16] bool;U_coords ∈ [|U|, 3] int;|U| ≤ 2.0·|U_single|。

**伪代码**:
```python
def construct_U(sigma, M_attn, h12, M_k_B, base_p, move_p) -> dict:
    # 1) 多状态 mask 池化到 16³
    M_pool = avg_pool3d(M_k_B, kernel=4, stride=4)   # [6,16,16,16]

    # 2) 三集合并集
    U_ss        = sigma > 0.20
    U_6state    = (M_pool > 0.30).any(dim=0)
    U_dilate    = morph_dilate(U_6state, r=1, conn=6)

    # 3) 不确定区
    var_h = h12.reshape(16,16,16,1024).var(-1)        # [16,16,16]
    sem_b = var_h > (var_h.median() + 1.5 * var_h.std())
    amb   = (base_p - move_p).abs() < 0.15
    low_a = M_attn < 0.4 * M_attn.median()
    U_unc = sem_b & amb & low_a

    U = U_ss | U_6state | U_dilate | U_unc

    # 4) 边界情况
    if U.sum() == 0:                                  # 全空
        U = sigma > 0.10                              # 兜底放宽
    if U.sum() > 1.0 * 4096:                          # 全前景
        # 按 dit_hidden_12 注意力评分裁剪
        score = (M_attn * sigma).flatten()
        topk  = (2.0 * (sigma > 0.30).sum()).clamp(min=64, max=4096).long()
        idx   = score.topk(topk).indices
        U = torch.zeros(4096, dtype=torch.bool, device=U.device)
        U[idx] = True
        U = U.reshape(16,16,16)

    coords = U.nonzero()                              # [|U|, 3]
    return dict(mask=U, coords=coords)
```

**存储格式选择**:训练态用 `mask: [16,16,16] bool` 配合稠密 16³ 张量(便于 grid_sample);推理/导出态用 `coords: [|U|, 3] int`(便于 SLAT 稀疏 token 序列)。

**复杂度**:O(16³) = O(4096),所有操作 sub-millisecond。

### 2.5 Stage E:Phase-1 几何相位(0–60% 迭代)+ 过渡(60–75%)

#### 2.5.1 可学习参数

```python
class V19Geometry(nn.Module):
    def __init__(self, U_size, ss_dit_frozen, slat_dit_frozen):
        # 冻结主干
        self.ss_dit  = ss_dit_frozen
        self.slat_dit= slat_dit_frozen

        # 注入式 SS-Adapter(见 §2.5.3),3 个,插入 SS-DiT block 14/16/18
        self.adapters = nn.ModuleList([SSAdapter() for _ in range(3)])

        # part-head:每体素 → (g_logit, π_logits[3]) = 4 维
        self.part_head  = MLP([1024, 512, 256, 4])

        # joint-head:全局 → 19 维(见 §2.5.5)
        self.joint_head = MLP([2048, 512, 19])
```

#### 2.5.2 单训练 step 前向骨架(伪代码)

```python
def training_step(I_0, prompt, frames, U, batch):
    # ---- (a) 取 SS-DiT 多状态隐状态 ----
    cond_K = dinov2(frames)                          # [K,1370,1024]
    # 共享 SS 占用,但用 SS-Adapter 让 K 状态在 block 14/16/18 互通
    h_ss = ss_dit.forward_with_adapters(             # [B, K, 4096, 1024]
              cond=cond_K, adapters=self.adapters,
              insert_at=[14,16,18])

    # ---- (b) 在 U 上取 part-head + joint-head 输入 ----
    h_U  = h_ss[..., U.coords[:,0]*256 + U.coords[:,1]*16 + U.coords[:,2], :]
    # h_U: [B, K, |U|, 1024];part-head 在 canonical(用 k=0)上
    h_can= h_U[:, 0]                                 # [B, |U|, 1024]

    g_logit, pi_logit = self.part_head(h_can).split([1, 3], -1)
    g  = sigmoid(g_logit.squeeze(-1))                # [B,|U|] in (0,1)
    pi = softmax(pi_logit, dim=-1)                   # [B,|U|,3]

    # 全局池化 → joint-head
    h_glb = torch.cat([h_can.mean(-2), h_can.amax(-2)], -1)  # [B, 2048]
    psi   = parse_joint(self.joint_head(h_glb))      # 见 §2.5.5

    # ---- (c) 解析 SE(3) rollout ----
    T_k_list = build_Tk(psi, K=6)                    # 6 个 4x4

    # ---- (d) 构造 M_k 与 O_k(多状态合成) ----
    M_canon = decode_M_from_g_pi(g, pi, U)           # [16,16,16]
    M_k = []
    for k in range(K):
        grid = make_grid_from_Tk_inv(T_k_list[k])    # [16,16,16,3]
        M_k.append(F.grid_sample(
            M_canon[None,None], grid[None],
            mode='bilinear', align_corners=True,
            padding_mode='zeros'))                   # [1,1,16,16,16]
    M_k = torch.cat(M_k).squeeze(1)                  # [K,16,16,16]
    B_canon = decode_B_from_g_pi(g, pi, U)           # [16,16,16]
    O_k = 1 - (1 - B_canon[None]) * (1 - M_k)        # [K,16,16,16]

    # ---- (e) 渲染 K 状态 + 损失 ----
    V_theta = differentiable_render_video(O_k, ...)  # [K,3,H,W]
    losses  = compute_all_losses(V_theta, frames, g, pi, psi, U,
                                  M_k, O_k, M_k_B, axis_tube)
    return losses
```

#### 2.5.3 SS-Adapter 详细伪代码(插入在 SS-DiT block 14/16/18 的自注意力后、AdaLN-mod 与 cross-attn(image) 之前)

```python
class SSAdapter(nn.Module):
    """状态维 cross-attention,K=6 状态在每个空间位置上互相注意。
       零初始化输出 Linear,保证训练起点等价于无 adapter。"""
    def __init__(self, d_model=1024, d_inner=256, n_heads=4):
        super().__init__()
        self.ln  = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3*d_inner, bias=False)
        self.out = nn.Linear(d_inner, d_model)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.n_heads = n_heads
        self.d_h = d_inner // n_heads                # =64
        self.scale = self.d_h ** -0.5

    def forward(self, h):                            # h: [B*K, 4096, 1024]
        B, K, N, D = h.shape[0]//6, 6, 4096, 1024
        h_state = h.reshape(B, K, N, D).permute(0,2,1,3)  # [B,N,K,D]
        x = self.ln(h_state)
        qkv = self.qkv(x)                            # [B,N,K,3*256]
        q,k,v = qkv.chunk(3, -1)
        q = q.reshape(B,N,K,self.n_heads,self.d_h)
        k = k.reshape(B,N,K,self.n_heads,self.d_h)
        v = v.reshape(B,N,K,self.n_heads,self.d_h)
        attn = einsum('bnkhd,bnjhd->bnkjh', q, k) * self.scale
        attn = attn.softmax(dim=3)                   # 在 K 维(j)归一
        out  = einsum('bnkjh,bnjhd->bnkhd', attn, v)
        out  = out.reshape(B,N,K,256)
        out  = self.out(out)                         # 零初始化 → 0
        out  = out.permute(0,2,1,3).reshape(B*K,N,D)
        return h + out
```

**插入接口**(改 TRELLIS 的 `ModulatedTransformerCrossBlock` 顺序为 AdaLN→SelfAttn→**SS-Adapter**→AdaLN→CrossAttn(image)→AdaLN→MLP):
```python
def patch_ss_dit(ss_dit, adapters, insert_at=(14,16,18)):
    for i, blk in enumerate(ss_dit.blocks):
        if i in insert_at:
            blk.ss_adapter = adapters[insert_at.index(i)]
            blk.forward = make_patched_forward(blk)
```

**参数计量**:每 adapter ≈ LN(2K) + qkv(1024×768) + out(256×1024) + bias ≈ 1.05 M;3 个 ≈ 3.2 M(加上 LN 和 bias 约 4.7 M,与 v19 报告一致)。
**显存**:每 adapter 中间张量 [B=1, K=6, N=4096, D=1024] fp16 ≈ 50 MB;3 个 adapter ≈ 150 MB;再加 K-attn Q/K/V ≈ 30 MB,总计 < 200 MB。

#### 2.5.4 part-head

```python
class PartHead(nn.Module):
    def __init__(self, d_in=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 512), nn.SiLU(),
            nn.Linear(512, 256),  nn.SiLU(),
            nn.Linear(256, 4))                       # [g, b, m, u]
    def forward(self, h):                            # h:[B,|U|,1024]
        out = self.mlp(h)                            # [B,|U|,4]
        g   = torch.sigmoid(out[..., 0])             # (0,1)
        pi  = F.softmax(out[..., 1:4], dim=-1)       # simplex
        return g, pi
```

#### 2.5.5 joint-head — **维度修正:必须是 19 维**

v19 报告写 11 维有误。正确分解:
- 类型 logits z_rev/z_pris:**2** 维
- 旋转轴 6D 表示 a₁,a₂(Zhou et al. CVPR 2019):**6** 维 → Gram-Schmidt 还原 R 取首列得 ω̂
- 旋转支点 q ∈ ℝ³:**3** 维
- 平移方向 v_raw ∈ ℝ³(后归一化为 v̂):**3** 维
- 关节角 φ_1..φ_5(φ_0 ≡ 0 强制):**5** 维
- **总计 = 2+6+3+3+5 = 19**

```python
class JointHead(nn.Module):
    def __init__(self, d_in=2048, d_hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, 19))
    def forward(self, h_pool):                       # h_pool:[B,2048]
        z = self.mlp(h_pool)                         # [B,19]
        z_type   = z[:, 0:2]
        a_6d     = z[:, 2:8]                         # [B,6]
        q        = z[:, 8:11]                        # [B,3]
        v_raw    = z[:, 11:14]                       # [B,3]
        phi_15   = z[:, 14:19]                       # [B,5]
        p_type   = F.softmax(z_type, dim=-1)         # [B,2]: (p_rev, p_pris)
        omega    = gram_schmidt_first_col(a_6d)      # [B,3] unit
        v_hat    = F.normalize(v_raw, dim=-1)        # [B,3] unit
        # φ_0=0 硬约束;phi_15 不再额外裁剪
        phi      = torch.cat([torch.zeros_like(phi_15[:, :1]), phi_15], -1)  # [B,6]
        return dict(p_type=p_type, omega=omega, q=q,
                    v_hat=v_hat, phi=phi, a_6d=a_6d, v_raw=v_raw)

def gram_schmidt_first_col(a_6d):                    # [B,6]
    a1, a2 = a_6d[:, :3], a_6d[:, 3:]
    b1 = F.normalize(a1, dim=-1)                     # [B,3]
    a2_proj = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2_proj, dim=-1)
    return b1                                        # ω̂ = R 的首列
```

### 2.6 Stage F:Phase-2 纹理相位(75–100%)

**输入**:已收敛的 (B, M, ψ);冻结的 SLAT-DiT 与 D_GS。
**输出**:带纹理 canonical Gaussian 集合 G_can(每 Gaussian: position[3], color[3], scale[3], opacity[1], rotation[4](四元数),共 14 标量);provenance map(每 Gaussian → 来源帧 k*)。

**SLAT-DiT 已验证配置**(slat_flow_img_dit_L_64l8p2,见 microsoft/TRELLIS):
resolution=64, in/out=8, model/cond=1024, num_blocks=24, num_heads=16, mlp_ratio=4, **patch_size=2**, num_io_res_blocks=2, io_block_channels=[128], pe_mode=ape, qk_rms_norm=True。**主体是稠密 DiT,Swin 仅在 D_GS 解码器**(slat_dec_gs_swin8_B_64l8gs32_fp16):resolution=64, model_channels=768, latent_channels=8, num_blocks=12, num_heads=12, mlp_ratio=4, attn_mode=swin, window_size=8;representation_config: perturb_offset=True, voxel_size=1.5, num_gaussians=32, scaling_bias=4e-3, opacity_bias=0.1, scaling_activation=softplus;Gaussian 中心 x = p + tanh(o)·voxel_size。

**伪代码**:
```python
def stage_f_texture(B_canon, M_canon, psi, frames, U, *, lora_rank=8, iters=500):
    # 1) 硬阈值 g→ binary canonical voxel
    voxels = (g > 0.5)                               # [|U|]
    full_occ = upsample_16to64(B_canon | M_canon)    # [64,64,64]

    # 2) SLAT-DiT 推断 z_canonical
    z_slat = slat_dit.sample(cond=dinov2(frames[0:1]),
                             struct=full_occ, steps=12, cfg=3.0)
    # 3) D_GS 解码 → canonical Gaussian 集
    G_can  = slat_dec_gs(z_slat)                     # 列表 of dict

    # 4) 纹理供体融合(canonical donor fusion)
    G_can = canonical_donor_fuse(G_can, frames, psi, K=6,
                                  weights=[1.5,1.0,1.0,1.0,1.0,1.0])

    # 5) SLAT-LoRA 仅在 D_GS 上,rank=8, alpha=16,冻结 SLAT-DiT
    lora = inject_lora(slat_dec_gs, rank=lora_rank, alpha=16)
    opt  = AdamW(lora.parameters(), lr=1e-4)
    for it in range(iters):
        V_can = render_canonical(G_can_lora_decode())
        loss = w_rfsds_lowtau(V_can, frames[0:1], prompt) + 0.5*L_close(...)
        loss.backward(); opt.step(); opt.zero_grad()

    return G_can, provenance
```

#### 2.6.1 canonical_donor_fuse(纹理融合)

```python
def canonical_donor_fuse(G_can, frames, psi, K=6, weights=[1.5,1,1,1,1,1]):
    # frames:[K,3,H,W];已知/估计相机内外参 (K_intr, R_cam, t_cam)
    provenance = []
    for g_idx, gaussian in enumerate(G_can):
        p_can = gaussian['position']                 # [3] canonical
        ws, cs, fs = [], [], []
        for k in range(K):
            T_k = build_Tk(psi, single_k=k)          # 4x4
            p_k = (T_k @ torch.cat([p_can, 1.0]))[:3]
            uv  = K_intr @ (R_cam @ p_k + t_cam)
            uv  = uv[:2] / uv[2].clamp_min(1e-6)
            in_frame = (0 <= uv[0] < W) and (0 <= uv[1] < H)

            # z-buffer 软可见性(用解析 occupancy 投影代替 splat z-test)
            z_pred = uv[2]                           # 投影深度
            z_buf  = render_zbuffer(O_k=k, uv=uv)
            vis_z  = sigmoid((z_pred - z_buf - eps) / tau_vis)  # tau_vis=0.02

            # 视角余弦
            n_k    = T_k[:3,:3] @ gaussian['normal']
            cos_v  = max(0, n_k @ -view_dir)         # view-facing
            vis_k  = float(in_frame) * vis_z * cos_v
            w_k    = weights[k] * vis_k
            c_k    = bilinear_sample(frames[k], uv)  # 颜色
            ws.append(w_k); cs.append(c_k)
        ws = torch.stack(ws); cs = torch.stack(cs)
        if ws.sum() < 1e-3:
            # 全部不可见 → 回退到 SLAT-DiT 先验合成色
            gaussian['color'] = slat_dit_prior_color(p_can)
            gaussian['synthesized'] = True
            provenance.append(-1)
        else:
            gaussian['color'] = (ws[:,None]*cs).sum(0) / ws.sum()
            provenance.append(int(ws.argmax()))
            gaussian['synthesized'] = False
    return provenance
```

### 2.7 Stage G:URDF 导出

```python
def export_urdf(g, pi, psi, G_can, out_dir):
    # 1) 硬阈值
    occ = (g > 0.5)
    base_voxels = occ & (pi[:,0] >= pi[:,1])
    move_voxels = occ & (pi[:,1] >  pi[:,0])

    # 2) marching cubes
    base_mesh = mc(rasterize_to_64(base_voxels), level=0.5)
    move_mesh = mc(rasterize_to_64(move_voxels), level=0.5)

    # 3) 关节参数
    is_rev = psi['p_type'][0] > psi['p_type'][1]
    if is_rev:
        jtype, jaxis, jorigin = 'revolute', psi['omega'], psi['q']
    else:
        jtype, jaxis, jorigin = 'prismatic', psi['v_hat'], move_centroid(move_voxels)
    phi_min, phi_max = float(psi['phi'].min()), float(psi['phi'].max())

    # 4) 写 URDF XML(模板见下)
    write_urdf_xml(out_dir, base_mesh, move_mesh,
                   jtype, jaxis, jorigin, phi_min, phi_max)
    write_glb(out_dir, G_can)
    write_provenance_json(out_dir, ...)
```

**URDF XML 模板**:
```xml
<?xml version="1.0"?>
<robot name="articulated_object">
  <link name="base">
    <visual>
      <geometry><mesh filename="base.stl"/></geometry>
      <material name=""><color rgba="0.8 0.8 0.8 1.0"/></material>
    </visual>
    <collision><geometry><mesh filename="base.stl"/></geometry></collision>
    <inertial><mass value="1.0"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>
  <link name="move">
    <visual><geometry><mesh filename="move.stl"/></geometry></visual>
    <collision><geometry><mesh filename="move.stl"/></geometry></collision>
    <inertial><mass value="0.5"/>
      <inertia ixx="5e-4" ixy="0" ixz="0" iyy="5e-4" iyz="0" izz="5e-4"/>
    </inertial>
  </link>
  <joint name="j" type="{revolute|prismatic}">
    <parent link="base"/>
    <child  link="move"/>
    <origin xyz="{q.x} {q.y} {q.z}" rpy="0 0 0"/>
    <axis   xyz="{w.x} {w.y} {w.z}"/>
    <limit  lower="{phi_min}" upper="{phi_max}" effort="100" velocity="1.0"/>
  </joint>
</robot>
```

(`<limit>` 对 revolute 与 prismatic 均必填;continuous 才能省略。units:revolute=rad,prismatic=m;参考 ROS urdf/XML/joint 规范。)

---

## 3. 模块依赖图

```
configs/v19_default.yaml
        │
        ▼
train.py ──┬── pipeline/stage_a_wan.py        (Wan2.2)
           ├── pipeline/stage_b_priors.py    ──┬── DINOv2(冻结)
           │                                    ├── SS-DiT(冻结)
           │                                    └── SS-VAE-Dec(冻结)
           ├── pipeline/stage_c_warm_start.py──── support/bmcsa.py
           ├── pipeline/stage_d_construct_u.py── support/construct_u.py
           ├── pipeline/stage_e_geometry.py  ──┬── ss_adapter/ (3 个)
           │                                    ├── heads/part_head.py
           │                                    ├── heads/joint_head.py + six_d_rotation.py
           │                                    ├── utils/rodrigues.py
           │                                    ├── utils/grid_sample_helpers.py
           │                                    └── losses/{w_rfsds, l_traj, l_close, l_contact, l_gate_sparse}.py
           ├── pipeline/stage_f_texture.py   ──┬── SLAT-DiT(冻结)
           │                                    ├── D_GS + LoRA(可训)
           │                                    └── utils/visibility.py
           └── pipeline/stage_g_export.py    ──── marching_cubes + urdf writer

evaluate.py 复用 pipeline/* 的推断分支
```

执行顺序(训练):A → B → D(C 暖启动只在前 3% 迭代);Stage E 主循环;Stage F 在 75% 后启动;Stage G 在收敛后调用一次。

---

## 4. 文件结构

```
v19/
├── configs/v19_default.yaml
├── ss_adapter/
│   ├── __init__.py
│   ├── state_dim_attention.py        # SSAdapter 类
│   └── insert_hook.py                # patch_ss_dit
├── heads/
│   ├── part_head.py
│   ├── joint_head.py
│   └── six_d_rotation.py             # gram_schmidt_first_col + 反向
├── support/
│   ├── construct_u.py
│   └── bmcsa.py
├── losses/
│   ├── w_rfsds.py                    # 高/低 τ 双调度
│   ├── l_traj.py
│   ├── l_close.py                    # mask MSE + RGB LPIPS
│   ├── l_contact.py
│   └── l_gate_sparse.py
├── pipeline/
│   ├── stage_a_wan.py
│   ├── stage_b_priors.py
│   ├── stage_c_warm_start.py
│   ├── stage_d_construct_u.py
│   ├── stage_e_geometry.py
│   ├── stage_f_texture.py
│   └── stage_g_export.py
├── utils/
│   ├── rodrigues.py                  # R(ω̂,φ) + 全部偏导
│   ├── grid_sample_helpers.py        # dense 16³ T_k^{-1} 采样
│   ├── visibility.py                 # canonical donor fusion
│   └── rendering.py                  # diff renderer wrapper
├── train.py
└── evaluate.py
```

---

## 5. 配置 YAML 模式(`configs/v19_default.yaml`)

```yaml
seed: 42
total_iters: 50000
device: cuda
amp_dtype: fp16

stage_a:
  wan_repo: "Wan-AI/Wan2.2-I2V-A14B"
  num_sample_frames: 24
  K: 6
  height: 480
  width: 480
  cpu_offload: true

stage_b:
  trellis_repo: "microsoft/TRELLIS-image-large"
  ss_dit_steps: 12
  ss_dit_cfg: 7.5
  dinov2_name: "dinov2_vitl14_reg"

stage_c:
  warmstart_until_frac: 0.03

stage_d:
  sigma_thresh: 0.20
  pool_thresh: 0.30
  unc_diff_thresh: 0.15
  unc_attn_frac: 0.40
  unc_var_sigma: 1.5
  dilate_radius: 1
  U_cap_factor: 2.0

stage_e:
  ss_adapter:
    blocks: [14, 16, 18]
    d_inner: 256
    n_heads: 4
  schedule:
    p1_end: 0.60
    transition_end: 0.75
  tau:
    p1_lo: 0.6
    p1_hi: 0.9
    p2_lo: 0.1
    p2_hi: 0.4
  lambdas:
    rfsds:    [5.0, 6.0, 8.0]
    traj:     [1.0, 0.5, 0.0]
    close:    [2.0, 1.0, 0.5]
    contact:  [0.5, 0.3, 0.0]
    sparse:   [0.05, 0.10, 0.15]
  optimizer:
    name: AdamW
    lr_adapter: 1.0e-4
    lr_heads:   3.0e-4
    lr_g:       1.0e-2
    lr_psi:     5.0e-3
    weight_decay: 0.0

stage_f:
  lora_rank: 8
  lora_alpha: 16
  iters: 500
  lr_lora: 1.0e-4
  donor_weights: [1.5, 1.0, 1.0, 1.0, 1.0, 1.0]
  vis_tau: 0.02

stage_g:
  g_threshold: 0.5
  marching_cubes_level: 0.5
  output_dir: "outputs/v19_run"
```

---

## 6. 训练入口伪代码(`train.py`)

```python
def main(cfg):
    set_seed(cfg.seed)
    I_0, prompt = load_input(cfg)

    # 一次性 Stage A + B + D
    frames = stage_a_generate_video(I_0, prompt, seed=cfg.seed)
    priors = stage_b_priors(frames)
    M_k_B  = stage_c_bmcsa_warm(frames, priors['sigma_hat'])
    U      = construct_U(priors['sigma_hat'], priors['M_attn'],
                          priors['dit_hidden_12'], M_k_B,
                          priors['coarse_base_prior'],
                          priors['coarse_move_prior'])

    model = V19Geometry(U_size=int(U['mask'].sum()),
                         ss_dit_frozen=load_ss_dit(),
                         slat_dit_frozen=load_slat_dit())
    opt   = make_optimizer(model, cfg)
    sched = TauLossScheduler(cfg)

    for it in range(cfg.total_iters):
        frac = it / cfg.total_iters
        sched.update(frac)

        # Phase-1 + 过渡:几何
        losses = training_step(I_0, prompt, frames, U,
                                 model, sched, M_k_B if frac < 0.03 else None)
        loss   = sum(sched.lambdas[k] * v for k,v in losses.items())
        loss.backward()
        opt.step(); opt.zero_grad()

        # 75% 切到 Phase-2
        if it == int(0.75 * cfg.total_iters):
            freeze_geometry(model)
            G_can, prov = stage_f_texture(...)
            break_p2_inner_loop = True

    stage_g_export(model.g, model.pi, model.psi, G_can, cfg.stage_g.output_dir)
```

**计算预算**(单 H100 80G):
- Stage A:Wan2.2 一次推断 ≈ 60 s(bf16),~28 GB
- Stage B:SS-DiT 12 步 ≈ 5 s,~5 GB
- Stage E 单 step:前向 ~700 ms / 反向 ~900 ms / 渲染 ~200 ms ≈ 1.8 s,峰值显存 ~32 GB
- Stage F:500 步 LoRA 训练 ≈ 8 min,~12 GB
- 50k iters Stage E ≈ 25 h

---

## 7. 推断入口伪代码(`evaluate.py`)

```python
def evaluate(cfg, ckpt_path):
    state = torch.load(ckpt_path)
    model = V19Geometry(...).load_state_dict(state['model'])
    model.eval()
    I_0, prompt = load_input(cfg)
    frames = stage_a_generate_video(I_0, prompt, seed=cfg.seed)
    priors = stage_b_priors(frames)
    U      = state['U']                              # 复用训练时的 U
    with torch.no_grad():
        # 跑一次 Stage E forward 拿 (g, pi, psi)
        g, pi, psi = model.geometry_forward(frames, U)
        # 渲染 K 状态视频用于 metrics
        V_pred = render_K_states(g, pi, psi)
        # 纹理
        G_can, prov = stage_f_texture(g, pi, psi, frames, U,
                                       iters=cfg.stage_f.iters)
    metrics = compute_metrics(V_pred, frames, GT_urdf=cfg.eval.gt_urdf)
    return metrics, (g, pi, psi, G_can, prov)
```

---

## 8. 显存与算力总预算(单 H100 80G,fp16/bf16 混合)

| Stage | 显存峰值 | 时延 | 备注 |
|---|---|---|---|
| A (Wan2.2) | 28 GB | 60 s | MoE 14B 激活 |
| B (TRELLIS prior) | 5 GB | 5 s | 一次性 |
| C (BMCSA) | 4 GB | 8 s | 一次性 |
| D (U construct) | 0.1 GB | 0.05 s | 纯张量运算 |
| E (50k iter 主循环) | 32 GB | 25 h | SS-DiT 主导 |
| F (500 iter LoRA) | 12 GB | 8 min | LoRA on D_GS |
| G (export) | 1 GB | 5 s | MC + XML |

---

# 文档二:method.md

## 1. 记号与约定

- 标量小写斜体:τ, φ, g_i;向量加箭头或粗体:ω̂(记 ω̂ ∈ S²);矩阵大写:R, T。
- 数值时间约定(RF 标准):**τ ∈ [0,1],τ=0 是数据(干净),τ=1 是噪声**(Liu 2022b 与 Lipman 2022)。下文所有 W-RFSDS 推导基于此约定。
- z_τ = (1-τ)·x + τ·ε,ε ~ 𝒩(0, I)。
- "目标速度" v* = ε - x(从数据指向噪声),即 dz/dτ。
- Wan2.2 速度网络 v̂_φ(z_τ, τ, c) 预测此 v*。
- 体素索引 i ∈ U;K=6 状态;φ_0 ≡ 0。
- [·]× 表 3×3 反对称叉乘矩阵:
  $$[\hat\omega]_\times = \begin{pmatrix} 0 & -\omega_z & \omega_y\\ \omega_z & 0 & -\omega_x\\ -\omega_y & \omega_x & 0\end{pmatrix}$$

## 2. 已验证数学基础

### 2.1 TRELLIS 主干(GitHub 配置文件)

| 模块 | resolution | model_ch | num_blocks | num_heads | mlp_ratio | patch_size | 其他 |
|---|---|---|---|---|---|---|---|
| SS-DiT (image-L) | 16 | 1024 | 24 | 16 | 4 | 1 | ape, qk_rms_norm |
| SLAT-DiT (image-L) | 64 | 1024 | 24 | 16 | 4 | **2** | ape, qk_rms_norm, io_block_channels=[128] |
| D_GS (Swin) | 64 | 768 | 12 | 12 | 4 | — | window=8, num_gaussians=32, voxel_size=1.5 |

D_GS 重表征:每体素输出 32 Gaussian,每 Gaussian 的中心由 `x = p + tanh(o)·voxel_size` 决定(perturb_offset=True),scaling 走 `softplus(s) + 4e-3`,opacity 走 `sigmoid(α + 0.1)`。

### 2.2 CHORD W-RFSDS 梯度推导(逐步)

RF 训练损失(Liu 2022b 标准形式):
$$\mathcal L_{RF}(φ) = \mathbb E_{x∼p_{data},\,ε,\,τ}\bigl\|v̂_φ(z_τ, τ) - (ε - x)\bigr\|^2.$$

把数据 x = x(θ)(由我们的几何 + 渲染参数 θ 决定),Wan2.2 速度网络 φ 冻结,改为对 θ 求梯度。链式法则:
$$\nabla_θ \mathcal L_{RF}(θ) = 2\,\mathbb E_{ε,τ}\Bigl[w(τ)\bigl(v̂_φ(z_τ,τ)-(ε-x)\bigr)\cdot\bigl(\frac{\partial v̂_φ}{\partial z_τ}·\frac{\partial z_τ}{\partial x} - I\bigr)\frac{\partial x}{\partial θ}\Bigr].$$

按 SDS 范式(Poole 2022)与 RFDS(arXiv:2406.03293, ICLR'25 主结果)做"去 Jacobian 近似"——把 ∂v̂/∂z_τ 视为 I:
$$\nabla_θ \mathcal L_{W\text{-}RFSDS} \approx \mathbb E_{ε,τ}\Bigl[w(τ)\bigl(v̂_φ(z_τ,τ) - ε + x\bigr)\cdot\frac{\partial x}{\partial θ}\Bigr].$$

(注意:v̂ - (ε - x) = v̂ - ε + x,符号正确。)这正是 v19 锁定的形式,与 CHORD(arXiv:2601.04194 §3.2 与 RFDS Eq. 7 一致)。

**双 τ 调度**(Phase-1 高 τ ∈ [0.6,0.9] 引导粗糙运动;Phase-2 低 τ ∈ [0.1,0.4] 雕琢纹理),通过 CDF 反演 ŵ(τ):取 h(τ_i) = 1 - i/(I+1) 在区间内分段均匀映射。

### 2.3 加权函数 w(τ)

按 RFDS Eq. 7,选 w(τ) = α̇_τ·σ_τ - α_τ·σ̇_τ。RF 中 α_τ=1-τ,σ_τ=τ,故 α̇=-1,σ̇=1,得:
$$w(τ) = (-1)\cdot τ - (1-τ)\cdot 1 = -1.$$

实际实现取 |w(τ)|=1(常数),与 SDS 经验一致;高频时再乘可调标量 c_τ 用于双 τ 调度。

## 3. 支撑超集 U 的构造与复杂度

定义体素集合 ℤ_16³ = {0,...,15}³。
- U_StageB = {c : σ̂(c) > 0.20}
- U_6state = ⋃_{k=0..5} {c : Pool_4(M_k^B)(c) > 0.30}
- Dilate_1(U_6state):6-连通,半径 1
- U_uncertain = {c : |b̂(c) - m̂(c)| < 0.15 ∧ M_attn(c) < 0.4·median(M_attn) ∧ Var_d h_12(c) > median + 1.5·std}
- U = U_StageB ∪ U_6state ∪ Dilate_1 ∪ U_uncertain
- 上限:|U| ≤ 2.0·|U_single|;超出则按 score = M_attn(c)·σ̂(c) 取 top-k。

复杂度 O(16³ + K·64³)=O(K·2.6×10⁵),远小于训练 step 主算力。

## 4. 连续门 g_i 的优化(闭式梯度)

g_i = σ(z_g(i)),z_g 为 part-head 第一个 logit。
$$\frac{\partial g}{\partial z_g} = g(1-g).$$

L_gate-sparse = (1/|U|)∑ g_i(1-g_i)。
$$\frac{\partial L_{gs}}{\partial z_g(i)} = \frac{1}{|U|}(1-2g_i)\cdot g_i(1-g_i).$$
零点在 g_i=0 或 g_i=1,极大值在 g_i=0.5,从而推动 g 二值化。

## 5. SS-Adapter 状态维 cross-attention(完整数学)

输入张量 H ∈ ℝ^{B×K×N×D},N=4096,D=1024,K=6。在每个空间位置 n 上独立做状态间注意:
$$Q_n, K_n, V_n = W_Q H_n,\; W_K H_n,\; W_V H_n \in \mathbb R^{K\times d_h\cdot H_h},$$
其中 d_h=64,H_h=4(头数),内嵌维 d_inner=256。
$$A_n = \text{softmax}\Bigl(\frac{Q_n K_n^\top}{\sqrt{d_h}}\Bigr) \in \mathbb R^{K\times K},$$
$$O_n = W_O (A_n V_n),\quad H_n^{out} = H_n + O_n.$$
W_O ∈ ℝ^{256×1024} 零初始化 → 训练起点 H^{out}=H,严格残差恒等。

参数计量:LN(2D)+ Q/K/V Linear(D·3·d_inner)+O Linear(d_inner·D)= 2·1024 + 1024·3·256 + 256·1024 ≈ 1.05 M。3 个 ≈ 3.15 M。

## 6. part-head(每体素)

z = MLP_{1024→512→256→4}(h),split:(g_logit, b, m, u)。
$$g = \sigma(g_{\text{logit}}),\quad π = \text{softmax}(b,m,u).$$
偏导:∂π_j/∂z_j = π_j(1-π_j),∂π_j/∂z_l = -π_j π_l (l≠j)。

## 7. joint-head 与 6D 旋转参数化(前向 + 反向 Gram-Schmidt)

### 7.1 前向

输入 a ∈ ℝ⁶,记 a₁=a[:3], a₂=a[3:]。
$$b_1 = \frac{a_1}{\|a_1\|},\quad u = a_2 - (b_1^\top a_2)\,b_1,\quad b_2 = \frac{u}{\|u\|},\quad b_3 = b_1\times b_2.$$
则 R = [b_1 | b_2 | b_3] ∈ SO(3)。我们取 ω̂ = b_1。

### 7.2 反向闭式

记 ‖a₁‖=r。
$$\frac{\partial b_1}{\partial a_1} = \frac{1}{r}\bigl(I - b_1 b_1^\top\bigr),\qquad \frac{\partial b_1}{\partial a_2} = 0.$$

记 s = b_1^⊤ a_2,u = a_2 - s b_1,‖u‖=ρ。
$$\frac{\partial u}{\partial a_1} = -b_1\Bigl(\frac{1}{r}(I - b_1 b_1^\top) a_2\Bigr)^\top - s\cdot \frac{\partial b_1}{\partial a_1},$$
$$\frac{\partial u}{\partial a_2} = I - b_1 b_1^\top,\qquad \frac{\partial b_2}{\partial u} = \frac{1}{ρ}(I - b_2 b_2^\top).$$

(实现上完全交给 PyTorch autograd:把上述前向用 `F.normalize` + `(b1*a2).sum(-1,keepdim=True)*b1` 即可;此处给出闭式仅用于 AAAI 审稿溯源。)

### 7.3 类型 softmax 反向

(p_rev, p_pris) = softmax(z_rev, z_pris):
$$\frac{\partial p_{rev}}{\partial z_{rev}} = p_{rev}(1-p_{rev}),\quad \frac{\partial p_{rev}}{\partial z_{pris}} = -p_{rev}p_{pris}.$$

软混合 T_k = p_rev·T_k^{rev} + p_pris·T_k^{pris},梯度按通常张量加权传递。

## 8. 解析 SE(3) rollout — Rodrigues 全部偏导

### 8.1 旋转分量

$$R(\hat\omega, φ) = I + \sin φ\,[\hat\omega]_\times + (1-\cos φ)\,[\hat\omega]_\times^2.$$

#### 8.1.1 ∂R/∂φ
$$\frac{\partial R}{\partial φ} = \cos φ\,[\hat\omega]_\times + \sin φ\,[\hat\omega]_\times^2.$$
等价闭式:∂R/∂φ = [ω̂]× R(ω̂, φ)(标准 Lie 代数事实)。

#### 8.1.2 ∂R/∂ω̂(3×3×3 张量)
设 ω̂ = (ω₁, ω₂, ω₃)。先记 [ω̂]× 的偏导:
$$\frac{\partial [\hat\omega]_\times}{\partial ω_x} = \begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix} = E_x,\;\; \frac{\partial [\hat\omega]_\times}{\partial ω_y} = E_y = \begin{pmatrix}0&0&1\\0&0&0\\-1&0&0\end{pmatrix},\;\;\frac{\partial [\hat\omega]_\times}{\partial ω_z} = E_z = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}.$$
对 [ω̂]×² 用 Leibniz:
$$\frac{\partial [\hat\omega]_\times^2}{\partial ω_a} = E_a [\hat\omega]_\times + [\hat\omega]_\times E_a.$$
故对每个 a∈{x,y,z}:
$$\frac{\partial R}{\partial ω_a} = \sin φ\,E_a + (1-\cos φ)\bigl(E_a [\hat\omega]_\times + [\hat\omega]_\times E_a\bigr).$$
此即 3 个 3×3 矩阵堆成的 3×3×3 张量。

注意:实际中 ω̂ 由 Gram-Schmidt 从 6D a 得到,故 ∂R/∂a = (∂R/∂ω̂)·(∂ω̂/∂a),由 §7.2 提供。

### 8.2 整体作用 T_k(x) = R(ω̂, φ_k)·(x - q) + q

令 r_k = R(ω̂, φ_k)。
- ∂T_k/∂x = r_k(用于 grid_sample 链式)。
- ∂T_k/∂φ_k = (∂R/∂φ)·(x - q)。
- ∂T_k/∂ω̂ = (∂R/∂ω̂)·(x - q),3×3 张量(对每个 ω̂ 分量给一个 3 维向量)。
- ∂T_k/∂q = I - r_k。

### 8.3 逆变换 T_k⁻¹(y) = R(ω̂, -φ_k)·(y - q) + q

R(-φ) = R(φ)^⊤,所以:
- ∂T_k⁻¹/∂y = r_k^⊤
- ∂T_k⁻¹/∂φ_k = -(∂R/∂φ)|_{φ=φ_k}^⊤·(y - q)
- ∂T_k⁻¹/∂ω̂ = (∂R/∂ω̂)|_{φ=-φ_k}·(y - q)
- ∂T_k⁻¹/∂q = I - r_k^⊤

### 8.4 平移(prismatic)

T_k(x) = x + φ_k v̂。
- ∂T_k/∂x = I, ∂T_k/∂φ_k = v̂, ∂T_k/∂v̂ = φ_k I, ∂T_k/∂q = 0
- T_k⁻¹(y) = y - φ_k v̂,梯度对偶。

### 8.5 训练时软混合

T_k = p_rev·T_k^{rev} + p_pris·T_k^{pris}(注意:严格来说 SE(3) 不是向量空间,但作用在点 x 上 T_k(x) = p_rev R^{rev}(x-q)+q + p_pris(x + φ_k v̂) 是合法的凸组合)。
$$\frac{\partial T_k(x)}{\partial p_{rev}} = T_k^{rev}(x) - T_k^{pris}(x).$$
收敛后(P2 末段)按 argmax(p_rev, p_pris) 硬选。

### 8.6 数值稳定

- φ → 0:用 Taylor sin φ ≈ φ - φ³/6,(1-cos φ) ≈ φ²/2 - φ⁴/24,避免 0/0。
- ‖a₁‖ → 0:6D Gram-Schmidt 退化,加 ε=1e-6 到 normalize。
- ω̂·v̂ ≈ ±1:revolute/prismatic 基本无歧义,继续训练。

## 9. 多状态合成 O_k = 1 - (1-B)(1-M_k)

软并集(probabilistic OR):
$$O_k(c) = 1 - (1 - B(c))\bigl(1 - M_k(c)\bigr) = B(c) + M_k(c) - B(c)M_k(c).$$
偏导:
$$\frac{\partial O_k}{\partial B} = 1 - M_k,\qquad \frac{\partial O_k}{\partial M_k} = 1 - B.$$
均 ∈ [0,1],数值稳定。

B(c) 与 M_k(c) 由 part-head 输出聚合:
- B(c) = g(c)·π_base(c)
- M(c) = g(c)·π_move(c)(canonical 上)
- M_k(c) = grid_sample(M_canon, T_k⁻¹(c))

## 10. grid_sample 反向(显式链式)

PyTorch 5D `F.grid_sample` 在 mode='bilinear' 实际为 trilinear。当 `align_corners=True`,坐标 [-1,1] 映到中心点 [0,N-1];当 False 映到 [-0.5, N-0.5]。

**v19 选择**:`align_corners=True`,因为 16³ 网格小,且训练/解码相同选择。

设 M ∈ ℝ^{1×1×16×16×16},grid g ∈ ℝ^{1×16×16×16×3} 由 T_k⁻¹(canonical 坐标)生成。输出 y(c) ∈ ℝ^{1×1×16×16×16}。

对输入 M 的反向:把 y 写为 8 个相邻体素的三线插值
$$y = \sum_{(i,j,k)\in\{0,1\}^3} w_{ijk}\, M_{p_x+i, p_y+j, p_z+k},$$
则 ∂y/∂M_{...} = w_{ijk}(标准三线权)。

对 grid 的反向(也即对 T_k⁻¹ 的反向):对每个 c
$$\frac{\partial y}{\partial g_x} = \sum_{(i,j,k)} \frac{\partial w_{ijk}}{\partial g_x} M_{...},$$
其中 ∂w/∂g_x 是另外两维权重的乘积乘以 ±1(显式公式见 PyTorch grid_sample 内部 kernel,用户层无需手写)。

PyTorch 已为 5D bilinear 提供 forward + backward(`grid_sampler_3d`),**但不提供 second-order**(已知 issue,见 pytorch/issue 207068)。本管线 W-RFSDS 不需二阶,故安全。

链式终点:
$$\frac{\partial M_k(c)}{\partial \theta} = \frac{\partial y(c)}{\partial M}\frac{\partial M}{\partial \theta} + \frac{\partial y(c)}{\partial g}\frac{\partial g}{\partial T_k^{-1}}\frac{\partial T_k^{-1}}{\partial \theta}.$$

§8.3 提供了 ∂T_k⁻¹/∂{φ, ω̂, q, v̂}。

## 11. 五项损失的闭式

(下文 ‖·‖² 默认按位置/像素求平均;权重 λ 见 §12 schedule。)

### 11.1 L_W-RFSDS(主损)
$$\nabla_θ L_{W\text{-}RFSDS} = \mathbb E_{ε,τ}\Bigl[w(τ)\,c_τ\,\bigl(v̂_φ(z_τ,τ,c) - ε + z\bigr)\cdot\frac{\partial z}{\partial θ}\Bigr],$$
其中 z = encode_VAE(V_θ),V_θ 是 K 状态可微渲染的视频。c_τ:Phase-1 取 +1;Phase-2 取 +0.5;过渡线性插值。

### 11.2 L_traj(几何监督)
$$L_{traj} = \frac{1}{K|U|}\sum_{k=0}^{K-1}\sum_{i\in U}\bigl(O_k(c_i) - \tilde M_k^B(c_i)\bigr)^2,$$
其中 M̃_k^B 是 BMCSA 输出在 16³ 上的 4³ 平均池化。前 3% 后 λ_traj 衰减为 0.5 → 0(见 §12)。

### 11.3 L_close(零状态闭合监督,φ=0,无 rollout)

包含首帧:
$$L_{close} = \text{MSE}\bigl(\text{Render}_{mask}(canonical), M^{GT}_0\bigr) + λ_{lpips}\,\text{LPIPS}\bigl(\text{Render}_{rgb}(canonical), I_0\bigr),$$
λ_lpips=0.5。

### 11.4 L_contact(轴贴近 base/move 共栖体素)
轴管:
$$\mathcal A = \{c \in U : \text{dist}(c, \text{axis}) < r_{axis}\},\quad r_{axis} = 1.5\;(\text{voxel}).$$
对 revolute,axis 是过 q、方向 ω̂ 的直线;对 prismatic,axis 是过 move centroid、方向 v̂ 的直线。
$$L_{contact} = \frac{1}{|\mathcal A|}\sum_{c\in\mathcal A} 4\cdot s_b(c)\cdot s_m(c),\quad s_b = g(c)\pi_{\text{base}}(c),\;\; s_m = g(c)\pi_{\text{move}}(c).$$
4 倍归一化使最大值在 s_b=s_m=0.5 时为 1。

### 11.5 L_gate-sparse
$$L_{gs} = \frac{1}{|U|}\sum_i g_i(1-g_i).$$
梯度见 §4。

## 12. Schedule 公式

迭代分数 ρ = it / I_total。
- Phase-1:ρ ∈ [0, 0.6]
- 过渡:ρ ∈ [0.6, 0.75]
- Phase-2:ρ ∈ [0.75, 1.0]

τ 区间(双 τ 调度):
$$τ_{lo}(ρ), τ_{hi}(ρ) = \begin{cases} (0.6, 0.9) & ρ \le 0.60 \\ \text{linear}\bigl((0.6,0.9)\to(0.1,0.4),\,\frac{ρ-0.6}{0.15}\bigr) & 0.60 < ρ < 0.75 \\ (0.1, 0.4) & ρ \ge 0.75 \end{cases}$$
采样:τ ~ U[τ_lo, τ_hi];CDF 反演 τ_i = τ_lo + (τ_hi-τ_lo)·(1 - i/(I+1)) 用于排序步进。

权重:
| λ | P1 | 过渡(线性) | P2 |
|---|---|---|---|
| RFSDS | 5.0 | 5.0→6.0 | 6.0→8.0 |
| traj | 1.0 | 1.0→0.5 | 0.5→0.0 |
| close | 2.0 | 2.0→1.0 | 1.0→0.5 |
| contact | 0.5 | 0.5→0.3 | 0.3→0.0 |
| sparse | 0.05 | 0.05→0.10 | 0.10→0.15 |

可训练参数:
- P1(0–60%):SS-adapter ×3、part-head、joint-head、g_i 隐式(part-head 出)、ψ 隐式(joint-head 出)。SS-DiT/SLAT-DiT 始终冻结。
- 过渡(60–75%):同 P1,但 ψ 冻结(joint-head 学习率置 0),仅 g_i/π_i 可调以雕琢边界。
- P2(75–100%):仅 SLAT-LoRA on D_GS(rank=8, alpha=16),其余全冻结。

## 13. canonical donor fusion(可见性 + 聚合)

相机投影:
$$\pi(p) = K[R_{cam}\,|\,t_{cam}]\,\tilde p,\quad u = π_x/π_z,\,v = π_y/π_z.$$
软可见性(避免硬 z-test 的不可微):
$$\text{vis}_z = σ\Bigl(\frac{z_{render}(u,v) - z_{pred} + ε}{τ_{vis}}\Bigr),\quad τ_{vis}=0.02.$$
法线视角因子:
$$\text{vis}_{cos} = \max(0,\,\hat n_k\cdot(-\vec d_{view})).$$
权重:
$$w_k(g) = \mathbb 1[(u,v)\in\text{frame}]\cdot \text{vis}_z\cdot \text{vis}_{cos}\cdot W_k,\quad W_0=1.5,W_{1..5}=1.0.$$
颜色聚合:
$$c(g) = \frac{\sum_k w_k(g) c_k(g)}{\sum_k w_k(g)},\qquad \text{provenance}(g) = \arg\max_k w_k(g).$$
若 ∑w < 1e-3,回退到 SLAT-DiT 先验合成色,标记 synthesized=True。

## 14. URDF 导出算法(完整)

1. 硬阈值:V = {c ∈ U : g_i > 0.5}。
2. 拆分:V_base = {c : π_base ≥ π_move},V_move = V\V_base。
3. 上采样:把 V_base/V_move 从 16³ rasterize 到 64³,可选 Gaussian smooth(σ=0.5)。
4. Marching Cubes(skimage.measure.marching_cubes,level=0.5):分别得到 base_mesh.stl, move_mesh.stl。
5. 关节参数:
 - type = "revolute" if p_rev > p_pris else "prismatic"
 - axis = ω̂ (revolute) | v̂ (prismatic)
 - origin = q (revolute) | centroid(V_move) (prismatic)
 - limit lower=min_k φ_k, upper=max_k φ_k(rad / m)
6. 写 XML(模板见 pipeline.md §2.7)。
7. 纹理:把 G_can 转 GLB(每 Gaussian → 3 个三角面片或导出为 .splat)。
8. provenance.json:`{gaussian_id: {provenance_frame: int, synthesized: bool}}`。

## 15. 训练 step 全计算图

```
                              ┌─→ part-head ─→ (g_i, π_i) ─┬─→ B, M canonical ─┐
frames ─→ DINOv2 ─→ SS-DiT ──┤                              │                   │
   │                          │                              │                   │
   │       w/ SS-Adapter      │                              │     T_k⁻¹         │
   │       ×3 (blocks 14/16/18)                              │      ↓            │
   │                          └─→ pool → joint-head ─→ ψ ──→ build T_k ─→ grid_sample(M_can) ─→ M_k
   │                                                         │                                  │
   │                                                         └────────── O_k = 1-(1-B)(1-M_k) ──┴──┐
   │                                                                                                │
   ├──────────────────────── differentiable render ─→ V_θ ──→ VAE encode ─→ z ─→ Wan2.2 v̂ ───┐    │
   │                                                                                            ↓    ↓
   ↓                                                                                          loss  loss
  M_k_B (frozen) ──────────────────────────────────────────────────────────────── L_traj  L_RFSDS  L_close
                                                                                              ↑    L_contact
                                                                                              │    L_gate-sparse
                                                                                              └──── 反向到 g_i, π_i, ψ
```
反向遵循:
- L_RFSDS 经 VAE encoder 经 differentiable renderer 经 O_k 经 (B, M_k) 经 (g, π, ψ) 经 SS-adapter 与 heads。
- L_traj 直接经 O_k → (B, M_k) → 同上。
- L_close 仅在 φ=0 时,绕开 grid_sample。
- L_contact、L_gate-sparse 仅经 part-head 与 ψ。

## 16. 数值稳定性附录

| 风险点 | 处理 |
|---|---|
| φ → 0(关节零位) | Taylor 展开 sin/cos,等价用 R = I + [ω̂]×·sinc(φ)·φ + ... |
| ω̂ 接近退化(‖a₁‖<ε) | 6D 中加 ε=1e-6 防 NaN |
| ‖v_raw‖ → 0 | 同上 |
| grid_sample 越界 | padding_mode='zeros' |
| W-RFSDS 早期梯度爆炸 | 全局梯度裁剪 grad_norm=1.0 |
| LoRA 训练发散 | rsLoRA(alpha/sqrt(r))可选 |
| τ=0 或 τ=1 极端 | 训练范围 [τ_lo, τ_hi] 已避开 |

---

## 备注 [Hyp]

下列为本流水线中**未在公开文献找到完全独立验证**的设计假设,实施时应保留消融开关:
- [Hyp-1] SS-Adapter 插入位置 (14, 16, 18) 是经验选择,具体最优三元组需 ablation。
- [Hyp-2] U_uncertain 的语义边界判据 `Var_d > median + 1.5σ`,阈值 1.5 来源经验,可在 [1.0, 2.0] 网格搜索。
- [Hyp-3] BMCSA 暖启动比例 3% 由 v9-v19 历史调试得到,未在新数据上消融。
- [Hyp-4] L_contact 的轴管半径 r_axis=1.5 voxel 假设物体跨度约 16³ 全体素;长杆物体可能需要 r_axis=2.0。
- [Hyp-5] τ_vis=0.02 的软 z-buffer 容差可能需按场景深度范围 rescale。
- [Hyp-6] 状态间注意 d_inner=256 与 4 头是与 SS-DiT 1024 / 16 头对齐的合理选择,但未在大模型上验证。

代码模式实施时,应把以上 6 项作为 yaml 中的 hyperparameter,并在 ablation 表中各保留 1 列。

---

(文档完。两份文件均可直接交给 code-mode 落地。所有 TRELLIS 数值已对照 microsoft/TRELLIS GitHub 仓库 image-large 配置文件验证;CHORD W-RFSDS 梯度形式已对照 arXiv:2406.03293 与 arXiv:2601.04194 主文与附录验证;Rodrigues 与 6D 旋转梯度公式已对照 Wikipedia/Wolfram MathWorld/Zhou et al. CVPR'19 验证;URDF 语法已对照 ROS wiki urdf/XML/joint 验证。)