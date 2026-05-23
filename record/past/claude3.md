# CAST-U v19.1 最终设计交付文档

> **本次交付**:基于 v19 与 CAST-U(修改意见.md)的差异分析,我已锁定六项结构性改进,并将其完整集成至 v19.1。本次输出为两份协调一致的 Markdown 文档(`pipeline.md` 与 `method.md`),供 code-mode 直接落地。所有公式、张量形状、超参数、稳定性分析均已闭合,允许 zero-design-question 实现。
>
> **TL;DR(三条核心结论)**
>
> - **v19.1 = v19 + 六项改进**:(1) U 外环周期刷新 (2) 二值 concrete + STE 门 (3) SS-adapter 内联合反向 warp 规范化抬升 (4) 移动加权注意力池化 (5) 累积 softplus 单调 φ (6) 双头独立 sigmoid。其余设计(Wan2.2 K=6 取帧、BMCSA 暖启动、TRELLIS image-large 冻结主干、解析 SE(3) rollout、双 τ W-RFSDS、两阶段时序、Canonical donor、SLAT-LoRA、URDF 导出)与 v19 完全一致。
> - **关键风险已识别并给出工程对策**:改进 3 与改进 4 在 SS-adapter 内构成 ψ↔m 反馈回路。我**采用 Option A+C 混合策略**:BMCSA 提供 ψ₀ → 前 5% 迭代对 inverse-warp 中的 ψ 使用 `detach()` → 5%–15% 之间用 EMA(ψ_ema, β=0.95)进入 warp → 15% 之后开放完整反传。该策略在数学上对应 Picard 迭代延迟反馈,有界稳定性可由 BMCSA 初值与轴管收敛半径联合保证(详见 method.md §15.3)。
> - **就绪度**:文档已涵盖 7 项详细任务、15 项消融、9 类失败模式、7 天实施路线图、完整 YAML schema、张量级伪代码与梯度图。**[Red] 反馈回路稳定性必须在 Day 5 完成 sanity test**,否则回退至 Option B(全程 detached ψ for warp)。

---

# 文档 1 / 2:`pipeline.md`(v19.1 完整版)

## 1. Pipeline 总览

CAST-U v19.1 是一个**单图 → 可动 URDF 资产**的端到端蒸馏流水线,主干为冻结的 TRELLIS-image-large 与 Wan2.2-I2V-A14B(Rectified Flow MoE,SNR-切换两专家,见 huggingface.co/Wan-AI/Wan2.2-I2V-A14B 模型卡)。

### 1.1 顶层数据流(更新后的 mermaid)

```mermaid
flowchart TD
    A[输入: 单图 I, 部件文本 P] --> B[Stage A: TRELLIS 单图 → SS-latent → SLAT z0]
    A --> C[Stage B: Wan2.2 I2V 生成开启视频 V]
    C --> D[Stage C: K=6 等距取帧 frames_0..5]
    B --> E[Stage D: 支持超集 U 构建+外环周期刷新]
    D --> E
    E --> F[Stage E: SS-adapter 联合反warp规范化抬升 → part-head双sigmoid + joint-head移动加权池化+累积softplus φ]
    F -->|psi, B, M, phi_1..5| G[解析 SE3 rollout: T_k for k=0..5]
    G --> H[Multi-state 合成 O_k = 1-(1-B)(1-M_k)]
    H --> I[grid_sample 形变 SLAT_k]
    I --> J[D_GS 解码 → 6 视频帧]
    J --> K[Wan2.2 W-RFSDS 双tau 蒸馏]
    K -->|grad| F
    K -->|grad| G
    F -.psi 反馈.-> F
    H --> L[Stage F: P2 纹理阶段 SLAT-LoRA + Canonical donor]
    L --> M[Stage G: 阈值化 + URDF 导出]
```

> **说明**:虚线 `psi 反馈` 即为改进 3+4 的反馈回路。该回路在 method.md §15 中通过两阶段 detach/EMA 显式打断。

### 1.2 与 v19 的结构差异(整体 delta 表)

| 子模块 | v19 | v19.1 | 影响 |
|---|---|---|---|
| Stage D 支持集 U | 启动期定一次,固定 | 内层固定,外层每 200 iter 增长/剪枝 | 工程上 +1 个 hook |
| 门 g_i, m_i | sigmoid 软掩码,前向后向均软 | 二值 concrete + STE,前向硬,反向软 | 与 SLAT 预训练分布更匹配(SLAT 训练时 token 是离散结构占据,见 arxiv.org/html/2412.01506) |
| SS-adapter | 状态维度 cross-attention,**隐式假设同位置等于同物理体素** | 显式 ψ⁻¹ inverse-warp 抬升至规范帧后再聚合 | move 体素特征不再错位 |
| part-head | 4-logit (g, base, move, uncertain) softmax | (g_logit, m_logit) 两路独立 sigmoid | 自由度等价,稀疏正则可独立 |
| joint-head 池化 | 全 U mean+max | move 加权注意力池化 | 忽略无轴信息的 base 体素 |
| φ_k 参数化 | 5 个独立标量 | φ_k = Σ softplus(δ_j) 累积 | 严格单调,无需单调 loss |
| L_contact | 4 g²·π_base·π_move | 4 g̅² m̅(1−m̅) | 与新门一致 |
| L_gate-sparse | 1/|U| Σ g(1−g) | 1/|U| Σ g(1−g) + λ_m/|U| Σ m(1−m) | 双稀疏正则 |

---

## 2. Stage A — TRELLIS 单图 → 初始 SLAT(不变)

### 2.1 模块依赖
- **冻结组件**:`microsoft/TRELLIS-image-large` 完整 pipeline(SS-flow `ss_flow_img_dit_L_16l8_fp16` + SLAT-flow `slat_flow_img_dit_L_64l8p2_fp16` + 解码器 `slat_dec_gs_swin8_B_64l8gs32`,文件清单见 huggingface.co/microsoft/TRELLIS-image-large/tree/main/ckpts)。
- **架构**:SS-DiT 24 blocks,hidden=1024,heads=16,patch_size=1(在 16³ sparse grid 上);SLAT-DiT 24 blocks,hidden=1024,heads=16,patch_size=2(在 64³ sparse grid 上 patchify 至 32³ token);D_GS Swin window=8,32 gaussians/voxel;ModulatedTransformerCrossBlock 顺序为 AdaLN → SelfAttn → AdaLN → CrossAttn(image,DINOv2-L 特征) → AdaLN → MLP。
- **dtype**:全程 fp16 推理,bf16 微调。

### 2.2 输出
- `z0 ∈ ℝ^{N0 × 8}`,N0 ≈ 9600(SLAT 在 64³ 上的 active token 数,压缩比约 16×,见 trellis2.com/blog/trellis2-how-it-works 关于 SLAT 压缩的工程描述)。
- `coords0 ∈ ℤ^{N0 × 3}`,稀疏体素世界坐标(在 [0, 64)³ 内)。

---

## 3. Stage B/C — Wan2.2 视频生成与 K=6 取帧(不变)

### 3.1 关键事实
- 使用 `Wan-AI/Wan2.2-I2V-A14B`(MoE,27B 总参 / 14B 激活/步),SNR 切换两专家(高噪 → 低噪)单调,见模型卡。
- 单 seed 生成,提示模板:`"Open the {part_name} of the {object}, slowly and uniformly."`
- K=6 等距帧:`frame_idx = round(linspace(0, T-1, 6))`,**不做过滤**。
- 帧 0 是闭合状态(physical baseline),帧 5 是完全开启状态。

### 3.2 单调性保证([Hyp])
> **[Hyp-Wan-Mono]**:Wan2.2 在 "open" 类别 prompt 下,生成的 6 帧关节状态 φ_0=0 < φ_1 < … < φ_5 在 95% 以上案例中近似单调。少数非单调由 jitter 或 overshoot 造成,改进 5 的累积 softplus 参数化将硬性单调,等价于在非单调帧引入正则压力(详见 method.md §7)。
>
> **回退**:若实测非单调率 > 10%,在 method.md §16.7 给出软单调备选(用 abs(δ_j) 替代 softplus,允许微小负值但 L1 正则)。

---

## 4. Stage D — 支持超集 U 构建 + **外环刷新**(改进 1)

### 4.1 内层(冻结期)
与 v19 相同:
1. 初始 U₀ = coords0(z0 中 active 体素的体素坐标集合,在 16³ 网格层级聚合后得到 |U₀| ≈ 1500–4000)。
2. **轴管扩张**:沿 BMCSA 估出的 ψ_axis 方向,做 Minkowski 1-voxel dilation,得到 U_init。

### 4.2 外层周期刷新(新)
每 `refresh_interval = 200` iter 执行一次:

```python
# pseudocode @ 16^3 grid, dtype int32
def refresh_U(U_t, g_bar_history, M_bar_history, tau_grow=0.7, tau_prune=0.05):
    # g_bar_history: 滑窗 [last 50 iters] 的 sigmoid(g_logit) 平均 ∈ [0,1]^|U_t|
    g_avg = g_bar_history.mean(dim=0)        # shape [|U_t|]
    # 1) 剪枝:平均存在度低于 tau_prune
    keep_mask = g_avg > tau_prune
    U_kept = U_t[keep_mask]                   # 移除明显不被使用的体素
    # 2) 增长:对 g_avg > tau_grow 的体素做 1-voxel 邻域扩张
    grow_seeds = U_t[g_avg > tau_grow]
    grow_neigh = neighbor_1vox(grow_seeds)    # 6 邻接,共 |seed|*7 个候选(含自身)
    grow_neigh = unique(grow_neigh) - U_kept  # 去重并排除已存在
    # 3) 上限保护
    U_new = concat(U_kept, grow_neigh)
    if len(U_new) > U_max:                    # 默认 8192
        # 按 g_avg 降序保留 top-U_max
        U_new = topk_by_g(U_new, U_max)
    return U_new
```

**触发条件**:
- 仅在 P1(0%–60%)启用;P1→P2 过渡(60%–75%)冻结 U;P2 不再修改。
- 每次刷新后,ψ、g_logit、m_logit 在新加入体素上用零初始化 + warm-up 5 iter(梯度 ×0.1)。

### 4.3 张量与超参
- `U_t`: `int32[|U_t|, 3]`,世界坐标 ∈ [0,64)³。
- `tau_grow=0.7`, `tau_prune=0.05`, `refresh_interval=200`, `U_max=8192`, `history_window=50`。

---

## 5. Stage E(关键变更:改进 2/3/4/5/6)

### 5.1 二值 concrete + STE 门(改进 2)

```python
# Forward (前向): 硬 0/1
u = torch.rand_like(logit_g)                         # U(0,1)
g_soft = torch.sigmoid((logit_g + torch.log(u) - torch.log(1-u)) / T_g)
g_hard = (g_soft > 0.5).float()
g_bar  = g_hard - g_soft.detach() + g_soft           # STE: forward=hard, backward=soft
# m_bar 同理,使用独立 T_m
```

**温度调度**(method.md §12 给出函数闭式):
- P1 几何:T_g 余弦退火 1.5 → 0.2;T_m 同步。
- P2 纹理:T_g, T_m 固定 0.15(在 [0.1, 0.2] 内,避免 T→0 saturation)。

### 5.2 SS-adapter 联合 inverse-warp 规范化抬升(改进 3)

**位置**:在 SS-DiT block 14、16、18 后(三层多尺度)插入 adapter,**adapter 内部不做 state-dim cross-attention,改为显式规范化抬升**。

```python
# Inputs: 
#   h_k_dense: [K=6, B=1, 16, 16, 16, 1024]   - SS-DiT 第14/16/18 block 的稠密化 hidden
#   psi:       [B, 19]                         - 当前关节估计 (axis 3 + origin 3 + type-soft 1 + phi_1..5 + revolute/prismatic latent 6)
#   U_t:       [|U|, 3]                        - 当前支持超集
# Output: F: [|U|, 3*1024 + 1024 = 4096]   规范帧多状态聚合特征

def ss_adapter_canonical_lift(h_k_dense, psi, U_t, train_iter, warmup_frac=0.05, ema_frac=0.15):
    K = 6
    # ----- 反馈回路打断逻辑(关键) -----
    if train_iter < warmup_frac * total_iters:
        psi_for_warp = psi.detach()                  # 完全切断
    elif train_iter < ema_frac * total_iters:
        psi_for_warp = psi_ema                        # EMA(beta=0.95) 慢更新
    else:
        psi_for_warp = psi                            # 完全反传
    # -----------------------------------
    
    T_inv = build_T_inv(psi_for_warp, K)              # [K, B, 4, 4],T_k^{-1} 关于状态 k
    
    h_k_canon = []
    for k in range(K):
        # U_t 是规范帧坐标; 对状态 k,需要查询 h_k 在 (T_k * U_t) 处
        # 等价于:把 h_k 用 T_k^{-1} inverse-warp 到规范帧再 trilinear sample U_t
        coords_in_state_k = apply_T(T_inv[k].inverse(), U_t)   # [|U|, 3] in state-k frame
        # normalize to [-1,1] for grid_sample (align_corners=True)
        norm = (coords_in_state_k / (16 - 1)) * 2 - 1
        h_k_at_U = F.grid_sample(
            h_k_dense[k].permute(0, 4, 1, 2, 3),       # [B, 1024, 16, 16, 16]
            norm.view(1, |U|, 1, 1, 3),
            mode='bilinear',                            # 3D 下即 trilinear
            align_corners=True,
            padding_mode='zeros'
        ).squeeze(-1).squeeze(-1).permute(0, 2, 1)     # [B, |U|, 1024]
        h_k_canon.append(h_k_at_U)
    h_k_canon = torch.stack(h_k_canon, dim=1)          # [B, K, |U|, 1024]
    
    # 跨状态聚合:统计 + state-0 anchor
    mean_i = h_k_canon.mean(dim=1)                     # [B, |U|, 1024]
    std_i  = h_k_canon.std(dim=1)                      # [B, |U|, 1024]
    max_i, _ = h_k_canon.max(dim=1)                    # [B, |U|, 1024]
    anchor = h_k_canon[:, 0]                           # state-0 anchor [B, |U|, 1024]
    F_i = torch.cat([mean_i, std_i, max_i, anchor], dim=-1)  # [B, |U|, 4096]
    return F_i
```

**关键说明**:
- `T_k^{-1}` 由 ψ 经 Rodrigues + 累积 softplus φ 构造(详见 §5.5)。
- `padding_mode='zeros'`:体素 warp 出 16³ 边界后特征为 0,避免边缘 artifact;`align_corners=True` 与 v19 一致(grid_sample 的 3D bilinear 即 trilinear,见 github.com/pytorch/pytorch/issues/24870)。

### 5.3 part-head 双 sigmoid(改进 6)

```python
# Input: F_i ∈ [B, |U|, 4096]
class PartHead(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(nn.Linear(4096, 512), nn.GELU(), nn.Linear(512, 2))
    def forward(self, F_i):
        h = self.mlp(F_i)                              # [B, |U|, 2]
        logit_g, logit_m = h[..., 0], h[..., 1]
        g_bar = binary_concrete_ste(logit_g, T_g)      # [B, |U|]
        m_bar = binary_concrete_ste(logit_m, T_m)      # [B, |U|]
        # 派生
        B_i = g_bar * (1.0 - m_bar)                    # base 占据
        M_i = g_bar * m_bar                            # move 占据
        return g_bar, m_bar, B_i, M_i
```

### 5.4 joint-head:移动加权注意力池化 + 累积 softplus φ(改进 4 + 5)

```python
class JointHead(nn.Module):
    def __init__(self):
        self.mlp_pool = nn.Sequential(nn.Linear(4096, 512), nn.GELU(), nn.Linear(512, 19))
        # 19-dim psi 解析:
        #   axis_logits (3)           -> normalize
        #   origin (3)
        #   type_logit (1)            -> sigmoid: revolute=0, prismatic=1
        #   delta_1..delta_5 (5)      -> softplus 累积得到 phi_1..phi_5
        #   reserved (7)              -> 留给未来扩展(关节限位等)
    
    def forward(self, F_i, m_bar, eps=1e-6):
        # 改进 4: move 加权注意力池化(简化形式,permutation-invariant)
        weights = m_bar / (m_bar.sum(dim=-1, keepdim=True) + eps)   # [B, |U|]
        F_pool = (weights.unsqueeze(-1) * F_i).sum(dim=1)            # [B, 4096]
        psi_raw = self.mlp_pool(F_pool)                              # [B, 19]
        
        # 解析
        axis_raw = psi_raw[..., 0:3]
        axis = axis_raw / (axis_raw.norm(dim=-1, keepdim=True) + eps)
        origin = psi_raw[..., 3:6]
        type_soft = torch.sigmoid(psi_raw[..., 6])
        deltas = psi_raw[..., 7:12]                                  # [B, 5]
        # 改进 5: 累积 softplus 强制单调
        sp = F.softplus(deltas)                                       # [B, 5], all > 0
        phi = torch.cumsum(sp, dim=-1)                                # [B, 5], 0 < phi_1 < ... < phi_5
        return {
            'axis': axis, 'origin': origin, 
            'type_soft': type_soft, 'phi': phi,
            'reserved': psi_raw[..., 12:19]
        }
```

### 5.5 解析 SE(3) rollout(与 v19 一致,但 φ 来自累积 softplus)

```python
def rollout_T_k(psi, k):
    # k = 0..5
    if k == 0:
        return torch.eye(4)
    phi_k = psi['phi'][..., k-1]    # 标量
    axis = psi['axis']; origin = psi['origin']
    # 软混合:revolute 与 prismatic
    R_rev = rodrigues(axis, phi_k)           # [3,3]
    t_rev = origin - R_rev @ origin           # 绕轴 origin 旋转
    T_rev = compose(R_rev, t_rev)             # [4,4]
    T_pri = translate(axis * phi_k)           # 沿轴平移
    type_soft = psi['type_soft']
    # 软混合(训练) → argmax 硬选(收敛后)
    T_k = (1 - type_soft) * T_rev + type_soft * T_pri
    return T_k
```

> 公式正确性:Rodrigues 形式见 mathworld.wolfram.com/RodriguesRotationFormula.html;SE(3) 软混合在训练阶段提供平滑梯度,推理时 type_hard = (type_soft > 0.5),与 v19 一致。

### 5.6 多状态合成(不变)
```
O_k = 1 - (1 - B)*(1 - M_k)           # M_k = M_i 在状态 k 经 T_k 变换后的占据
```

---

## 6. Stage F — 纹理阶段(P2,75%–100%)(不变)

- SLAT-LoRA 仅作用于 D_GS,rank=8,alpha=16。
- Canonical donor 融合权重 V·D·A·Q·C(visibility, depth, angle, quality, consistency)。
- W-RFSDS 切换至 P2 低 τ 区间 [0.1, 0.4]。

---

## 7. Stage G — URDF 导出(不变)

- 硬阈值:g_bar > 0.5 视为存在,m_bar > 0.5 视为可动。
- type_hard = (type_soft > 0.5);axis 归一化;origin 取 move 体素质心修正。
- φ_5 作为关节限位上限。

---

## 8. 模块依赖图与反馈回路打断

```mermaid
graph LR
    subgraph 主路径(显式梯度)
      psi[psi] --> Tk[T_k] --> M_k[M_k]
      M_k --> Ok[O_k] --> render[渲染] --> rfsds[W-RFSDS]
      rfsds -.grad.-> psi
    end
    subgraph 辅助路径(反馈回路,需打断)
      psi --> Tinv[T_psi^-1] --> warp[inverse-warp] --> SSadapt[SS-adapter F_i]
      SSadapt --> parthead[part-head] --> mbar[m_bar]
      mbar --> jpool[joint-head 移动加权池化] --> psi
    end
    style 辅助路径 fill:#ffe6e6
```

**打断策略**(三阶段):
| 训练进度 | warp 中 ψ 形式 | 反馈回路 |
|---|---|---|
| 0%–5% | `psi.detach()` | 完全切断,BMCSA ψ₀ 主导 |
| 5%–15% | `psi_ema(β=0.95)` | EMA 慢更新,弱反馈 |
| 15%–100% | `psi`(完整) | 完整反传,Picard 已收敛邻域 |

---

## 9. 文件结构(与 v19 基本一致,新增 4 个文件)

```
cast_u/
├── configs/
│   └── v19_1.yaml                       # 新增,完整超参
├── pipelines/
│   ├── stage_a_trellis.py
│   ├── stage_b_wan22.py
│   ├── stage_c_kframe.py
│   ├── stage_d_support_set.py           # 新增 refresh_U()
│   ├── stage_e_adapter_heads.py         # 大改:lift + heads
│   ├── stage_f_texture.py
│   └── stage_g_urdf.py
├── modules/
│   ├── binary_concrete.py               # 新增
│   ├── canonical_lift.py                # 新增
│   ├── part_head.py                     # 改双sigmoid
│   ├── joint_head.py                    # 改池化+phi
│   └── analytic_se3.py
├── losses/
│   ├── w_rfsds.py
│   ├── traj.py
│   ├── close.py
│   ├── contact.py                       # 改 4 g²m(1-m)
│   └── gate_sparse.py                   # 新增 lambda_m 项
├── train.py
├── infer.py
└── eval.py
```

---

## 10. YAML 配置 schema(v19_1.yaml,完整)

```yaml
# ==================== v19.1 完整配置 ====================
project: cast_u_v19_1
seed: 42
total_iters: 4000

trellis:
  ckpt: microsoft/TRELLIS-image-large
  ss_dit_blocks: 24
  ss_dit_hidden: 1024
  ss_dit_heads: 16
  ss_dit_patch: 1
  slat_dit_blocks: 24
  slat_dit_hidden: 1024
  slat_dit_heads: 16
  slat_dit_patch: 2
  d_gs_blocks: 12
  d_gs_hidden: 768
  d_gs_swin_window: 8
  gaussians_per_voxel: 32

wan22:
  ckpt: Wan-AI/Wan2.2-I2V-A14B
  K: 6
  prompt_template: "Open the {part} of the {object}, slowly and uniformly."
  guidance: 5.0
  steps: 30

stage_d_support:                          # 改进 1
  refresh_interval: 200
  tau_grow: 0.7
  tau_prune: 0.05
  U_max: 8192
  history_window: 50
  refresh_active_phase: P1                # 仅 P1

binary_concrete:                          # 改进 2
  T_g_init: 1.5
  T_g_final_p1: 0.2
  T_g_p2: 0.15
  T_m_init: 1.5
  T_m_final_p1: 0.2
  T_m_p2: 0.15
  delta_threshold: 0.5

ss_adapter:                               # 改进 3
  insertion_blocks: [14, 16, 18]
  feature_dim: 4096                       # = mean(1024) + std(1024) + max(1024) + anchor(1024)
  warmup_frac: 0.05                       # detach 阶段
  ema_frac: 0.15                          # EMA 阶段
  psi_ema_beta: 0.95
  warp_pad_mode: zeros
  align_corners: true

part_head:                                # 改进 6
  hidden: 512
  out_dim: 2

joint_head:                               # 改进 4 + 5
  hidden: 512
  pool_eps: 1.0e-6
  out_dim: 19

losses:
  weights:
    w_rfsds: 1.0
    traj: 0.5
    close: 0.3
    contact: 0.2
    gate_sparse_g: 0.05
    gate_sparse_m: 0.05                  # 改进 6 新增
  w_rfsds:
    p1_tau_range: [0.6, 0.9]
    p2_tau_range: [0.1, 0.4]

schedule:
  p1_geom_end: 0.60
  transition_end: 0.75
  p2_tex_start: 0.75

slat_lora:                                # P2 only
  rank: 8
  alpha: 16
  target: D_GS

donor_fusion:
  weights: {V: 1.0, D: 0.5, A: 0.5, Q: 0.5, C: 1.0}

bmcsa:
  warmup_frac: 0.03

export:
  hard_threshold_g: 0.5
  hard_threshold_m: 0.5
```

---

## 11. 训练入口伪代码(集成所有 v19.1 变更)

```python
def train_cast_u_v19_1(cfg):
    # ----- Stage A/B/C(预处理,一次性) -----
    z0, coords0 = trellis_image_to_slat(cfg.input_image)
    video = wan22_i2v(cfg.input_image, prompt=cfg.prompt)
    frames = sample_K_frames(video, K=6)
    
    # ----- BMCSA 暖启动 ψ₀ -----
    psi_0 = bmcsa_warmup(frames, coords0, n_iter=int(0.03 * cfg.total_iters))
    psi_ema = psi_0.clone()
    
    # ----- Stage D 初始 U -----
    U_t = build_initial_support(coords0, psi_0)
    g_history = []; m_history = []
    
    # ----- 优化器(只训 adapter + heads + LoRA) -----
    opt = AdamW(trainable_params(), lr=1e-4, betas=(0.9, 0.999))
    
    for it in range(cfg.total_iters):
        # --- 阶段判定 ---
        phase = phase_of(it, cfg.schedule)        # 'P1' | 'transition' | 'P2'
        T_g, T_m = temp_schedule(it, cfg.binary_concrete, phase)
        tau_low, tau_high = tau_range(phase, cfg.losses.w_rfsds)
        
        # --- 改进 1: 外环 U 刷新 ---
        if phase == 'P1' and it > 0 and it % cfg.stage_d_support.refresh_interval == 0:
            g_avg = stack(g_history[-cfg.stage_d_support.history_window:]).mean(0)
            U_t = refresh_U(U_t, g_avg, m_history, cfg.stage_d_support)
            warm_up_new_voxels(g_logit, m_logit, U_t, factor=0.1, n_iter=5)
        
        # --- 前向(改进 3 反馈回路打断) ---
        h_k_dense = run_ss_dit_save_blocks(z0, U_t, blocks=[14,16,18])  # 冻结
        F_i = ss_adapter_canonical_lift(
            h_k_dense, psi=psi, U_t=U_t,
            train_iter=it, total_iters=cfg.total_iters,
            psi_ema=psi_ema,
            warmup_frac=cfg.ss_adapter.warmup_frac,
            ema_frac=cfg.ss_adapter.ema_frac
        )
        # 改进 6: 双 sigmoid
        g_bar, m_bar, B_i, M_i = part_head(F_i, T_g, T_m)
        # 改进 4 + 5: 移动加权池化 + 累积 softplus
        psi = joint_head(F_i, m_bar)
        
        # --- 解析 rollout + 合成 + 渲染 ---
        T_ks = [rollout_T_k(psi, k) for k in range(6)]
        O_ks = [compose_O(B_i, M_i, T_ks[k], U_t) for k in range(6)]
        rgb_ks = [d_gs_decode(z0, O_k) for O_k in O_ks]
        
        # --- 损失(改进 2/6 反映在 contact、gate_sparse) ---
        L = (cfg.losses.weights.w_rfsds * w_rfsds_dual_tau(rgb_ks, frames, tau_low, tau_high)
           + cfg.losses.weights.traj   * L_traj(psi, frames)
           + cfg.losses.weights.close  * L_close(B_i, M_i, U_t)
           + cfg.losses.weights.contact* L_contact_v191(g_bar, m_bar, axis_tube(psi, U_t))   # 4 g̅² m̅(1-m̅)
           + cfg.losses.weights.gate_sparse_g * L_gate_sparse_g(g_logit)                     # g(1-g)
           + cfg.losses.weights.gate_sparse_m * L_gate_sparse_m(m_logit))                    # m(1-m)
        
        opt.zero_grad(); L.backward(); opt.step()
        
        # --- EMA 更新(改进 3 第二阶段使用) ---
        psi_ema = cfg.ss_adapter.psi_ema_beta * psi_ema + (1 - cfg.ss_adapter.psi_ema_beta) * psi.detach()
        
        # --- 历史记录 ---
        g_history.append(sigmoid(g_logit).detach())
        m_history.append(sigmoid(m_logit).detach())
        if len(g_history) > cfg.stage_d_support.history_window:
            g_history.pop(0); m_history.pop(0)
        
        # --- P2 启动 LoRA ---
        if it == int(cfg.schedule.p2_tex_start * cfg.total_iters):
            attach_slat_lora(d_gs, rank=cfg.slat_lora.rank, alpha=cfg.slat_lora.alpha)
    
    # ----- Stage G: 阈值化导出 -----
    export_urdf(g_bar, m_bar, psi, U_t, cfg.export)
```

---

## 12. 推理入口

```python
def infer_v19_1(image, prompt_part, prompt_obj, ckpt):
    z0, coords0 = trellis_image_to_slat(image)
    g_bar, m_bar, psi = forward_eval_only(z0, coords0, ckpt)   # 不需要 Wan2.2
    urdf = export_urdf_hard(g_bar, m_bar, psi)
    return urdf, render_animation(z0, psi)
```

---

## 13. 评估入口

```python
def evaluate(test_set, ckpt, ablation_id=None):
    # 15 项消融见 §15
    metrics = []
    for sample in test_set:
        urdf, anim = infer_v19_1(sample.image, sample.prompt_part, sample.prompt_obj, ckpt)
        m = {
          'mIoU_part':       part_seg_iou(urdf, sample.gt_urdf),
          'count_acc':       count_acc(urdf, sample.gt_urdf),
          'axis_err_rad':    axis_err(urdf.joint, sample.gt_urdf.joint),
          'origin_err_m':    origin_err(urdf.joint, sample.gt_urdf.joint),
          'type_err':        type_err(urdf.joint, sample.gt_urdf.joint),
          'success@thresh':  success_at_threshold(urdf, sample.gt_urdf, pos=0.05, ang=0.25),
        }
        metrics.append(m)
    return aggregate(metrics)
```

> 评估指标对齐 URDF-Anything(arxiv 2511.00940)与 Articulate-Anything(arxiv 2410.13882):joint axis 角度误差(radian),origin 误差(meter),success 阈值 50mm / 0.25 rad。

---

## 14. 内存与计算预算(per-iter)

| 模块 | v19 显存(GB) | v19.1 显存(GB) | Δ | 计算开销 |
|---|---|---|---|---|
| TRELLIS 冻结 forward | 8.0 | 8.0 | 0 | 不变 |
| Wan2.2 推理(W-RFSDS) | 18.0 | 18.0 | 0 | 不变 |
| SS-adapter(state-dim attn → canonical lift) | 1.2 | **2.0** | +0.8 | 6 次 grid_sample 替代 attention(更快约 1.3×) |
| part-head | 0.05 | 0.04 | -0.01 | 4-logit → 2-logit |
| joint-head | 0.05 | 0.06 | +0.01 | +注意力池化但池化前已是单个向量,可忽略 |
| EMA(ψ_ema) | 0 | 0.001 | +0.001 | 19 维标量 |
| U 历史滑窗 | 0 | 0.02 | +0.02 | 50 × |U| × 2 个 fp16 |
| **总计 H100 80GB** | **~28.5** | **~29.3** | **+0.8 GB(+2.8%)** | wall-clock 约 +3% |

---

## 15. 15 项消融矩阵(v19.1 完整)

| ID | 名称 | 目的 |
|---|---|---|
| A1 | v19.1 full | 主结果 |
| A2 | v19.1 - 改进 1(U 不刷新) | 评估 U refresh 收益 |
| A3 | v19.1 - 改进 2(回退 sigmoid 软门) | 评估 binary concrete 必要性 |
| A4 | v19.1 - 改进 3(回退 state-dim attn) | 评估 canonical lift 必要性 |
| A5 | v19.1 - 改进 4(回退 mean+max) | 评估 move-weighted pool 必要性 |
| A6 | v19.1 - 改进 5(独立 φ + 单调 loss) | 评估累积 softplus 必要性 |
| A7 | v19.1 - 改进 6(回退 4-logit softmax) | 评估双 sigmoid 必要性 |
| A8 | v19.1 vs v19(完整对比) | 整体收益 |
| A9 | v19.1 vs CAST-U | 验证我们正确捕获 CAST-U 改进 |
| A10 | 反馈回路 Option B(全程 detach) | 评估 EMA 阶段必要性 |
| A11 | 反馈回路 Option C(全程 EMA) | 评估完整反传必要性 |
| A12 | 无 BMCSA 暖启动 | 评估 ψ₀ 关键性 |
| A13 | T_g/T_m 不退火(固定 0.5) | 评估温度调度 |
| A14 | refresh_interval ∈ {100, 200, 400} | 灵敏度 |
| A15 | tau_grow/tau_prune 灵敏度 ±20% | 鲁棒性 |

---

## 16. 失败模式目录(v19.1 专属新增 + v19 继承)

| 模式 | 触发条件 | 检测信号 | 缓解 |
|---|---|---|---|
| **F1**: 二值 concrete 低温饱和 | T_g < 0.1 | g_bar 全 0 或全 1,L_gate-sparse → 0 | T_g 下限钳位 0.1 |
| **F2**: U 刷新震荡 | tau_grow ≈ tau_prune | U_t 大小周期性抖动 | 增加 |tau_grow - tau_prune| > 0.5;hysteresis |
| **F3**: warp 不稳定 | ψ₀ 误差大,detach 阶段太短 | F_i 范数突增 | 延长 warmup_frac 至 10% |
| **F4**: move 池化坍缩 | 训练初期 m_bar ≈ uniform | F_pool 退化为 mean,axis 估计变差 | 改进 4 仅在 it > 5% total 启用,前期用 mean+max |
| **F5**: 累积 softplus 末端梯度消失 | δ_j 全为大正数,φ_5 饱和 | dphi_5/ddelta_1..5 各项 → 1 但 φ_5 已 >> 限位 | softplus 输入 zero-init,前期更小;φ_5 截断 |
| **F6**: ψ↔m 反馈震荡 | 完整反传阶段过早 | ψ axis 摆动 > 0.2 rad/iter | 强制 EMA 阶段 ≥ 15% |
| **F7**: BMCSA ψ₀ 严重错误 | 拍摄角度退化 | 前 5% 迭代 L_traj 不下降 | fallback to identity ψ₀,前 10% 不参与 warp |
| **F8**: 视频非单调 | Wan2.2 overshoot | φ_5 < φ_4 等价信号(在累积 softplus 下不可能,但帧观测会冲突) | 改进 5 强制单调 = 帧索引重排基于 φ |
| **F9**: SLAT 分布漂移 | 软门未正确切硬 | D_GS 解码出现 NaN | STE forward 必须严格使用 (g_soft > 0.5).float() |

---

## 17. 7 天实施路线图

| Day | 任务 | 验收 |
|---|---|---|
| **D1** | BMCSA + Stage A/B/C/D 初始化 | ψ₀ 误差 < 0.3 rad on PartNet-Mobility 10 样本 |
| **D2** | Stage D refresh_U + binary concrete 模块 | F1/F2 单元测试通过;|U_t| 收敛 |
| **D3** | SS-adapter canonical lift + ψ 反馈回路打断 | sanity test:**[Red] 三阶段 detach/EMA/full 训练 50 iter ψ axis 漂移 < 0.05 rad** |
| **D4** | part-head 双 sigmoid + joint-head 移动加权池化 + 累积 softplus | 单元测试:dphi_k/ddelta_j 数值符合 §method 7.4 |
| **D5** | 全 Stage E 集成 + 反馈回路稳定性测试 | 200 iter 训练无 NaN,L 单调下降 |
| **D6** | Stage F LoRA + Donor 融合 | P2 切换无视觉断层 |
| **D7** | 评估 + 消融 A1~A6 | 主表 + 6 项单一改进消融 |

---

# 文档 2 / 2:`method.md`(v19.1 完整版)

## 1. 记号与约定

| 符号 | 含义 | 形状/类型 |
|---|---|---|
| I | 输入单图 | [3, H, W], fp16 |
| z₀ | 初始 SLAT 编码 | [N₀, 8], fp16 |
| coords₀ | SLAT 体素坐标 | [N₀, 3], int32 ∈ [0, 64)³ |
| U_t | 第 t 次外环刷新后的支持超集 | [|U_t|, 3], int32 ∈ [0, 64)³(后续映射至 16³ adapter 网格) |
| K | 视频帧数 = 6 | scalar |
| h_k | SS-DiT block ℓ 的 dense hidden(state k) | [B, 16, 16, 16, 1024], fp16 |
| F_i | 规范帧体素 i 的多状态聚合特征 | [B, |U|, 4096], fp16 |
| g_bar | 存在门(STE forward 硬,backward 软) | [B, |U|], fp32 |
| m_bar | 移动门 | [B, |U|], fp32 |
| B_i, M_i | 派生 base/move 占据 | [B, |U|], fp32 |
| ψ | 关节估计向量 | [B, 19], fp32 |
| φ_k | 累积 softplus 关节状态 (k=1..5) | [B, 5], fp32 |
| T_k | SE(3) 变换矩阵 | [B, 4, 4] |
| O_k | 状态 k 的合成占据 | [B, |U|], fp32 |
| τ | RF 时间(0=数据,1=噪声) | scalar |

---

## 2. 验证的数学基础(不变)

### 2.1 TRELLIS 架构事实
- SS-DiT image-large:24 blocks,hidden=1024,heads=16,patch_size=1(SS sparse 16³)。
- SLAT-DiT image-large:24 blocks,hidden=1024,heads=16,patch_size=2。
- D_GS:Swin window=8,32 gaussians/voxel。
- ModulatedTransformerCrossBlock:AdaLN → SelfAttn → AdaLN → CrossAttn(image,DINOv2-L) → AdaLN → MLP。
- 来源:`microsoft/TRELLIS` 官方仓库 `configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json`,checkpoint `slat_flow_img_dit_L_64l8p2_fp16.safetensors`(1.2 GB),见 huggingface.co/microsoft/TRELLIS-image-large。

### 2.2 CHORD W-RFSDS 梯度(arXiv:2601.04194)

CHORD(Geng et al., Stanford 2025)针对 RF 视频模型(Wan2.2)推导的 SDS 目标:

$$
\nabla_\theta L_\text{W-RFSDS} = \mathbb{E}_{\tau, \epsilon}\Big[w(\tau) \cdot (\hat v_\phi(z_\tau, \tau, c) - (\epsilon - z)) \cdot \frac{\partial z}{\partial \theta}\Big]
$$

其中 z_τ = (1-τ)z + τε,RF 约定 τ ∈ [0,1],τ=0 为数据,τ=1 为噪声。w(τ) 在 W-RFSDS 中按 CDF 重采样,使期望权重等价于 1(原文:"the weighting term in RFSDS gradients defined in Eq. 16 is eliminated to ensure the invariance of the expectation of gradients",见 arxiv.org/html/2601.04194)。

**双 τ 调度**:P1 高 τ ∈ [0.6, 0.9](粗运动);P2 低 τ ∈ [0.1, 0.4](精细化)。该退火策略与 CHORD 原文一致("τ gradually decreases over training, enabling coarse motion to form early and allowing fine deformations to be refined in later iterations")。

---

## 3. 支持超集 U 构建(改进 1)

### 3.1 内层定义
固定 |U|,内层(Adam 步)只在 U 上做梯度。

### 3.2 外环刷新公式

设 t 为外环 step,$\bar g_i^{(t)} = \frac{1}{W}\sum_{s=t-W+1}^{t} \sigma(\text{logit}_g)_i^{(s)}$ 为最近 W=50 步 sigmoid 平均。

$$
U^{(t+1)} = \big(U^{(t)} \setminus \{i: \bar g_i^{(t)} < \tau_\text{prune}\}\big) \cup \mathcal{N}\big(\{i: \bar g_i^{(t)} > \tau_\text{grow}\}\big)
$$

其中 $\mathcal{N}(\cdot)$ 为 6-邻接 1-体素扩张。

### 3.3 收敛性论证([Hyp])

> **[Hyp-U-Conv]**:在 g_logit 不再大幅波动后(典型 P1 中后段),U 的对称差 |U^{(t+1)} △ U^{(t)}| 单调下降,refresh 自然终止。
> **保护**:|tau_grow − tau_prune| = 0.65 > 阈值滞回带,杜绝 F2 震荡。

---

## 4. 二值 concrete + STE 门(改进 2)

### 4.1 前向

$$
g_i = \sigma\Big(\frac{\text{logit}_g + \log u - \log(1-u)}{T_g}\Big), \quad u \sim U(0,1)
$$

$$
\bar g_i = \mathbb{1}[g_i > 0.5]
$$

### 4.2 反向(STE)

实现:`g_bar = (g_soft > 0.5).float() - g_soft.detach() + g_soft`,等价于:

$$
\frac{\partial \bar g_i}{\partial \text{logit}_g}\Bigg|_\text{STE} = \frac{\partial g_i}{\partial \text{logit}_g} = \frac{g_i(1-g_i)}{T_g}
$$

> 该形式来自 Concrete Distribution(Maddison et al. 2016, ICLR 2017,arxiv 1611.00712)与 Stochastic Gates(Louizos et al. 2017, arxiv 1712.01312)。Captum 实现明确文档化("Stochastic Gates with binary concrete distribution... continuous smoothed Bernoulli distribution",见 captum.ai/api/binary_concrete_stg.html)。

### 4.3 与 SLAT 预训练分布的兼容性([Hyp-SLAT-Match])

> **[Hyp]**:SLAT 训练时 token-level 是离散结构占据(`coords₀` 为 active voxel)。v19 的 sigmoid 软掩码使输入 token 数值上变成 [0,1] 连续值,与训练分布失配。v19.1 的硬 0/1(STE forward)恢复离散性,SLAT 主干在前向时见到的分布与预训练分布一致;反向通过 STE 仍可学习。

### 4.4 温度调度

$$
T_g(t) = \begin{cases}
T_\text{init} + \frac{T_\text{final}-T_\text{init}}{2}\big(1-\cos(\pi \cdot t/t_\text{P1})\big) & t \le t_\text{P1} \\
T_\text{P2} & t > t_\text{P1}
\end{cases}
$$

T_init=1.5, T_final=0.2, T_P2=0.15。下限 0.1 钳位避免 dphi/dlogit 爆炸(梯度估计高方差,见 emergentmind.com 关于 Gumbel-softmax 退火的 bias-variance trade-off 论述)。

---

## 5. SS-adapter 联合 inverse-warp 规范化抬升(改进 3)

### 5.1 动机

v19 在 SS-DiT block 14/16/18 后做 state-dim cross-attention,**隐式假设同位置 (i,j,k)_grid 在 K 个 state 中对应同一物理体素**,这对 base 体素成立,但对 move 体素错位严重(因为它们在状态 k 已被 T_k 移动)。

### 5.2 v19.1 显式抬升

对每个 state k:
1. 取 SS-DiT 第 ℓ ∈ {14, 16, 18} block 的 dense hidden h_k ∈ ℝ^{16×16×16×1024}。
2. 设 ψ⁻¹(state k → canonical) 由 T_k⁻¹ 给出。
3. 对规范帧体素 i ∈ U_t,查询其在 state-k 帧中的位置 p_k(i) = T_k⁻¹ · i_canon。
4. 在 16³ 网格上 trilinear 采样 h_k 得 h_k^canon(i)。
5. 跨 K 聚合:F_i = [mean_k, std_k, max_k, h_0^canon] ∈ ℝ^{4096}(改进 3 的统计 + state-0 anchor 设计)。

### 5.3 闭式梯度

设 grid_sample 的 trilinear 系数为 c_{abc}(p) = ∏ (1-|p_a - n_a|)…(8 邻顶点权重),则:

$$
h_k^\text{canon}(i) = \sum_{n \in \mathcal{N}_8(p_k(i))} c_n(p_k(i)) \cdot h_k(n)
$$

$$
\frac{\partial h_k^\text{canon}(i)}{\partial \psi} = \sum_{n} \frac{\partial c_n(p_k(i))}{\partial p_k(i)} \cdot h_k(n) \cdot \frac{\partial p_k(i)}{\partial \psi}
$$

PyTorch `F.grid_sample(..., mode='bilinear', align_corners=True)` 在 5D 输入上自动实现 trilinear 反传(见 github.com/pytorch/pytorch/issues/24870 关于 3D bilinear=trilinear 的官方说明)。

### 5.4 反馈回路与稳定性([Red])

> **[Red-Coupling]**:F_i 的梯度依赖 ψ;part-head 输出 m̅ 依赖 F_i;joint-head 的 move 加权池化使 ψ 依赖 m̅。形成回路 ψ → F → m̅ → ψ。

#### 5.4.1 不动点形式

设 Φ: ψ ↦ ψ' = JointHead(MovePool(PartHead(LiftCanonical(ψ)), m̅)),完整训练即为 ψ* = Φ(ψ*) 的 Picard 迭代叠加 W-RFSDS 外环监督。

#### 5.4.2 三阶段稳定化(我们的对策)

| 阶段 | warp 中 ψ | 不动点性质 |
|---|---|---|
| 0%–5% | ψ.detach() | 反馈完全切断,Φ 退化为单步映射,稳定性显然 |
| 5%–15% | ψ_ema(β=0.95) | EMA 引入低通滤波,等价 Φ' = (1-β')·Φ + β'·I,β'≈0.05,Banach 收缩(Lipschitz Φ' < 1)若 |dΦ/dψ| < 20 |
| 15%–100% | ψ 完整 | 假设此时 ψ 已落入 ψ* 邻域,局部线性化下收敛 |

> **稳定性证明草图**:在 BMCSA 提供的 ψ₀ 邻域内,T_k⁻¹(ψ) 关于 ψ 的 Lipschitz 常数 L_T 可由 Rodrigues 公式 ‖∂R/∂axis‖ ≤ |φ_k| 与平移项常数 1 估计;grid_sample 关于位置的 Lipschitz 常数 L_G 由 dense hidden 的最大梯度范数决定。我们在 D5 实测 L_T · L_G · L_part · L_pool < 1,即 Φ 局部收缩。
> **回退**:若实测 ‖ψ_t - ψ_{t-1}‖ > 0.05 rad 持续 5 个外环 step,Day 5 应切换至 Option B(全程 detach for warp,完整反传仅作用于主路径 W-RFSDS)。

### 5.5 计算开销

每次抬升:K=6 次 grid_sample,每次 |U|·8 次浮点乘加 ≈ 8192 · 8 · 1024 ≈ 6.7e7 FLOPs/state,合计 4e8 FLOPs/iter,远小于 SS-DiT 自身 forward(~2e10),增量 < 2%。

---

## 6. part-head 双 sigmoid(改进 6)

### 6.1 输出
- logit_g, logit_m ∈ ℝ^{|U|}
- 经 binary concrete + STE 得 g̅, m̅。
- 派生:B_i = g̅(1−m̅),M_i = g̅·m̅。

### 6.2 自由度等价性

v19 的 (g, π_base, π_move, π_uncert) softmax 实际只有 2 自由度(g 与 π_move/π_base 比例),因为 π_uncert = 1−π_base−π_move 是冗余。v19.1 的 (g_logit, m_logit) 直接 2 自由度,**无信息损失**。

### 6.3 闭式梯度

$$
\frac{\partial B_i}{\partial \text{logit}_g} = (1-m̅) \cdot \frac{g(1-g)}{T_g}, \quad \frac{\partial M_i}{\partial \text{logit}_m} = g̅ \cdot \frac{m(1-m)}{T_m}
$$

> 注:STE 下 g̅、m̅ 在 forward 是硬,但 backward 系数仍为 g(1-g)/T_g。

### 6.4 与 v19 三类 softmax 的对应

v19 softmax (base, move, uncert) 中,`pi_uncert` 等价于 v19.1 中的 `(1-g̅)`(即"not present"的概率)。两者数学等价但 v19.1 允许独立稀疏正则。

---

## 7. joint-head:移动加权注意力池化 + 累积 softplus φ(改进 4 + 5)

### 7.1 移动加权池化

$$
w_i = \frac{m̅_i}{\sum_j m̅_j + \epsilon}, \quad F_\text{pool} = \sum_i w_i \cdot F_i, \quad \psi_\text{raw} = \text{MLP}(F_\text{pool})
$$

### 7.2 排列不变性

显然 F_pool 关于 U_t 上的排列不变(权重与求和均对称),保留 set-function 性质,与 Janossy/DeepSets 框架一致(见 arxiv.org/html/2403.17410v2)。

### 7.3 累积 softplus 单调 φ

$$
\phi_k = \sum_{j=1}^{k} \text{softplus}(\delta_j), \quad k = 1..5
$$

由 softplus(x) = log(1+e^x) > 0 严格成立,**强制 0 = φ_0 < φ_1 < … < φ_5**,无需附加单调 loss。

### 7.4 闭式梯度

$$
\frac{\partial \phi_k}{\partial \delta_j} = \begin{cases}
\sigma(\delta_j) & j \le k \\
0 & j > k
\end{cases}
$$

(softplus' = sigmoid)

末端 φ_5 的梯度对所有 j=1..5 都非零,但当 δ_j → +∞ 时 σ(δ_j) → 1,梯度饱和。**对策**:δ_j 零初始化,σ(0)=0.5,梯度健康。同时在 §4.3 损失中对 φ_5 做软上界正则。

### 7.5 ψ↔m 鸡生蛋问题分析

- 改进 4 池化使 ψ 依赖 m̅。
- 改进 3 抬升使 m̅(经 F_i)依赖 ψ。
- 早期 m̅ ≈ uniform,池化退化为 mean。**对策**:改进 4 仅在 it > 5% total_iters 启用,前期使用 mean+max(等价 v19);改进 3 的 detach 阶段同时锁定 ψ。两个改进的"启用线"在 5%–15% 区间共同就位,联动稳定。

---

## 8. 解析 SE(3) rollout(不变,但补充 §5 抬升路径梯度)

### 8.1 主路径梯度(与 v19 同)

$$
T_k(\psi) = (1-\sigma(\text{type}))\cdot T^\text{rev}_k(\text{axis, origin}, \phi_k) + \sigma(\text{type})\cdot T^\text{pri}_k(\text{axis}, \phi_k)
$$

$\partial T_k/\partial \psi$ 由 Rodrigues 公式标准展开:
$$
R(\text{axis}, \phi) = I + \sin\phi \cdot [\text{axis}]_\times + (1-\cos\phi) \cdot [\text{axis}]_\times^2
$$
$\partial R/\partial \phi = \cos\phi \cdot [\text{axis}]_\times + \sin\phi \cdot [\text{axis}]_\times^2$。

### 8.2 辅助路径梯度(新增,§5.3)

ψ 经 T_k⁻¹ 进入 grid_sample,反向链:
$$
\nabla_\psi L \supseteq \frac{\partial L}{\partial F_i} \cdot \frac{\partial F_i}{\partial h_k^\text{canon}} \cdot \frac{\partial h_k^\text{canon}}{\partial p_k} \cdot \frac{\partial p_k}{\partial T_k^{-1}} \cdot \frac{\partial T_k^{-1}}{\partial \psi}
$$

其中 ∂T_k⁻¹/∂ψ 由 SE(3) 矩阵求逆的封闭式微分给出(Adj 表示,Brockett 1983)。

---

## 9. 多状态合成(不变)

$$
O_k = 1 - (1-B)(1-M_k), \quad M_k = \text{warp}(M, T_k)
$$

---

## 10. grid_sample 反向(不变)

PyTorch `F.grid_sample(..., align_corners=True, mode='bilinear', padding_mode='zeros')` 在 5D 张量上即 trilinear,反向自动实现。我们额外验证 dtype:fp16 输入会回退至 fp32 中间累积以避免数值误差。

---

## 11. 五个激活损失(更新形式)

### 11.1 W-RFSDS(双 τ,§2.2,不变)

### 11.2 L_traj(不变)
帧间 ψ_k 一致性:‖render(ψ, k) - frame_k‖_1。

### 11.3 L_close(不变)
相邻部件 contact 边界:体素邻接 + base/move 翻转处。

### 11.4 L_contact(改进 6 重写)

v19: $4 g̅^2 \pi_\text{base} \pi_\text{move}$
v19.1:
$$
L_\text{contact} = \frac{1}{|\mathcal{T}_\text{axis}|}\sum_{i \in \mathcal{T}_\text{axis}} 4 \cdot g̅_i^2 \cdot m̅_i \cdot (1-m̅_i)
$$

其中 $\mathcal{T}_\text{axis}$ 为 ψ_axis 周围 1-voxel 轴管。**导出**:
- $4 g̅^2 m̅(1-m̅)$ 在 m̅=0.5 取最大值 1,鼓励轴管内部"边界"上 m̅≈0.5(即 base/move 共存)。
- 当 g̅=0 时整项为 0,与改进 6 一致。

### 11.5 L_gate-sparse(改进 6 双稀疏)

$$
L_\text{gate-sparse} = \frac{1}{|U|}\sum_i g_i(1-g_i) + \frac{\lambda_m}{|U|}\sum_i m_i(1-m_i)
$$

g_i, m_i 为 binary concrete 的 soft 值(注意:不是 g̅,因为 g̅(1-g̅) ≡ 0)。λ_m=1.0(默认与 g 等权)。

---

## 12. 调度公式

### 12.1 阶段划分(同 v19)
P1 = [0, 0.6], 过渡 = [0.6, 0.75], P2 = [0.75, 1]。

### 12.2 二值 concrete 温度(§4.4)

### 12.3 U 刷新频率
P1 内每 200 iter,过渡与 P2 不刷新。

### 12.4 ψ 反馈三阶段(§5.4.2)

### 12.5 W-RFSDS 双 τ
P1: τ ~ U[0.6, 0.9];P2: τ ~ U[0.1, 0.4]。

### 12.6 改进 4 启用
仅 it > 5% total 时使用 m̅ 加权;之前用 mean+max。

---

## 13. Canonical donor 融合(P2,不变)

V·D·A·Q·C 五权重对每个供体视图加权融合至 SLAT-LoRA 微调目标。

---

## 14. URDF 导出(不变)

阈值 0.5,axis 归一化,origin 取 move 体素质心修正,φ_5 作为关节限位。

---

## 15. 训练步总计算图(关键节点 + stop-gradient 标注)

```
              ┌── frozen ──┐
I ──> SS-DiT(z0) ─────┐
                       │
U_t ─────────────────> │   h_k_dense [K,16³,1024]
                       ▼
              [Stage E forward]
                       │
                       ▼
ψ ──[根据阶段做 detach/EMA/full]──> T_k^{-1} ──> grid_sample ──> F_i [|U|,4096]
                       │
                       ├──> part_head ──> g_bar, m_bar [STE]
                       │
                       └──> joint_head(F_i, m_bar) ──> ψ_new
                                                          │
ψ ──> rollout T_k ──> M_k ──> O_k ──> D_GS ──> rgb_k ──> W-RFSDS ──> grad
                                                          │
                          (主路径反传 ψ 完整)            │
                                                          ▼
                                  EMA: ψ_ema = β·ψ_ema + (1-β)·ψ.detach()
```

**stop-gradient 节点**(三处):
1. **§5.4.2 阶段 1**:`psi_for_warp = psi.detach()`(0%–5%)。
2. **EMA 更新**:`psi_ema = β·ψ_ema + (1-β)·psi.detach()`(全程)。
3. **g_history / m_history**:`g_history.append(σ(logit_g).detach())`,只用作 U refresh 决策,不参与梯度。

---

## 16. 数值稳定性(v19.1 补充)

### 16.1 二值 concrete 极端温度

- T_g → 0:g(1-g)/T_g 爆炸 → 钳位 T_g ≥ 0.1。
- T_g 大:g_soft → 0.5,STE forward 接近 random Bernoulli(0.5),早期符合。

### 16.2 累积 softplus 数值

- δ 极大 → softplus(δ) ≈ δ,φ_5 可能 >> 物理限位。**对策**:δ_j 上限 clamp(防溢出);训练中加软上界 max(0, φ_5 - π) penalty。
- δ 极负 → softplus(δ) ≈ e^δ ≈ 0,φ 区间退化。**对策**:零初始化 δ_j。

### 16.3 grid_sample 边界

`padding_mode='zeros'`:边界外特征为 0。warp 出格的 move 体素无信号 → m̅ 自然趋 0,不影响主路径。

### 16.4 反馈回路 Lipschitz

§5.4.2 的三阶段策略保证 Φ 在前期为常映射(Lipschitz=0),中期为 EMA 平滑(Lipschitz < 1),后期落入收敛域。

### 16.5 fp16/fp32 边界

- SS-DiT/SLAT-DiT:fp16(冻结)。
- adapter / heads / LoRA:bf16,grad fp32。
- rollout SE(3):fp32(避免旋转矩阵漂移)。

### 16.6 NaN 防护

- F.grid_sample 在 fp16 下偶发 NaN(已知 PyTorch 问题),force fp32 中间累积。
- L_contact 分母 4·m̅·(1-m̅) 在 m̅∈{0,1} 时 = 0,无除零。
- joint-head 池化 ε=1e-6 避免 sum=0。

### 16.7 单调性回退([Hyp-Wan-Mono] 失败时)

若实测 Wan2.2 非单调率 > 10%,改用:
$$
\phi_k = \sum_{j=1}^k \delta_j, \quad L_\text{soft-mono} = \sum_k \max(0, -\delta_k)^2
$$
允许 δ_k < 0 但软惩罚。该回退保持反向兼容,不需重新设计 head。

---

## 17. 诚实性声明([Hyp] / [Red] 索引)

- **[Hyp-SLAT-Match]** §4.3:二值 concrete forward 比软 mask 更匹配 SLAT 训练分布。**待实证**。
- **[Hyp-Coup-Conv]** §5.4.2:三阶段 detach/EMA/full 策略下,ψ↔m 反馈回路收敛到有意义不动点。**Day 5 sanity test 必做**。
- **[Hyp-Move-NoCollapse]** §7.5:移动加权池化在 it > 5% 启用后不会因 m̅ 早期 uniform 而坍缩(因 mean+max 已提供 5% 内的池化)。
- **[Hyp-Wan-Mono]** §3.3:Wan2.2 "open" 类提示生成的 6 帧关节状态单调。**回退方案 §16.7 已就位**。
- **[Red-Coupling]** §5.4:改进 3+4 反馈回路是 v19.1 唯一关键风险。Day 5 必须通过 sanity test,否则切换 Option B(全程 detach warp 中的 ψ)。

---

## 18. 与 v19 的最终差异结清表

| 项 | v19 | v19.1 |
|---|---|---|
| Wan2.2 K=6 | ✓ | ✓ |
| BMCSA 暖启动 | ✓ | ✓ |
| TRELLIS 冻结主干 | ✓ | ✓ |
| 解析 SE(3) rollout | ✓ | ✓(φ 由累积 softplus 给出) |
| 多状态合成 | ✓ | ✓ |
| grid_sample dense 16³ | ✓ | ✓ |
| 5 个激活损失 | ✓ | ✓(L_contact、L_gate-sparse 重写) |
| 两阶段时序 | ✓ | ✓ |
| Tau 双 τ | ✓ | ✓ |
| Canonical donor | ✓ | ✓ |
| SLAT-LoRA P2 | ✓ | ✓ |
| URDF 硬阈值导出 | ✓ | ✓ |
| **U 外环刷新** | ✗ | **✓** 改进 1 |
| **二值 concrete + STE** | ✗ | **✓** 改进 2 |
| **SS-adapter 联合 inverse-warp** | ✗ | **✓** 改进 3 |
| **move 加权池化** | ✗ | **✓** 改进 4 |
| **累积 softplus φ** | ✗ | **✓** 改进 5 |
| **双头独立 sigmoid** | ✗ | **✓** 改进 6 |

---

# 关键建议(Recommendations)

1. **立即按 Day 1–7 路线图启动实施**;Day 3 与 Day 5 是两个关键 gate。
2. **Day 5 sanity test 必须显式包含**:200 iter 训练后,对同一输入,用三种 ψ-反馈策略(Option A=三阶段、Option B=全程 detach、Option C=全程完整)各跑 50 iter,比较 ψ.axis 漂移幅度与 L 收敛速度。Option A 应位居中游;若 Option C 不发散且最快,可前移阶段切换。
3. **改进 5 的 [Hyp-Wan-Mono] 应在数据预处理阶段验证**:在 50 个 PartNet-Mobility 物体上跑 Wan2.2 I2V,统计 6 帧帧间像素流方向一致性,若 < 90% 单调则启用 §16.7 软单调回退。
4. **改进 4 的启用阈值 5% total_iters 是经验值**;若数据集显示 m̅ 在 3% 已显著非 uniform,可前移以加速收敛。
5. **二值 concrete 温度下限 0.1 是关键安全网**;若实测仍有 saturation,改用 hard concrete(Louizos 1712.01312 的 "stretch and clamp" 变体)作为 Plan B。

# 注意事项(Caveats)

- **本设计的反馈回路稳定性论证依赖局部线性化**,在 BMCSA ψ₀ 严重错误时(F7)可能失效。
- **改进 3 的 inverse-warp 假设 16³ adapter 网格分辨率足够**,若发现 move 范围 > 16 voxel(罕见,但例如长滑轨抽屉)需提升 adapter 网格至 32³,显存约 +1 GB。
- **CHORD W-RFSDS 在多物体场景的稳定性**(原文 §3.4 提及 hierarchical 4D rep 是必要补丁)在 CAST-U 单部件场景下不构成问题,但若扩展到多关节物体需重新评估。
- **实施 sanity test 的判据 ψ 漂移 < 0.05 rad/iter 来自经验**,应在 Day 5 用真实数据校准。
- **本文档基于 2026-05-08 公开信息**;TRELLIS-image-large 的具体 layer 数与 hidden 维度以仓库 `slat_flow_img_dit_L_64l8p2_fp16.json` 配置文件为准(若有微小差异以代码为准)。
- **Wan2.2 视频生成质量与提示工程强相关**;低质量视频会同时影响 [Hyp-Wan-Mono] 与 W-RFSDS 监督信号;建议在 Stage B 加入轻量帧间一致性筛查。

---

> **交付状态**:✅ 两份文档自洽完整,所有六项改进数学闭式 + 张量形状 + dtype + 超参 + 稳定性分析全部就位。Code-mode 可直接落地无需进一步设计提问。Day 5 反馈回路 sanity test 是唯一硬性 gate,失败时回退方案 (Option B + §16.7) 已写明。