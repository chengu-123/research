# CAST-U A'B v3 (revision .3) — Method

> 单图（已含 carpet）+ Wan2.2 单视角伪视频 → canonical articulated 3D 资产（base mesh + move mesh + single-DoF joint + UV/atlas + URDF），TRELLIS 主干完全冻结。
>
> **核心创新一句话**：把 TRELLIS 的 `SparseStructureFlowModel` 当成一个 grad-enabled 的 *one-step structural refiner*（不当 sampler 用），通过 Wan2.2 I2V 的 W-RFSDS（含 dual-expert switching）+ Wan VAE latent reconstruction 双监督，把视频蒸馏梯度经解析 SE(3) rollout 与可微 Gaussian 渲染端到端回到 SS-DiT 内部新增的 zero-init residual adapter，在不动 TRELLIS 主干的前提下学习 canonical articulated 几何、part segmentation 与 single-DoF joint；P2 直接优化 **canonical SLAT 本身**（GS 只是 renderer，3D 资产是 SLAT）。
>
> **本版与 v3.2 的差异**（v3.2 → v3.3）：
> 1. **Carpet 是用户输入的一部分** — `s_0_with_carpet` 直接进 Stage A，Wan 生成全程含 carpet 视频，**carpet voxel 保留在 U_object 中渲染**，只在 Stage G Export 时根据 `is_carpet_mask` 剔除。Render 与 target 都含 carpet，一致无冲突。
> 2. **Stage A 不做 multi-seed 筛选**，单 seed 跑（参考 CHORD §A.1 不做 multi-candidate selection）
> 3. **P2 正则改为 normalized delta** `(z_slat - z_slat_init) / slat_std`，按 channel std 归一化后再 L2，避免大 std channel 被惩罚不足
> 4. **Confidence-aware anchor**：三段（base_conf / move_conf / uncertain）替代二元 (1-m_soft)/m_soft，对 P1 边界不确定区更鲁棒
> 5. **Decoded-geometry drift monitor**：每 100 iter 计算 z_slat 解码后的 xyz/scale/opacity 与 z_slat_init 解码的差异，作为 sanity check + 可选 regularizer
> 6. **P2 joint type confidence gate**：若 P1 学到的 type_confidence < 0.7，P2 前 20% 保持 two-branch soft render；之后才硬切
> 7. **Resource fallback 提升为默认**：F=21 + 832×480 全程使用（v3.3.1 切到官方 SUPPORTED_SIZES 后，288×512 / 384×216 等 off-distribution 分辨率作为 fast-debug ablation 保留但不进主线）
> 8. **Provenance 改名** `supervision_provenance`，类别收窄为 `visible_in_all_states / visible_in_open_states / never_visible`，不再声称 donor texture source
> 9. **SCAR：z_t mix → x₀_pred mix**（保留位置对齐 + 修 ODE 一致性）。在 x₀ 空间混合保留 mean alignment（位置对齐效果不变），用原始 per-state ε 重建 z_{t-dt} 保留 noise variance（model 看 in-distribution 输入）。验证依据：用户实测 "drop SCAR → base 漂移"，证明 SCAR 是位置对齐器；x₀ 混合的 mean 与 z_t 混合的 mean 完全等价，所以对齐效果保留。
> 10. **BMCSA：static M_base → dynamic per-block M**（每个 block 内从当前 hidden 实时算 K 状态 token cosine，sigmoid 后作 gate）。修复 static M 来自 Pass-1 末态、Pass-2 演化中 stale 的问题。代价 < 1% compute。
>
> **v3.3 post-critique fixes (C1/S1/S3/S4/M3)**：
> 11. **(C1) τ_sds 改 inverse-CDF of logit-normal**：删除 main_g1 的 phase-based mixture sampling；按 TRELLIS SS-DiT 训练 schedule `logitNormal(mean=1.0, std=1.0)`（ss_flow_img_dit_L_16l8_fp16.json:60-64）的归一化权重 `ŵ(σ) = w(σ)/∫w` 反 CDF 采样。对齐 CHORD Eq.4 + Figure 8 ablation（uniform 失败、inverse-CDF 才稳）。phase-based mixture 收为 ablation 项。
> 12. **(S1) Stage C.5 改周期 silhouette consistency check**：one-time preflight → 每 1000 iter 检查渲染 vs Wan target 的 silhouette IoU；若 IoU < 0.85 触发一次性 U expand + SLAT 重采 + parent_idx 重建。U_seed 初始 dilate radius 1 → 2，更保守覆盖。
> 13. **(S3) P1 末做 deterministic type vote**：删除 P2 中的 `type_uncertain / get_p2_render_mode / two_branch_soft / single_branch_hard` 二分。P1 进度 ~85% 时做 8×(t_ss, seed) deterministic eval 取 type_logit 平均；若 confidence ≥ 0.7 commit；否则克隆 P1 state 两份分别强制 revolute / prismatic，各跑剩余 ~10% iter，比 final SDS loss 选低者 commit。**P2 永远单 branch render with committed type_hard**。
> 14. **(S4) P2 改 tanh reparameterization 替代 nn.Parameter on z_slat**：`delta_z = nn.Parameter(zeros_like(z_init))`, `z_slat = z_init + 3·slat_std·tanh(delta_z)`。manifold-aware 3-σ 硬上界 + 全程可微，AdamW 在 delta_z 上跑无 momentum 破坏；硬截断的边界梯度不连续 / AdamW m,v stale / box≠manifold 三个问题全部消除。L_base_anchor 软正则保留但权重可降。
> 15. **(M3) Bootstrap 删 "SLAT sampler on U_seed"**：13 步 → 12 步。原 B6/B7（method.md 编号 B6 / pipeline.md 编号 B7）输出的 z_slat0_seed 从未被下游使用——joint init 用 z_final 不是 SLAT，后续 always rerun SLAT on U_object 直接覆盖。删除省 ~30s SLAT 采样 + 一份 6-30k×8 中间 latent。
> 16. **(伴随 S3) 删除 P2 confidence gate 渲染路径**：P2 inner loop 简化为单 branch。
> 17. **(NEW.1) Canonical state = s_c（默认 c=2）**：phi 序列零点从 s_0 移到 s_c，即渲染时 `phi_render[c] = 0` 而非 `phi_render[0] = 0`。**Motivation**：TRELLIS 在 s_0 闭合态对 move 部件几何 underrepresent（DINOv2 cond 信息不对称——drawer 完全在 cabinet 内只看到外壳轮廓 → SS-DiT 重建出的"内部 occupancy"偏低 → s_0 实体偏小）；s_2 半开时 drawer 暴露 front face + 部分侧面 → DINOv2 看到更多几何 → canonical move 重建更稳。**L_first 仍 anchor frame 0 = s_0_with_carpet 真实输入**（不损失真实数据 anchor 强度）。**实现**：phi 序列 cumsum + normalize 后做 shift `u_shifted = u - u[c]`，`phi_render_rev/pri = u_shifted × {theta_max, disp_max}` 可正可负（state < c 反向，state > c 正向）。`c = 2` 是 fixed hyperparameter，不自动选（per-instance auto-select c 留作 future work / ablation）。
> 18. **(Q1 增强) B5 加 `O_init_max` 兜底**：`U_seed = {O_mean > 0.3} ∪ {O_max > 0.5} ∪ boundary_band`，覆盖三种 voxel：mean 高置信主体 ∨ 任一 state 强占据 ∨ mean 中等不确定带。原因：大平移 / K state 间不重叠时 latent-mean 解码出的 trajectory 概率会跌出 boundary band (0.1-0.3) → mean only 漏 voxel；O_max（per-state decode 后 voxel-wise max）兜底"至少一个 state 强占据"的位置。代价：6 次 SS-VAE decode（Bootstrap 一次性，不进训练循环）。
> 19. **(NEW.2) Stage A 默认分辨率切到官方 480P 横屏 (H=480, W=832)**：旧 v3.3 默认 288×512 不在 Wan2.2 I2V-A14B SUPPORTED_SIZES（仅 720×1280 / 1280×720 / 480×832 / 832×480 四档），是 off-distribution 的 area scale，会让 W-RFSDS 用 Wan DiT 时 v_pred 不可靠（核心创新 2 失效）。改成官方 (480, 832) 后：lat_h=60, lat_w=104, z_wan_target=[16, 6, 60, 104]，wan_video_target_3FHW=[3, 21, 480, 832]，seq_len = 6·60·104/(2·2) = 9360。Stage D backward 计算开销相对旧默认 ↑2.71×（H800 单卡 P1 5000 iter 从 ~7h 涨到 ~19h，可接受）。`pipelines/stage_a_wan.py` 新增 SUPPORTED_SIZES 硬校验，不在官方列表的 (H, W) 直接 ValueError。fast-debug 用 off-distribution 分辨率作 ablation 不进主线。

---

## 0. 目录

1. 目标与边界
2. 与既有方案的对比和裁决
3. 核心设计哲学
4. 变量定义
5. 输入条件构造（分离 trellis_cond 和 wan_cond）
6. 一次性 Bootstrap（12-step ordering, ★ v3.3.1 M3 删 B6 后）
7. 几何阶段：A'B 主体
8. W-RFSDS 与 Wan2.2 I2V 接口
9. 损失函数
10. 训练协议
11. 纹理阶段
12. 导出（含 deterministic gate protocol）
13. 与 CHORD 的关系与差异
14. 终极一句话

---

## 1. 目标与边界

### 1.1 任务定义

**输入**：
- `s_0_with_carpet` (RGB 闭合状态图, **已含 FreeArt3D grounding disk**) — 用户在输入端合成 carpet，pipeline 不再 add
- `prompt` (自然语言铰接描述)

**输出**：`base.glb` (carpet 剔除), `move.glb`, `joint.json`, `atlas.png`, `supervision_provenance.json` (v3.3 改名), `object.urdf`

### 1.2 硬约束

- TRELLIS 主干（SS-VAE / SS-DiT / SLAT-DiT / D_GS / DINOv2）全程冻结
- Wan2.2 全程冻结
- 不使用外部分割伪标签
- per-instance optimization
- single-DoF（revolute 或 prismatic）

### 1.3 帧数与分辨率

- **F = 21**（Wan2.2 `F % 4 == 1` 约束下的合法值；F=21 给 6 个 latent frames，匹配 K=6 articulation states）
- **分辨率 832 × 480**（H=480, W=832；H/8=60, W/8=104 满足 Wan VAE 8 倍数约束；lat_h=60, lat_w=104 满足 DiT patch_size=(1,2,2) 的 2 倍数约束）

---

## 2. 与既有方案的对比和裁决

（v2 §2 的内容保留，仅追加 v3 在 v2 上的修正）

### 2.1 v2 → v3 的 17 处必修
（v2 → v3 的 17 项见旧版 §2.1 历史记录，本节不再展开）

### 2.2 v3 → v3.1 的 13 处必修（GPT 二轮审查后）

**🔴 严重 Python / shape bug（2 项）**：

1. (D-v3.3) `build_wan_i2v_cond(s, prompt, F=21, ...)` 的 `F=21` 会遮蔽 `torch.nn.functional as F`，`F.interpolate(...)` 直接报错。改 `frame_num=21`。
2. (D-v3.4) `build_wan_i2v_cond` 内 `h_lat / w_lat / F_lat` 只在注释里出现，使用前未赋值。必须 `C_lat, F_lat, h_lat, w_lat = y_vae.shape` 显式取。

**🔴 数值语义 / 坐标系 bug（4 项）**：

3. (D-v3.5) Wan VAE 输入必须 `[-1, 1]` 范围。`image2video.py:259` 是 `TF.to_tensor(img).sub_(0.5).div_(0.5)`。v3 没显式加这一步，rendered RGB 直接喂 wan_vae.encode 会让 latent 分布漂移。
4. (D-v3.6) Joint 坐标系混用。`Gaussian.get_xyz` 在 world space [-0.5, 0.5]（`decoder_gs.py:95` `aabb = [-0.5,-0.5,-0.5, 1,1,1]`），而 `U_object / anchors / ψ.origin / corridor` 在 voxel space [0, 63]。SE(3) warp 直接用 voxel-space `ψ.origin` 旋转 world-space `xyz` 会完全错位。必须显式 `voxel_to_world(u, res=64) = (u + 0.5) / 64 - 0.5` 转换。
5. (D-v3.7) revolute 和 prismatic 共用同一个 `phi`，单位不一致（radian vs world unit）。必须拆成 `u = normalized progress ∈ [0,1]`，再分别 `phi_rev = u * theta_max`、`phi_pri = u * disp_max`，其中 `theta_max / disp_max` 是独立的 learnable scalar。
6. (D-v3.8) `gaussian_parent_idx` 不应从 `gauss.get_xyz` 反推。`decoder_gs.py:101` 显示 `x.coords[x.layout[i]][:, 1:]` 直接给 parent voxel coords；`decoder_gs.py:108-110` 的 `xyz.unsqueeze(1) + offset` flatten 后 parent_idx 严格是 `arange(N_obj).repeat_interleave(32)`。用 `DGSWithParent` wrapper 取得，**trivial 映射**。

**🟡 中等 / 训练正确性（4 项）**：

7. (D-v3.9) dual-expert switching τ > 0.9 与训练 schedule mismatch。当前 schedule `main_g1` 是 `U(0.6, 0.9)`，几乎不会 hit `τ > 0.9`，high_noise_expert 实际不被调用。改 schedule 让 30% 概率取 `U(0.90, 0.98)` 真正用 high expert。
8. (D-v3.10) ~~Wan CFG uncond 用空字符串不是官方做法，应用 `wan_config.sample_neg_prompt`~~。
   **v3.1 撤回**：核到 `shared_config.py:19` 内容后发现 Wan 默认 `sample_neg_prompt` 包含 `"静态"` 和 `"静止不动的画面"`——这是为"动态场景视频"设计的 neg，会**主动惩罚 locked-camera**。直接用会与我们的 locked-off camera 目标冲突。改为：自定义 `build_articulated_prompts()` 分层 prompt——用户输入只描述物体运动，universal camera-lock addon 我们追加，neg prompt 只针对 camera motion + 视觉质量（**omit 静态/静止字样**）。详见 §5.6。
9. (D-v3.11) Stage B B9 "10% 阈值才 rerun SLAT" 会丢 corridor / anchor 关键 voxel。Stage B 是一次性，无论新增多少都应 always rerun SLAT sampler on final U_object。
10. (D-v3.12) Warmup G- 的 α_m 只通过闭态 first-frame render 受间接监督，闭态图看不见 move 边界。改成 0-5% 完全冻结 α_m，5-10% 用 `L_m_prior = BCE(α_m, logit(M_attn_boot_64))` 弱监督，10%+ 才让 RFSDS 推。

**🟡 损失 / shape / 表述（3 项）**：

11. (D-v3.13) LPIPS / L1 reconstruction 的 shape 不对。Wan VAE encode 用 `[3, T, H, W]`（list per video），但 LPIPS 期望 `[N, 3, H, W]`，first frame 用 `[1, 3, H, W]`。两条路径必须明确分开 permute。
12. (D-v3.14) `L_gate = mean(σ(r)(1-σ(r)))` 只鼓励 binary，不鼓励 sparsity。对 fixed conservative U 可能出现 ghost geometry。增加 `L_sparsity_shell = sigmoid(r[shell_mask]).mean()` 只对 boundary / uncertain shell voxel 加 sparsity prior。
13. (D-v3.15) W-RFSDS residual 公式 `residual = v_pred - ε + z_θ` 是 CHORD-style flow-matching 假设。Wan2.2 是 `flow_prediction`（`fm_solvers_unipc.py:321-323` 用 `x0_pred = z_t - sigma_t * v_pred`），数学上一致，但需要在 Day-1 跑 sanity test：用 Wan inference 中真实 `(z_t, t, v_pred)` 三元组验证 `residual` 符号与 scheduler.step 方向一致。

**🟢 CHORD 二轮核读后新增 / 强化（3 项）**：

14. (D-v3.16) **CFG scale 应为 25 → 12 linear decay**（CHORD §A.1 line 788-789 实测），不是 v3 原默认的 5 → 3。**SDS distillation 需要远高于普通 inference 的 CFG**，普通 generation 用 5-7，SDS 必须 12-25 才能让 model 输出 sharp velocity 收敛。v3 默认 5.0 太弱。
15. (D-v3.17) **W-RFSDS Day-1 sanity test 必须是 direction-based 不是 numerical allclose**（详见 §8.4 重写版）。`torch.allclose(residual, v_pred_real, atol=0.1)` 在高维 latent 上没意义。改用：(a) cos(residual, z_θ - x0_pred) 符号；(b) finite-difference descent 后距离下降；(c) 合成 wrong joint/mask 后梯度方向回正确 axis。
16. (D-v3.18) **τ schedule 可选 ablation: inverse-CDF deterministic annealing**（CHORD Eq.4: `h(τ_i) = 1 - i/(I+1)`）。v3.1 默认用 phase-based random uniform 采样，但 CHORD 的 deterministic monotone schedule 更严谨。两者作 ablation 对比。

**🟢 GPT 这次说错的（0 项）**

GPT v3 审查全部 13 项指控均 CONFIRMED。比 v2 那轮审查（GPT 错 1 项）严谨度更高。

### 2.3 GPT 部分正确（PARTIAL）

- (P-v3.1) z_s0 = mean(z_final)：旧 StageB 实测可用。保留默认，加 ablation A/B/C/D（z_final[0] / mean / endpoint-weighted / learned）
- (P-v3.2) 832×480 + F=21 数学上合法（H/8=60, W/8=104, lat/2 也满足 DiT patch=(1,2,2)），但仍需 Day-1 shape test 实测
- (P-v3.3) D_GS geometry 固定时论文表述要准确："RFSDS refines canonical support occupancy, part assignment, and articulation over a fixed TRELLIS SLAT/Gaussian scaffold"，**不要**说 "RFSDS refines canonical Gaussian geometry"
- (P-v3.4) P2 atlas 主张要收窄："atlas is baked from optimized per-Gaussian SH₀ colors after geometry freezing"，**不要**强调 "donor fusion" 主贡献

---

## 3. 核心设计哲学

### 3.1 Canonical-first
（同 v2 §3.1）

### 3.2 冻结主干，只学新增层与残差
（同 v2 §3.2）

### 3.3 W-RFSDS 通过解析路径回到 SS

视频差异 → diff_gaussian_rasterize → Gaussian center / opacity → analytic SE(3) → joint head + part head → SS-DiT hidden → adapter weights。整条链可微。

**Wan2.2 dual expert switch**：W-RFSDS 中根据采样 τ 切换 `wan22_high_noise_dit` (τ > 0.9) 或 `wan22_low_noise_dit` (τ ≤ 0.9)。这与 Wan2.2 官方 inference path 一致。

### 3.4 离散决策放在 D_GS 输出端
（同 v2 §3.4）

### 3.5 outer-loop 操作最小化
（同 v2 §3.5）

### 3.6 Cond 严格分离

TRELLIS SS-DiT 和 Wan2.2 DiT 的 condition 格式完全不同：

- `trellis_cond = DINOv2(s_0_carpet)` — `[1, N_dino, 1024]`，作为 SS-DiT cross-attn KV
- `wan_cond = WanI2VCondBuilder(s_0_clean, prompt, F=21, res=(480,832))` — dict 含 T5 text + 4-ch mask + VAE-encoded first frame，作为 Wan DiT 的 `context` + `y`

两者**不可混用**。

---

## 4. 变量定义

### 4.1 冻结资产

| 符号 | 来源 / file:line | 形状 / 约定 |
|---|---|---|
| `SS-DiT` | `models/sparse_structure_flow.py:56` | `resolution=16, patch_size=1, model_channels=1024, num_blocks=24` |
| `SS-VAE-decoder` | `models/sparse_structure_vae.py:295-306` | 16³→64³ conv decoder，输出 `[B,1,64³]` logit |
| `SLAT-DiT` | `models/structured_latent_flow.py` | sparse transformer，仅 Stage B init 用 |
| `D_GS` | `models/structured_latent_vae/decoder_gs.py:118` | `forward(x: SparseTensor) -> List[Gaussian]`，**32 Gaussians/voxel** |
| `DINOv2-L` | `torch.hub` | 1024-d patch tokens |
| `Wan22 VAE` | `wan/modules/vae2_1.py:603` | `vae_stride=(4,8,8)`，输入 `List[[C=3,T,H,W]]`，输出 16-ch latent |
| `Wan22 high_noise_dit` | `Wan-AI/Wan2.2-I2V-A14B` | τ > 0.9 时用 |
| `Wan22 low_noise_dit` | 同 | τ ≤ 0.9 时用 |
| `T5 text encoder` | UMT5-XXL | Wan I2V text cond |

### 4.2 Bootstrap 资产

| 符号 | 形状 | 含义 |
|---|---|---|
| `z_s0` | `[1, 8, 16, 16, 16]` | BMCSA 合并 SS latent（默认 `mean(z_final, dim=K)`，可 ablate） |
| `z_slat0` | `[N_obj, 8]` | SLAT bootstrap latent，post-norm `(slat_raw * std + mean)` |
| `dit_hidden_cache` | `[1, 4096, 1024] × 3` | block 14/16/18 之后 hidden（诊断用） |
| `O_init` | `[1, 1, 64, 64, 64]` | `sigmoid(SS-VAE-decoder(z_s0))` |
| `M_attn_boot_16` | `[16, 16, 16]` | BMCSA 在 16³ token 上算的 cross-state cosine |
| `M_attn_boot_64` | `[64, 64, 64]` | trilinear 上采到 64³ |
| `is_carpet_mask` | `[64³]` bool | FreeArt3D plane fit 结果 |
| `U_object` | `[N_obj, 3]` int32 | 已剥离 carpet 的候选 voxel |
| `U_object_with_batch` | `[N_obj, 4]` int32 | 加 batch index 列 |
| `gaussian_parent_idx` | `[N_gauss]` int32 | D_GS 输出 Gaussian → U_object 索引（从 D_GS 输出 coords 反推） |
| `ψ_0, φ_0` | dict, `[6]` | StageC joint init |
| `anchors_object` | `[N_a, 3]` | 接触带 |
| `wan_cond_cached` | dict | `{'context': T5([prompt]), 'seq_len': int, 'y': [tensor]}`（一次性算，全程用） |
| `z_wan_target` | `[1, 16, 6, 60, 104]` | Wan VAE encoder 编码 21-帧 clean 视频，post-norm |
| `wan_video_target_3FHW` | `[3, 21, 480, 832]` | clean Wan 视频 raw RGB（LPIPS / L1 用） |

### 4.3 可学习参数

**P1 几何**：

| 符号 | 形状 | 初始化 |
|---|---|---|
| `Δz_s` | `[1, 8, 16, 16, 16]` | zeros |
| `α_g` | `[N_obj]` | zeros (residual) |
| `α_m` | `[N_obj]` | `logit(M_attn_boot_64[U_object].clamp(0.05, 0.95))` |
| `ψ_param` | `[19]` | `encode_joint(ψ_0)` |
| `delta_phi` | `[5]` | `inverse_softplus(φ_0[1:] - φ_0[:-1])` |
| `adapter_{14,16,18}` | MLP | output proj zero-init |
| `H_sup, H_part` | MLP | output proj zero-init |
| `H_joint` | MLP | output proj zero-init |

**P2 纹理**（**P1 几何变量全冻**：α_g, α_m, ψ_param, delta_phi, theta_limit_raw, disp_limit_raw, adapter, H_sup, H_part, H_joint, Δz_s）：

| 符号 | 形状 | 初始化 | 含义 |
|---|---|---|---|
| **`delta_z`** | `[N_obj, 8]` | `torch.zeros_like(z_slat0)` | **tanh-reparameterized canonical SLAT 残差** |

实际进入 forward 的 SLAT：

```
z_slat = z_slat_init + 3.0 * slat_std.view(1, -1) * torch.tanh(delta_z)
```

其中 `z_slat_init = bootstrap.z_slat0.clone().detach()`、`slat_std` 是 SLAT VAE 的 per-channel std（从 normalization config 取，shape `[8]`）。`delta_z = 0` 时 `z_slat = z_slat_init` 保持 BMCSA 起点。

P2 **删除**：`Δ_features_dc`（v3.1 hack）、`donor_weights`、`Δz_slat residual`（旧设计）、`D_GS_LoRA`、直接 `nn.Parameter(z_slat0)`（v3.2/v3.3 设计，被 S4 取代）。

**为什么用 tanh reparameterization 而不是直接 nn.Parameter(z_slat)（S4 修正动机）**：

直接 `nn.Parameter(z_slat0.clone())` + AdamW + L_base_anchor 软正则有三个根本问题——
1. AdamW 维护的 m/v 状态被 hard clamp 破坏；
2. clamp 边界处梯度不连续；
3. clamp 是 L∞ box ≠ valid SLAT manifold（mean ± std 是统计支撑，box clamp 后可能进入 box-内-manifold-外的 dead zone，D_GS 解码 garbage）。

tanh reparameterization 同时解决三者：
- `z_slat` 严格在 `z_init ± 3·slat_std` 内（manifold-aware 软上界，对齐 SLAT 训练分布的 99.7% 支撑）；
- 全程可微、光滑；
- 边界附近 tanh 梯度饱和 → 自然抑制越界更新而非阻断；
- AdamW 跑在 `delta_z` 上，与值的边界解耦，m/v 不被破坏。

L_base_anchor 软正则保留（见 §11.5），但权重可降（tanh 已提供硬上界）；用于鼓励 base voxel 的 `delta_z` 倾向 0。

**训练 - 导出一致**：D_GS 和 D_Mesh 都读取 reparameterized 后的 `z_slat`（不是 `delta_z`），与 D_GS 训练时分布一致。Mesh 导出时复用相同的 tanh 公式重新算 `z_slat` 喂 D_Mesh。

---

## 5. 输入条件构造

### 5.1 Stage A vs Stage B 数据分离

**v3.3 修正**：carpet 是用户输入端合成的，**所有阶段都在含 carpet 的视频上工作**，pipeline 内不再 add/strip carpet。

| 用途 | 内容 | carpet | 帧数 / 形状 |
|---|---|---|---|
| **用户输入** | `s_0_with_carpet` (用户在输入端合成 grounding disk) | **有** | `[3, H_in, W_in]` |
| **Wan 监督 target** | `wan_video_target_3FHW` (Wan I2V 输入 s_0_with_carpet) | **有** | `[3, 21, 480, 832]` |
| **L_first anchor** | `s_0_with_carpet` | **有** | `[3, 480, 832]` |
| **TRELLIS Bootstrap input** | 直接抽样 `s_0...s_5` from `wan_video_target_3FHW` | **有** | 不再额外 add carpet |
| **carpet 检测** | `is_carpet_mask` via FreeArt3D plane fit on `O_init` | — | `[64³]` bool, 仅用于 contact loss / Stage G export 时过滤 |

**核心简化**：
- v3.2 错误地"先生成 clean video → Stage B 再 add carpet"，造成 render/target 不一致
- v3.3 carpet 全程在 video / latent / render / target 里，**Stage D/F 渲染含 carpet 的 canonical asset 直接对齐含 carpet 的 Wan target**
- 仅 Stage G Export 时按 `is_carpet_mask` 过滤掉 carpet voxels

### 5.2 SCAR (x₀-mix) + BMCSA (dynamic M) — v3.3 重写

旧 StageB v4.3 的 SCAR 在 z_t 上混合、BMCSA 用 static M_base 共享 24 block。两者有理论缺陷：

| 旧设计 | 缺陷 | v3.3 修法 |
|---|---|---|
| SCAR 混合 noisy `z_t` (steps 0-7) | 混合后 noise variance 降 66%（权重平方和 0.34），model 看 OOD 输入 → ODE 不一致 | **改混合 `x₀_pred`（clean signal）**，用原始 ε 重建 z_{t-dt}，noise variance 保留 100% |
| BMCSA 用 Pass-1 末态算 `M_base`，Pass-2 全部 24 block 共用 | Pass-2 是 SDEdit 重新加噪去噪，hidden 演化中 M_base stale | **每个 block 内从 current hidden 实时算 M**，per-block adaptive |

**为什么 SCAR 不能 drop（用户实测验证）**：SCAR 通过 latent-space 混合**直接对齐 voxel 的空间位置**——后续 BMCSA 在"已对齐的 token 位置"上做特征平均才有意义。drop SCAR → BMCSA 在错位 token 上工作 → base 漂移。

**为什么 x₀ 混合保留位置对齐**：
$$E[z_t^{(k),\text{mixed}}] = (1-t) · [0.3·z_0^{(0)} + 0.4·z_0^{(k)} + 0.3·z_0^{(K-1)}]$$
混合 x₀_pred = 混合 clean signal estimate = 混合 z_0 的估计 ⇒ 与 z_t 混合的 mean 完全等价。

#### Pass 1（SCAR-x₀，25 steps K-parallel）

```python
K = 6
z_t_K = init_K_noise()                        # [K, 8, 16, 16, 16]
cond_K = [DINOv2(s_k_from_wan_video) for k in range(K)]

for step in range(25):
    t_current = sampler.timesteps[step]
    t_next = sampler.timesteps[step + 1] if step < 24 else 0.0
    
    # 1) K-parallel model forward
    v_K = ss_dit(z_t_K, 1000 * t_current, cond_K)        # [K, 8, 16, 16, 16]
    
    if step < 8:
        # ★ v3.3 SCAR-x₀: 混合 x₀_pred, 不混合 z_t
        # 2) Compute x₀_pred 和 ε_pred per state
        x_0_pred_K = (1 - σ_min) * z_t_K - (σ_min + (1 - σ_min) * t_current) * v_K
        ε_pred_K   = (1 - t_current) * v_K + z_t_K
        
        # 3) SCAR mix on x_0_pred (formula same as old SCAR but on x_0)
        x_0_mixed_K = torch.empty_like(x_0_pred_K)
        for k in range(K):
            x_0_mixed_K[k] = (
                0.3 * x_0_pred_K[0] +
                0.4 * x_0_pred_K[k] +
                0.3 * x_0_pred_K[K - 1]
            )
        
        # 4) Reconstruct z at t_next using ORIGINAL per-state ε (preserves noise stats)
        z_t_K = (1 - t_next) * x_0_mixed_K + (σ_min + (1 - σ_min) * t_next) * ε_pred_K
    else:
        # 8..24: pure K-parallel Euler step
        z_t_K = z_t_K - (t_current - t_next) * v_K

z_final_p1 = z_t_K        # [K, 8, 16, 16, 16]
```

**关键修正**：
- step `< 8`：从 (z_t, v) 提取 (x₀_pred, ε_pred)，混合 x₀_pred，用**原始** ε_pred 重建 z_{t-dt}
- step `≥ 8`：纯 K-parallel Euler，无混合
- **位置对齐保留**（mean 与旧 SCAR 等价），**ODE 一致**（noise variance 正确）

#### Pass 1 末尾计算（同 v3.2）

```python
M_attn_16 = compute_cross_state_cosine(z_final_p1)            # [16, 16, 16]
M_attn_64 = trilinear_upsample(M_attn_16, size=64)
P_aug_intersect = build_augmented_intersection(z_final_p1, M_attn_64)
z_guide_K = encode_with_ss_vae(P_aug_intersect)
```

#### Pass 2（SDEdit + dynamic-M BMCSA，12 steps from t*=0.5）

```python
eps_shared = torch.randn_like(z_guide_K[0])
z_t_K_pass2 = [
    (1 - 0.5) * z_guide_K[k] + (σ_min + (1 - σ_min) * 0.5) * eps_shared
    for k in range(K)
]
z_t_K_pass2 = torch.stack(z_t_K_pass2, dim=0)

timesteps_pass2 = np.linspace(0.5, 0, 13)  # 12 fixed steps

for step in range(12):
    t_current = timesteps_pass2[step]
    t_next = timesteps_pass2[step + 1]
    
    # ★ v3.3: SS-DiT forward with dynamic-M BMCSA at every block
    v_K = ss_dit_with_dynamic_bmcsa(
        z_t_K_pass2,
        timestep=1000 * t_current,
        cond_K=cond_K,
        bmcsa_blocks=range(24),    # all 24 blocks
    )
    
    # Pure Euler step
    z_t_K_pass2 = z_t_K_pass2 - (t_current - t_next) * v_K

z_final = z_t_K_pass2
z_s0 = z_final.mean(dim=0, keepdim=True)
```

#### Dynamic-M BMCSA per block 实现

每个 transformer block 的 self-attention 内：

```python
def ss_dit_block_with_dynamic_bmcsa(h_K, t_emb, cond_K,
                                     K=6, tau_M=0.7, kappa_M=0.05):
    """
    h_K: [K, L=4096, D=1024]
    """
    # ===== Pre-attention: AdaLN modulation =====
    h_K_modulated = adaLN(h_K, t_emb)
    
    # ===== ★ Compute per-block dynamic M from CURRENT hidden =====
    h_normed = h_K_modulated / (h_K_modulated.norm(dim=-1, keepdim=True) + 1e-6)  # [K, L, D]
    
    # Pairwise off-diagonal cosine per token
    # einsum gives [K, K, L] where [i, j, ℓ] = h_normed[i,ℓ,:] · h_normed[j,ℓ,:]
    pairwise = torch.einsum('kld,jld->kjl', h_normed, h_normed)
    eye_K = torch.eye(K, device=h_K.device, dtype=torch.bool)
    pairwise.masked_fill_(eye_K.unsqueeze(-1), 0.0)            # zero out diagonal
    
    agree = pairwise.sum(dim=(0, 1)) / (K * (K - 1))           # [L]
    
    # Sigmoid gate (same hyperparams as v3.2 static M)
    M_dynamic = torch.sigmoid((agree - tau_M) / kappa_M)        # [L]
    M_dynamic = M_dynamic.view(1, -1, 1)                        # [1, L, 1]
    
    # ===== BMCSA dual-attention mix =====
    y_self = self_attn(h_K_modulated)                          # [K, L, D], K-parallel Q/K/V
    y_shared = self_attn(h_K_modulated, share_kv_across_batch=True)  # K averaged K/V
    
    eff_M = torch.clamp(bmcsa_strength * M_dynamic, 0, 1)
    h_K = h_K + (1 - eff_M) * y_self + eff_M * y_shared        # residual
    
    # ===== Cross-attention + MLP (unchanged from vanilla DiT block) =====
    h_K = h_K + cross_attn(adaLN(h_K, t_emb), cond_K)
    h_K = h_K + mlp(adaLN(h_K, t_emb))
    
    return h_K
```

**与 v3.2 static M 的差别**：
- v3.2: `M_base = sigmoid((P_base_shared - 0.5) / τ_M)` from Pass-1 z_final_p1，**static, 全 24 block 共用**
- v3.3: `M_dynamic = sigmoid((agree(h_current) - τ_M) / κ_M)`，**每个 block 用当前 hidden 实时算**

代价：每 block 多 `K² × L × D` 次乘加 ≈ 150M ops。24 block × 12 step = 43G ops，相对 SS-DiT 总 forward 6T ops 是 0.7%，可忽略。

#### v3.2 → v3.3 SCAR + BMCSA 对比一句话

> "SCAR 从 noisy latent 混合改为 clean signal 混合（保位置对齐 + 修 ODE）；BMCSA 从 Pass-1 静态 gate 改为 per-block 当前 hidden 实时算 gate（修 stale），其余机制（K/V cross-batch averaging、bmcsa_strength clamp、(1-M)·y_self + M·y_shared 残差融合）不变。"

### 5.3 TRELLIS canonical cond（仅 Stage D one-step SS-DiT 用）

```python
# v3.3: 直接用用户输入的 s_0_with_carpet, 不再 add carpet
trellis_cond_can = DINOv2(s_0_with_carpet)    # [1, N_dino, 1024]
```

只用 `s_0_carpet` 一个状态的 DINO cond，K_DiT=1 canonical forward。

### 5.4 Wan2.2 I2V cond builder（W-RFSDS 用，v3.1 修两处 Python bug + 加归一化）

**核心修正**：(a) `F` 参数改名避免遮蔽 `torch.nn.functional as F`；(b) `h_lat/w_lat/F_lat` 从 `y_vae.shape` 动态取；(c) Wan VAE 输入加 `[-1, 1]` 归一化。

```python
import torch
import torch.nn.functional as F          # F 在外面是 torch.nn.functional


def to_wan_vae_input(video_3FHW_float01):
    """
    Wan VAE 期望输入 [-1, 1]. 见 wan/image2video.py:259:
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)
    """
    return video_3FHW_float01 * 2.0 - 1.0


def build_wan_i2v_cond(s_0_clean_float01, prompt, frame_num=21, H=480, W=832,
                        device='cuda', wan_config=None):
    """
    一次性构造 Wan2.2 I2V condition (Stage B 内调用，全程缓存).
    
    s_0_clean_float01: [3, H_in, W_in] in [0, 1]
    """
    # ★ v3.1 修 device bug (GPT 2.2): 先把输入挪到 device, 避免 cat with mismatched device
    s_0_clean_float01 = s_0_clean_float01.to(device)
    
    # 1) 构造 fake video [3, frame_num, H, W]: first frame = s_0, rest = zeros
    s_resized = F.interpolate(
        s_0_clean_float01.unsqueeze(0), size=(H, W),
        mode='bicubic', align_corners=False
    ).squeeze(0)                                              # [3, H, W] on device
    
    fake_video_float01 = torch.cat([
        s_resized.unsqueeze(1),                               # [3, 1, H, W]
        torch.zeros(3, frame_num - 1, H, W, device=device),   # [3, F-1, H, W]
    ], dim=1)                                                  # [3, F, H, W]
    
    # ★ v3.1 关键：归一化到 [-1, 1]
    fake_video_neg11 = to_wan_vae_input(fake_video_float01)
    
    # 2) Wan VAE encode (输入 list, 取 [0])
    y_vae = wan_vae.encode([fake_video_neg11])[0]             # tensor
    
    # ★ v3.1 关键：从 shape 动态取 (不再硬写 h_lat=60/w_lat=104)
    C_lat, F_lat, h_lat, w_lat = y_vae.shape
    assert C_lat == 16, f"Wan VAE latent channels expected 16, got {C_lat}"
    
    # 3) 4-channel mask: only first frame visible
    msk = torch.ones(1, frame_num, h_lat, w_lat, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:]
    ], dim=1)                                                  # [1, F+3, h_lat, w_lat]
    msk = msk.view(1, msk.shape[1] // 4, 4, h_lat, w_lat).transpose(1, 2)[0]
    # msk shape: [4, F_lat, h_lat, w_lat]
    
    # 4) Channel-concat mask + vae → 20 channels
    y = torch.cat([msk, y_vae], dim=0)                         # [20, F_lat, h_lat, w_lat]
    
    # 5) ★ v3.1 修正: 用我们自己的分层 prompt, 不直接用用户输入也不直接用 Wan 默认 neg
    pos_prompt, neg_prompt = build_articulated_prompts(prompt, lang='zh')
    context = wan_t5_text_encoder([pos_prompt], device)
    context_null = wan_t5_text_encoder([neg_prompt], device)
    
    # 6) seq_len (按 wan_i2v_A14B.py 的 patch_size=(1,2,2))
    patch_size = (1, 2, 2)
    max_seq_len = (F_lat * h_lat * w_lat) // (patch_size[1] * patch_size[2])
    
    return {
        'context': context,
        'context_null': context_null,
        'seq_len': max_seq_len,
        'y': [y],
        'F_lat': F_lat, 'h_lat': h_lat, 'w_lat': w_lat,
    }
```

**v3.1 修正点**：
1. 参数名 `F` → `frame_num`，避免 Python 作用域遮蔽
2. `C_lat, F_lat, h_lat, w_lat = y_vae.shape` 动态取，不硬写
3. 加 `to_wan_vae_input()` 归一化到 [-1, 1]（Wan 官方约定）
4. **不直接使用 `wan_config.sample_neg_prompt`**（GPT 早先建议有误）—— Wan 默认 neg 包含"静态"和"静止不动的画面"，会**惩罚 camera 静态**，与我们的 locked-camera 目标相反。改用我们自己的 `build_articulated_prompts()`（见 §5.6）

### 5.6 Articulated Prompt 分层设计（**v3.1 关键新增**）

Wan2.2 默认 `sample_neg_prompt`（`shared_config.py:19`）包含 `"静态"` 和 `"静止不动的画面"`，是为"动态全场景视频"设计的——它**会主动惩罚 locked-camera 行为**。我们必须用自己的分层 prompt：

```python
def build_articulated_prompts(user_object_motion_prompt: str, lang='zh'):
    """
    分层构造 Wan I2V prompt:
      用户描述物体运动 (per-object) + 我们的相机锁定约束 (universal)
      自己的视觉质量 neg (universal, 不与 camera-lock 冲突)
    
    Args:
      user_object_motion_prompt: 用户描述铰接部件运动, 例如:
        zh: "抽屉缓慢、连续地向外滑出"
        en: "the drawer slowly slides outward in a continuous motion"
    
    Returns:
      (pos, neg): 直接喂 Wan T5 text encoder.
    """
    if lang == 'zh':
        camera_lock_addon = (
            "。镜头完全固定，相机架在三脚架上，"
            "没有平移、旋转、推拉、变焦、视角变化或镜头切换。"
            "物体主体保持在画面中央且大小不变。"
            "背景、光照、纹理保持一致。"
        )
        # ★ 重要: 这个 neg 故意 OMIT "静态" / "静止" 字样, 避免误伤 camera lock
        # 反而显式列出 camera motion artifacts
        neg_prompt = (
            "镜头平移，镜头旋转，镜头推拉，镜头变焦，镜头晃动，相机移动，"
            "视角变化，画面切换，多镜头切换，蒙太奇，"
            "背景偏移，物体整体平移，物体放大缩小，物体变形，物体身份漂移，"
            "色调艳丽，过曝，欠曝，细节模糊不清，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，瑕疵，水印，字幕，画质模糊"
        )
    elif lang == 'en':
        camera_lock_addon = (
            ". Locked-off camera on a tripod. "
            "No camera pan, tilt, roll, dolly, zoom, or viewpoint change. "
            "Single continuous shot. "
            "Object stays centered and same size throughout. "
            "Stable lighting and background."
        )
        neg_prompt = (
            "camera pan, camera tilt, camera dolly, camera zoom, camera shake, camera roll, "
            "viewpoint change, perspective shift, scene transition, cut, montage, "
            "background drift, object scale change, object identity drift, "
            "overexposed, underexposed, lighting change, "
            "low resolution, blur, artifacts, watermark, subtitle, deformed, distorted"
        )
    else:
        raise ValueError(f"Unsupported lang: {lang}")
    
    pos_prompt = user_object_motion_prompt + camera_lock_addon
    return pos_prompt, neg_prompt
```

**关键设计原则**：
1. **用户只描述物体运动**，例如 `"抽屉缓慢向外滑出"` —— 不需要懂相机术语
2. **camera lock 是 universal addon**，对所有 articulated 物体一致
3. **neg prompt 必须 omit "静态/静止" 字样**，否则 CFG 推 camera 动起来
4. **neg prompt 显式列出 camera-motion 关键词**（"镜头平移、镜头旋转、镜头推拉"等），让 CFG 知道这是要避免的

**用户输入示例**（中文）：
- drawer: `"抽屉缓慢、连续地向外水平滑出"`
- cabinet door: `"柜门缓慢、连续地向外旋转打开"`
- microwave: `"微波炉门缓慢向下旋转打开"`
- laptop: `"笔记本电脑屏幕缓慢向后旋转打开"`
- refrigerator: `"冰箱门缓慢向外旋转打开"`
- washing machine: `"洗衣机门缓慢向外旋转打开"`

**用户输入示例**（English）：
- drawer: `"the drawer slowly slides outward in a continuous rigid motion"`
- cabinet door: `"the cabinet door slowly swings open on its hinge in a continuous motion"`
- microwave: `"the microwave oven door slowly tilts downward to open"`
- laptop: `"the laptop lid slowly rotates upward to reveal the keyboard and screen"`
- refrigerator: `"the refrigerator door slowly swings open on its hinge"`
- washing machine: `"the washing machine door slowly swings open on its single hinge"`

只要用户写清楚 **"什么部件 + 缓慢连续 + 运动方向"** 即可，相机锁定约束我们追加。

### 5.6.1 Build 后的完整 Wan 输入示例（drawer English）

```python
user_motion = "the drawer slowly slides outward in a continuous rigid motion"
pos, neg = build_articulated_prompts(user_motion, lang='en')

# pos:
#   "the drawer slowly slides outward in a continuous rigid motion. Locked-off camera
#    on a tripod. No camera pan, tilt, roll, dolly, zoom, or viewpoint change. Single
#    continuous shot. Object stays centered and same size throughout. Stable lighting
#    and background."

# neg:
#   "camera pan, camera tilt, camera dolly, camera zoom, camera shake, camera roll,
#    viewpoint change, perspective shift, scene transition, cut, montage,
#    background drift, object scale change, object identity drift,
#    overexposed, underexposed, lighting change,
#    low resolution, blur, artifacts, watermark, subtitle, deformed, distorted"
```

注意 neg 显式列出 camera-motion 关键词，但 **故意 omit** "static / motionless"，避免 CFG 把"相机静止"也当成 negative 行为推走。这一点是相对 Wan 官方 `sample_neg_prompt`（含"静态/静止"）的重要修正。

### 5.5 Voxel ↔ World 坐标转换（v3.1 新增）

TRELLIS Gaussian 的 `get_xyz` 在 world space [-0.5, 0.5]（`aabb = [-0.5,-0.5,-0.5, 1,1,1]` 来自 `decoder_gs.py:95`）。但我们的 `U_object / anchors / corridor / ψ.origin` 都在 voxel space [0, 63]。**所有进入 SE(3) warp / contact loss 的几何量必须先转 world space**。

```python
def voxel_to_world(u_xyz, res=64):
    """
    u_xyz: [N, 3] int in [0, res-1]
    Returns: [N, 3] float in (-0.5, 0.5), voxel center convention.
    """
    return (u_xyz.float() + 0.5) / res - 0.5

def world_to_voxel(w_xyz, res=64):
    """Inverse, for export back to voxel indices."""
    return ((w_xyz + 0.5) * res - 0.5).round().long().clamp(0, res - 1)
```

**调用约定**：
- Bootstrap 内 `U_object`, `anchors_object`, `corridor` 全部 cache 为 voxel coords（节省存储）
- Inner loop 进入 SE(3) rollout 前**显式转 world**：`U_world = voxel_to_world(U_object)`
- ψ.origin 是 learnable，**直接以 world space 参数化**（init from `voxel_to_world(rough_pivot_voxel)`）
- ψ.axis 是单位方向向量，无坐标系单位问题

---

## 6. 一次性 Bootstrap（Stage B）

**v3 修复 v2 的 step 顺序循环依赖**。

```python
@torch.no_grad()
def stage_b_bootstrap_v3(s_0_clean, prompt):
    # ===== B1: Wan2.2 生成 clean 21 帧视频 =====
    wan_video_target_3FHW = wan22_i2v_pipeline(
        image=s_0_clean, prompt=prompt,
        n_frames=21, resolution=(480, 832),
        seeds=[s1, s2, s3, s4], steps=50, guidance=5.0,
    )
    # Output: [3, 21, 480, 832]
    
    # ===== B2: 抽 6 帧 + 加 carpet =====
    state_indices = [0, 4, 8, 12, 16, 20]
    s_clean_6 = [wan_video_target_3FHW[:, i] for i in state_indices]
    s_carpet_6 = [add_grounding_disk(s) for s in s_clean_6]
    
    # ===== B3: BMCSA K-parallel SS sampler =====
    trellis_cond_k = [dinov2(s_carpet_k) for s_carpet_k in s_carpet_6]
    z_final, dit_hidden_cache = run_bmcsa_ss_sampler(
        trellis_cond_k, capture_blocks=[14, 16, 18],
    )
    # z_final: [K=6, 8, 16, 16, 16]
    
    # ===== B4: Merge K states & decode occupancy =====
    z_s0 = z_final.mean(dim=0, keepdim=True)        # [1, 8, 16, 16, 16]
    O_init = torch.sigmoid(ss_vae_decoder(z_s0))    # [1, 1, 64, 64, 64]
    
    # M_attn at 16³, then upsample to 64³
    M_attn_boot_16 = compute_token_cosine_consistency(z_final)  # [16, 16, 16]
    M_attn_boot_64 = F.interpolate(
        M_attn_boot_16[None, None], size=64, mode='trilinear', align_corners=True
    ).squeeze()                                       # [64, 64, 64]
    
    # ===== B5: Carpet 检测 → U_seed (无 corridor/anchor 依赖) =====
    # ★ v3.3.1 S1.b: dilate radius 1 → 2 (更保守覆盖, 给周期 silhouette check 减压)
    # ★ v3.3.1 Q1: O_max 兜底, 接住 mean(z_final) 解码漏掉的 large-displacement trajectory
    is_carpet_mask = freeart3d_detect_carpet_plane(O_init)    # [64³] bool flat
    
    O_mean_flat = O_init.view(-1)                              # [64³] = sigmoid(decoder(mean(z_final)))
    O_obj = O_mean_flat * (~is_carpet_mask).float()
    boundary_band = ((O_mean_flat > 0.1) & (O_mean_flat < 0.3)) & ~is_carpet_mask
    
    # ★ Q1: per-state decode + voxel-wise max
    #   K=6 个 latent 单独 decode 得 [K, 1, 64, 64, 64], 然后 voxel-wise max.
    #   捕获 "任一 state 强占据" 的位置, 对大平移 / state 不重叠尤其有效.
    O_per_state = torch.sigmoid(ss_vae_decoder(z_final))       # [K=6, 1, 64, 64, 64]
    O_max = O_per_state.max(dim=0, keepdim=False).values       # [1, 64, 64, 64]
    O_max_flat = O_max.view(-1) * (~is_carpet_mask).float()
    
    # 三路并集: mean 高置信 ∨ 任一 state 强占据 ∨ mean 不确定带
    raw_voxel_flat_idx = torch.nonzero(
        (O_obj > 0.3) | (O_max_flat > 0.5) | boundary_band, as_tuple=False
    ).squeeze(-1)
    raw_xyz = flat_idx_to_xyz(raw_voxel_flat_idx, res=64)   # [N_seed_raw, 3]
    U_seed = dilate_voxels(raw_xyz, radius=2)               # [N_seed, 3]   ★ S1.b
    
    # ★ v3.3.1 M3: 原 B6 (SLAT sampler on U_seed) 已删除 —— z_slat0_seed 从未被下游使用
    # joint init (下面 B6) 用 z_final 不是 SLAT; B8 always rerun SLAT on U_object 直接覆盖
    
    # ===== B6: StageC joint init (uses z_final + M_attn + U_seed) =====
    ψ_0, φ_0, anchors_object = stage_c_joint_init(
        z_final, M_attn_boot_64, O_init, is_carpet_mask, U_seed
    )
    
    # ★ v3.3.1 NEW.1: 把 phi_0 的零点从 s_0 平移到 s_c (canonical state, 默认 c=2)
    #   下游 corridor / warmup_G- / Stage D delta_phi 初始化都用 shifted phi_0.
    #   差分 (phi_0[1:] - phi_0[:-1]) 在 shift 后不变, 所以 delta_phi init 不受影响.
    c = CANONICAL_STATE_IDX                                    # default 2, hyperparameter
    φ_0 = φ_0 - φ_0[c]                                          # φ_0[c] = 0; 其他 φ_0[k] 可正可负
    
    # ===== B7: Expand U_seed → U_object using ψ_0 / anchors =====
    corridor = swept_volume_corridor(ψ_0, φ_0)              # voxel set
    anchor_band = dilate_voxels(anchors_object, radius=2)
    
    U_object_xyz = unique_voxels(U_seed | corridor | anchor_band)
    
    # ===== B8: SLAT sampler on U_object (★ 唯一一次 SLAT 采样) =====
    # 旧 v3 跑过两次 (U_seed 上一次浪费 + U_object 上一次)。M3 删除前者。
    U_object_with_batch = add_batch_col(U_object_xyz)
    z_slat_raw_obj = slat_sampler.sample(
        slat_flow_model,
        sp.SparseTensor(
            feats=torch.randn(len(U_object_xyz), slat_flow_model.in_channels, device=device),
            coords=U_object_with_batch,
        ),
        cond=trellis_cond_k[0], neg_cond=neg_cond,
        steps=25, cfg_strength=7.5, verbose=True,
    ).samples
    z_slat0 = z_slat_raw_obj.feats * slat_std + slat_mean
    
    # ===== B9: ★ v3.1 修 D-v3.8: parent_idx 严格 trivial (arange.repeat_interleave) =====
    # decoder_gs.py:108-110 显示 _xyz.unsqueeze(1) + offset, flatten(0,1) 后:
    #   gaussian_idx = voxel_idx * 32 + gauss_within_voxel
    # 所以 parent_idx 严格是 arange(N_voxel).repeat_interleave(32), 不需要从 get_xyz 反推
    N_voxel = len(U_object_xyz)
    gaussian_parent_idx = torch.arange(N_voxel, device=device).repeat_interleave(32)
    
    # ===== B10: Wan I2V cond builder (一次性, 全程缓存) =====
    wan_cond_cached = build_wan_i2v_cond(
        s_0_clean=s_0_clean, prompt=prompt,
        F=21, H=480, W=832, device=device,
    )
    
    # ===== B11: Wan VAE 编码 21 帧 clean 视频 (latent reconstruction target) =====
    z_wan_target = wan_vae.encode([wan_video_target_3FHW])[0].detach()
    # z_wan_target: [16, 6, 60, 104]
    
    # ===== B12: 写盘 =====
    save_to_disk({
        'z_s0': z_s0,
        'z_slat0': z_slat0,
        'dit_hidden_cache': dit_hidden_cache,
        'O_init': O_init.cpu().numpy(),
        'M_attn_boot_64': M_attn_boot_64.cpu().numpy(),
        'is_carpet_mask': is_carpet_mask.cpu().numpy(),
        'U_object': U_object_xyz.int().numpy(),
        'gaussian_parent_idx': gaussian_parent_idx,
        'psi_0': ψ_0,
        'phi_0': φ_0,
        'anchors_object': anchors_object.numpy(),
        'trellis_cond_can': dinov2(s_carpet_6[0]).detach(),
        'wan_cond_cached': wan_cond_cached,
        'z_wan_target': z_wan_target,
        'wan_video_target_3FHW': wan_video_target_3FHW,
        's_0_clean': s_0_clean,
    })
```

### 6.1 Periodic silhouette consistency check (Stage C.5, ★ v3.3.1 S1 修)

旧 v3.3 是 Stage D 开始前 one-time preflight。**问题**：W-RFSDS 在训练过程中可能"想"激活 U_object 外的 voxel（例如 P1 中后期才学到 axis 的 origin 在 body 外，对应过往 stageC v8.1 用 13-q multi-start 才修好的 7201/7128），一次性 preflight 漏的部分整个训练救不回来。

**S1 修复**：改成**每 1000 iter 一次的周期性 silhouette consistency check**。触发条件由 IoU 阈值（默认 0.85）决定；触发后做一次性 U expand + SLAT 重采 + parent_idx 重建。

```python
@torch.no_grad()
def stage_c5_periodic_silhouette_check(
    bootstrap, learnable, current_state,
    period=1000, iou_threshold=0.85, max_total_expansions=3,
):
    """
    P1 训练每 1000 iter 调一次。比较 N 个采样 state 的当前渲染 silhouette
    与对应 Wan target frame 的 silhouette IoU。低于阈值则一次性扩 U。

    保护：max_total_expansions 限制总扩张次数（防止反复扩到爆显存）。
    """
    if current_state.it % period != 0 or current_state.it == 0:
        return False
    if current_state.n_expansions >= max_total_expansions:
        return False

    # 1) 采 5 个均匀间隔的 state (含闭态 0 和开态 -1)
    state_idx_to_check = [0, 5, 10, 15, 20]
    iou_list = []
    coverage_fail_voxels_all = []

    for k_state in state_idx_to_check:
        # 渲染当前 canonical with current learnables (含 adapter / α_g / α_m / ψ_pred)
        rgb_k = quick_render_state_k(bootstrap, learnable, state_idx=k_state)

        # Wan target frame at this state
        target_k = bootstrap.wan_video_target_3FHW[:, k_state]

        # silhouette = (alpha mask > 0.5)
        sil_pred = silhouette_from_render(rgb_k)
        sil_target = silhouette_from_target(target_k)

        iou_k = (sil_pred & sil_target).sum() / max((sil_pred | sil_target).sum(), 1)
        iou_list.append(iou_k.item())

        if iou_k < iou_threshold:
            # 投影 silhouette diff 到 3D 找候选缺失 voxel
            diff_mask_2d = sil_target & ~sil_pred
            fail_voxels = backproject_silhouette_diff_to_voxel(
                diff_mask_2d, bootstrap.U_object, camera=camera_locked,
                state_k_T=current_state.T_list[k_state],
            )
            coverage_fail_voxels_all.append(fail_voxels)

    min_iou = min(iou_list)
    log(f"[stage_c5] it={current_state.it} states_IoU={iou_list} min={min_iou:.3f}")

    if min_iou >= iou_threshold:
        return False    # 不触发

    # 2) 触发 expand: 取并集 + dilate radius 2
    fail_union = unique_voxels(torch.cat(coverage_fail_voxels_all, dim=0))
    fail_dilated = dilate_voxels(fail_union, radius=2)

    new_voxels = setdiff_voxels(fail_dilated, bootstrap.U_object)
    if len(new_voxels) == 0:
        return False

    log(f"[stage_c5] expanding U by {len(new_voxels)} voxels (n_exp={current_state.n_expansions + 1})")

    U_expanded = unique_voxels(torch.cat([bootstrap.U_object, new_voxels], dim=0))
    bootstrap.U_object = U_expanded
    bootstrap.U_object_with_batch = add_batch_col(U_expanded)

    # 3) Re-run SLAT sampler on expanded U（这一次相比训练时间是小开销 ~30s）
    bootstrap.z_slat0 = rerun_slat_sampler(U_expanded, bootstrap.trellis_cond_can)

    # 4) 重建 parent_idx + 重新初始化 new voxel 的 α_g / α_m
    N_new = len(U_expanded)
    bootstrap.gaussian_parent_idx = torch.arange(N_new, device=device).repeat_interleave(32)

    # 旧 voxel 的 α_g / α_m 保留, 新 voxel 用 logit(0.5)=0 (uncertain) 初始化
    learnable.expand_alpha_g_m_to_new_voxels(U_expanded, init_logit=0.0)

    # 5) 同步 P2 的 delta_z (如已进 P2): expand to new_size, new entries = 0
    learnable.expand_delta_z_to_new_voxels(U_expanded, init=0.0)

    current_state.n_expansions += 1
    return True
```

**关键设计**：
- `period=1000` 而非 every iter（每 iter check 浪费 ~3s）；
- `iou_threshold=0.85` 经验值，要在 toy / 30857 / 7201 上 calibrate；
- `max_total_expansions=3` 兜底（极端情况 U_object 不应该被无限扩，否则说明 Bootstrap 几何信号根本错了，应当 abort 重启）；
- 新 voxel 的 `α_g, α_m` 初始化为 `logit(0.5) = 0`（uncertain），让 W-RFSDS 决定它们是否激活；
- 新 voxel 的 `delta_z`（P2 用）初始化为 0（tanh(0)=0 → z_slat = z_init），保持 manifold；
- 与 Bootstrap 时一致，重采 SLAT 用 `trellis_cond_can`（s_0 cond）。

**为什么不是 every iter**：silhouette check 自己要做 quick render（21 帧 → 5 帧也要 ~2-3s），每 iter 跑会拖慢 30%+ 训练速度；周期检查的延迟是 ≤ 1000 iter，可接受。

**为什么不允许无限 expand**：每次 expand 会让 SLAT 重采 + 渲染 cost 上升；3 次硬上界是 sanity check，超过说明 Bootstrap 阶段失败需要重启而非继续打补丁。

---

## 7. 几何阶段：A'B 主体（Stage D）

### 7.1 RF q_sample（TRELLIS 公式）

```python
σ_min = 1e-5    # trainers/flow_matching/flow_matching.py:62
ε = torch.randn_like(z_s_base)
z_t = (1 - t) * z_s_base + (σ_min + (1 - σ_min) * t) * ε
```

### 7.2 One-step SS-DiT structural refiner（composition wrapper，**v3 完整复刻**）

```python
import torch.nn.functional as F
from trellis.modules.spatial import patchify, unpatchify

class SS_DiT_WithAdapters(nn.Module):
    """Composition wrapper, NOT subclass. Faithful replica of TRELLIS forward."""
    def __init__(self, base_ss_dit, adapters: dict):
        super().__init__()
        self.base = base_ss_dit
        self.adapters = nn.ModuleDict(adapters)
        for p in self.base.parameters():
            p.requires_grad_(False)
    
    def forward_capture(self, x, t_raw, cond):
        """
        x:     [B, in_channels=8, 16, 16, 16]
        t_raw: scalar float in [0, 1]
        cond:  [B, N_dino, 1024]
        Returns: pred_v [B, 8, 16, 16, 16], captured {14, 16, 18}
        """
        B = x.shape[0]
        
        # ★ TRELLIS 约定 t × 1000 for SS-DiT
        t_model = torch.full((B,), float(1000.0 * t_raw),
                             device=x.device, dtype=torch.float32)
        
        # ★ 用 patchify 而非 x.view
        h = patchify(x, self.base.patch_size)                          # [B, 8, 16, 16, 16] (patch=1)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()     # [B, 4096, 8]
        h = self.base.input_layer(h)                                   # [B, 4096, 1024]
        h = h + self.base.pos_emb[None]                                # APE add
        
        t_emb = self.base.t_embedder(t_model)
        if self.base.share_mod:
            t_emb = self.base.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.base.dtype)
        h = h.type(self.base.dtype)
        cond = cond.type(self.base.dtype)
        
        captured = {}
        for k, block in enumerate(self.base.blocks):
            h = block(h, t_emb, cond)
            if str(k) in self.adapters:
                h = h + self.adapters[str(k)](h)            # ★ post-block residual
                captured[k] = h                              # ★ post-adapter hidden
        
        h = h.type(x.dtype)
        
        # ★ 补 layer_norm (v2 缺)
        h = F.layer_norm(h, h.shape[-1:])
        
        # ★ out_layer + permute + view + unpatchify (v2 缺)
        h = self.base.out_layer(h)                          # [B, 4096, out_channels]
        h = h.permute(0, 2, 1).view(
            B, h.shape[-1], *[self.base.resolution // self.base.patch_size] * 3
        )
        pred_v = unpatchify(h, self.base.patch_size).contiguous()
        
        return pred_v, captured
```

### 7.3 Hidden → U 坐标采样（v3 修正 reshape 和 grid_sample 轴序）

```python
def sample_hidden_at_U(hidden_dict, U_object_xyz, occ_logits):
    """
    hidden_dict: {14: [B, 4096, 1024], 16: ..., 18: ...}
    U_object_xyz: [N, 3] int in [0, 63]
    occ_logits: [B, 1, 64, 64, 64]
    """
    B = hidden_dict[14].shape[0]
    C = hidden_dict[14].shape[-1]
    grid_res = round(hidden_dict[14].shape[1] ** (1/3))    # = 16
    
    # Token order: row-major over (D, H, W) via meshgrid(..., indexing='ij')
    # See sparse_structure_flow.py:103
    
    # ★ Coord normalization: U 在 64³, 除以 63
    d = U_object_xyz[:, 0].float() / 63.0 * 2 - 1     # [N], -1..1
    h_axis = U_object_xyz[:, 1].float() / 63.0 * 2 - 1
    w = U_object_xyz[:, 2].float() / 63.0 * 2 - 1
    
    # ★ grid_sample 5D 期望顺序 (x=W, y=H, z=D), stack [w, h, d]
    grid = torch.stack([w, h_axis, d], dim=-1)         # [N, 3]
    grid = grid.view(1, -1, 1, 1, 3).expand(B, -1, -1, -1, -1)   # [B, N, 1, 1, 3]
    
    sampled_list = []
    for block_idx in [14, 16, 18]:
        h_token = hidden_dict[block_idx]               # [B, 4096, 1024]
        
        # ★ permute + contiguous + view (v2 错: 直接 view)
        h_grid = h_token.permute(0, 2, 1).contiguous().view(
            B, C, grid_res, grid_res, grid_res
        )                                              # [B, 1024, 16, 16, 16]
        
        f = F.grid_sample(
            h_grid, grid,
            mode='bilinear',         # 5D bilinear = trilinear
            align_corners=True,
            padding_mode='border',
        )                                              # [B, 1024, N, 1, 1]
        f = f.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, N, 1024]
        sampled_list.append(f.squeeze(0))               # [N, 1024]
    
    # Fourier positional encoding
    pe = fourier_pe(U_object_xyz.float() / 63.0, num_freqs=6)   # [N, 36]
    
    # Local occ logit per voxel
    U_flat_idx = (U_object_xyz[:, 0] * 64 * 64
                  + U_object_xyz[:, 1] * 64
                  + U_object_xyz[:, 2])
    occ_at_U = occ_logits.view(-1)[U_flat_idx].unsqueeze(-1)   # [N, 1]
    
    feat = torch.cat([*sampled_list, pe, occ_at_U], dim=-1)
    # 总维度: 3·1024 + 36 + 1 = 3109
    return feat, occ_at_U
```

**关键修正（v2 → v3）**：
- `permute(0,2,1).contiguous().view(B, C, R, R, R)` 还原 (D,H,W) 网格（v2 直接 view 把 token/channel 内存搞混）
- grid stack 顺序 `[w, h, d]`（v2 写成 `[d, h, w]` 与 grid_sample 5D 约定不符）

### 7.4 几何底座 + residual heads

```python
z_s_base = bootstrap.z_s0 + Δz_s
occ_logits = ss_vae_decoder(z_s_base)              # [1, 1, 64, 64, 64]

feat, occ_at_U = sample_hidden_at_U(
    captured, bootstrap.U_object, occ_logits
)                                                  # [N_obj, 3109]

r = occ_at_U.squeeze(-1) + α_g + λ_sup  * H_sup(feat).squeeze(-1)
b =                        α_m + λ_part * H_part(feat).squeeze(-1)
```

`α_g` zero-init residual：初始 `r_i = occ_at_U[i] = logit(O_init_i)`，单次 prior。

### 7.5 BinaryConcrete + STE gate

```python
g_obj = BinaryConcreteSTE(r, T_g)
m_obj = BinaryConcreteSTE(b, T_m)
```

### 7.6 Joint residual + 单位分离的 phi（**v3.1 修 D-v3.7**）

```python
# ★ pool 用 soft sigmoid(b), 不用 hard m_obj (BinaryConcrete forward)
m_soft = torch.sigmoid(b)
weights = m_soft / (m_soft.sum() + 1e-6)
F_pool = (weights.unsqueeze(-1) * feat).sum(dim=0)     # [3109]

# H_joint 输出 axis (3) + origin (3) + type_logit (1) + theta_limit_raw (1) + disp_limit_raw (1) + delta_u (5) + reserve = 19
Δψ = H_joint(F_pool)
ψ_for_warp = stage_detach(ψ_param, f_global, mode="joint")
ψ_pred = project_joint(ψ_for_warp + λ_joint * Δψ)

# ψ_pred 含字段:
#   axis        [3]    单位向量
#   origin      [3]    world space (-0.5, 0.5)
#   type_logit  [1]    sigmoid 后是 prismatic 概率
#   theta_limit_raw [1]  softplus 后是 revolute 最大角度 (radians)
#   disp_limit_raw  [1]  softplus 后是 prismatic 最大位移 (world unit)
```

### 7.7 Normalized progress + canonical-state shift + 单位分离的 phi rollout（**v3.1 修 D-v3.7 + v3.3.1 NEW.1**）

之前 v3 直接 `phi_render = linear_interp(phi)` 然后 revolute 用 angle 解释，prismatic 用 displacement 解释——单位不一致。v3.1 改成 normalized progress + 各自 limit。**v3.3.1 NEW.1 再加一步**：把 phi 的零点从 s_0 平移到 s_c (canonical state, 默认 c=2)，避开 TRELLIS 闭合态对 move 几何 underrepresent 的 bias。

```python
# 累积 softplus 得 5 个正增量
delta_u_inc = F.softplus(learnable.delta_phi)                    # [5], >0
u_raw = torch.cat([
    torch.zeros(1, device=device),
    torch.cumsum(delta_u_inc, dim=0)
])                                                                # [6], 严格递增

# ★ Normalize to [0, 1]
u = u_raw / (u_raw[-1] + 1e-6)                                   # [6], u[0]=0, u[-1]=1

# ★ v3.3.1 NEW.1: shift canonical 零点到 s_c (默认 c=2)
#   u[c] 处为 0; state < c 的 u_shifted 为负 (反向 warp), state > c 为正
c = CANONICAL_STATE_IDX                                          # default 2
u_shifted = u - u[c]                                             # [6], u_shifted[c]=0

# 21 帧插值 (用 shifted u, 不再单调从 0 到 1)
u_render = linear_interp_through(u_shifted, n_out=21)            # [21], 可正可负

# ★ 拆 revolute / prismatic 各自 range
theta_max = F.softplus(ψ_pred.theta_limit_raw)                    # scalar, radians
disp_max  = F.softplus(ψ_pred.disp_limit_raw)                     # scalar, world unit

phi_render_rev = u_render * theta_max                             # [21], radians, 可正可负
phi_render_pri = u_render * disp_max                              # [21], world units, 可正可负
```

**与 v3.3 旧设计的差异**：

| 项 | v3.3 旧 (canonical = s_0) | v3.3.1 NEW.1 (canonical = s_c, c=2) |
|---|---|---|
| `u[0]` | 0 (state 0 = canonical) | -u[2] < 0 (state 0 在 canonical 反方向) |
| `u[c=2]` | 0.4 左右 | **0** (canonical) |
| `u[5]` | 1 | 1 - u[2] ≈ 0.6 |
| `phi_render_rev[0]` | 0 (state 0 是 zero rotation) | 负值 (canonical 反向 rotate 到闭合) |
| `phi_render_rev[2]` | u[2] * θ_max | **0** (canonical) |
| SE3_revolute / SE3_prismatic 处理负 phi | n/a | 反向 rotation / 反向 translation, 解析公式天然支持 |

这样 `phi_render_rev[c] = 0, phi_render_pri[c] = 0`（canonical 在 state c）；其他 state 通过有符号 phi 双向 warp 到自身位置。Two-branch render 时各自用自己的 phi。

**初始化**：
- `theta_limit_raw` init from `inverse_softplus(π/2)` ≈ `inverse_softplus(1.57)` 让默认 revolute 最大 90°
- `disp_limit_raw` init from `inverse_softplus(0.3)` 让默认 prismatic 最大 0.3 world unit (≈30% 物体尺寸)
- `delta_u` zero-init → cumsum 后是 `[0, 1.05, 2.10, 3.15, 4.20, 5.25]` → normalize 后 `u = [0, 0.2, 0.4, 0.6, 0.8, 1.0]` → **shift 后 `u_shifted = [-0.4, -0.2, 0, 0.2, 0.4, 0.6]`** (with c=2)

**为什么 L_first 不变（仍 anchor frame 0 = s_0 真实输入）**：L_first 比对的是渲染 frame 0 vs s_0_with_carpet 真实图。渲染 frame 0 时 SE3 用 `phi_render[0]` 已经是负值，canonical 经反向 warp 自然落到 s_0 位置（闭合态）。anchor 比对仍然成立，且用的是真实数据。如果改成 anchor s_2 (= Wan 生成的中间帧)，会用 hallucinated 数据 anchor → 错。

**为什么 c=2 是经验默认 + 不自动选**：
- c=2 对 6-state, 0.5 step/state 这种均匀 trajectory 是中点附近 (u≈0.4)
- 不同物体最佳 c 可能不同 (长抽屉 vs 短门 vs 微波炉门)
- per-instance auto-select c (e.g., 按 move volume argmax) 留作 future work + ablation
- 第一版固定 c=2 跑通；ablation 计划见 §16

### 7.8 Canonical Gaussian 解码 + warp + opacity gating（**v3.1 修 D-v3.8 parent_idx trivial**）

```python
# ★ v3.1: DGSWithParent wrapper, parent_idx 严格 trivial
class DGSWithParent(nn.Module):
    """
    Wrapper over frozen D_GS.
    Returns (gauss, parent_idx) where parent_idx maps Gaussian → U_object index.
    
    Per decoder_gs.py:101-110, D_GS does:
        xyz = (x.coords[x.layout[0]][:, 1:] + 0.5) / resolution     # [N_voxel, 3]
        offset = tanh(offset) / resolution * 0.5 * voxel_size
        _xyz = xyz.unsqueeze(1) + offset                              # [N_voxel, 32, 3]
        flatten(0, 1)                                                  # [N_voxel * 32, 3]
    
    So Gaussian index = voxel_idx * 32 + gauss_within_voxel.
    parent_idx is TRIVIAL: arange(N_voxel).repeat_interleave(32).
    """
    def __init__(self, d_gs_frozen):
        super().__init__()
        self.d_gs = d_gs_frozen
        for p in self.d_gs.parameters():
            p.requires_grad_(False)
    
    def forward(self, sparse_in, n_gauss_per_voxel=32):
        # Same input ordering as `x.coords[x.layout[0]]`
        gauss_list = self.d_gs(sparse_in)
        gauss = gauss_list[0]
        N_voxel = sparse_in.coords[sparse_in.layout[0]].shape[0]
        parent_idx = torch.arange(N_voxel, device=sparse_in.coords.device).repeat_interleave(n_gauss_per_voxel)
        # parent_idx[i*32 + j] = i, for i ∈ [0, N_voxel), j ∈ [0, 32)
        return gauss, parent_idx


sparse_in = SparseTensor(
    feats=bootstrap.z_slat0,
    coords=bootstrap.U_object_with_batch
)
gauss_can, gaussian_parent_idx = d_gs_with_parent(sparse_in)

# Frozen geometry channels
xyz_canon     = gauss_can.get_xyz                  # ★ world space [-0.5, 0.5]
opacity_canon = gauss_can.get_opacity              # post-sigmoid, [N_gauss, 1]
rot_canon     = gauss_can.get_rotation             # [N_gauss, 4] quaternion
scale_canon   = gauss_can.get_scaling              # [N_gauss, 3]
sh_canon      = gauss_can._features_dc             # [N_gauss, 1, 3] SH₀ DC

# Per-voxel → per-Gaussian (trivial via parent_idx)
g_per_gauss = g_obj[gaussian_parent_idx]
m_per_gauss = m_obj[gaussian_parent_idx]


def warp_gauss_state_t(T_t):
    """
    T_t: [4, 4] SE(3) matrix
    Returns: arrays ready for diff_gaussian_rasterize.
    
    base 贡献：位置不变, opacity = g·(1-m), 旋转不变
    move 贡献：位置 warp, opacity = g·m, 旋转 quaternion 也跟着旋转
    """
    R = T_t[:3, :3]                                # [3, 3]
    t = T_t[:3, 3]                                 # [3]
    
    # Base contribution
    means_base    = xyz_canon
    opacity_base  = opacity_canon.squeeze(-1) * g_per_gauss * (1 - m_per_gauss)
    rot_base      = rot_canon
    
    # Move contribution: ★ quaternion 必须随 R 旋转
    means_move    = (R @ xyz_canon.T).T + t                          # [N_gauss, 3]
    opacity_move  = opacity_canon.squeeze(-1) * g_per_gauss * m_per_gauss
    R_quat        = rotation_matrix_to_quaternion(R)                  # [4]
    rot_move      = quat_multiply(R_quat[None].expand(rot_canon.shape[0], 4),
                                  rot_canon)                          # [N_gauss, 4]
    
    # Concat
    means_all    = torch.cat([means_base, means_move], dim=0)         # [2*N_gauss, 3]
    opacity_all  = torch.cat([opacity_base, opacity_move], dim=0).unsqueeze(-1)
    rot_all      = torch.cat([rot_base, rot_move], dim=0)
    scale_all    = torch.cat([scale_canon, scale_canon], dim=0)
    sh_all       = torch.cat([sh_canon, sh_canon], dim=0)
    
    return means_all, opacity_all, rot_all, scale_all, sh_all


# Two-branch render (joint type soft blend at RGB)
# ★ v3.3.1 NEW.1: phi shift to canonical state c, 见 §7.7 完整说明
delta_u_inc = F.softplus(delta_phi)
u_raw = torch.cat([torch.zeros(1, device=device), torch.cumsum(delta_u_inc, dim=0)])  # [6]
u = u_raw / (u_raw[-1] + 1e-6)                                                         # [6] in [0, 1]
u_shifted = u - u[CANONICAL_STATE_IDX]                                                 # [6], shifted[c]=0
u_render = linear_interp_through(u_shifted, n_out=21)                                  # [21], 可正可负

theta_max = F.softplus(ψ_pred.theta_limit_raw)
disp_max  = F.softplus(ψ_pred.disp_limit_raw)
phi_render_rev = u_render * theta_max                                                  # [21], radians, 可正可负
phi_render_pri = u_render * disp_max                                                   # [21], world units, 可正可负

T_revolute  = [SE3_revolute(ψ_pred.axis, ψ_pred.origin, phi_render_rev[t]) for t in range(21)]
T_prismatic = [SE3_prismatic(ψ_pred.axis, phi_render_pri[t]) for t in range(21)]

rgb_revolute  = torch.stack([
    diff_gaussian_rasterize(*warp_gauss_state_t(T_revolute[t]),
                            camera=camera_locked) for t in range(21)
])                                                  # [21, 3, 480, 832]
rgb_prismatic = torch.stack([
    diff_gaussian_rasterize(*warp_gauss_state_t(T_prismatic[t]),
                            camera=camera_locked) for t in range(21)
])

type_soft = torch.sigmoid(ψ_pred.type_logit)
rgb_frames = (1 - type_soft) * rgb_revolute + type_soft * rgb_prismatic    # [21, 3, 480, 832]
```

---

## 8. W-RFSDS 与 Wan2.2 I2V 接口

### 8.1 Wan dual-expert + [0, 1000) timestep + [-1, 1] 归一化（**v3.1 修 D-v3.5 / D-v3.10 / D-v3.16**）

```python
def W_RFSDS_Wan(rgb_frames_3FHW_float01, wan_cond, τ_raw, cfg_scale=20.0):    # ★ v3.1: CHORD 实测 25-12, 默认 20
    """
    rgb_frames_3FHW_float01: [3, 21, 480, 832] in [0, 1]  (grad-enabled, from renderer)
    wan_cond: dict from build_wan_i2v_cond (cached, context_null 已用 sample_neg_prompt)
    τ_raw: scalar in [0, 1]
    """
    # ★ v3.1 关键: 归一化到 Wan VAE 期望的 [-1, 1] (D-v3.5)
    rgb_frames_neg11 = rgb_frames_3FHW_float01 * 2.0 - 1.0
    
    # ★ Wan VAE encode takes List[Tensor] of [C=3, T, H, W]
    z_θ_list = wan_vae.encode([rgb_frames_neg11])             # List[Tensor], grad-enabled
    z_θ = z_θ_list[0].unsqueeze(0)                            # [1, 16, F_lat=6, h_lat=60, w_lat=104]
    
    # SDS q_sample (CHORD-style, σ_min=0)
    with torch.no_grad():
        ε = torch.randn_like(z_θ)
        z_τ = (1 - τ_raw) * z_θ.detach() + τ_raw * ε
        
        # ★ Wan timestep IS [0, 1000) directly, NOT *1000 inside
        t_wan = torch.tensor([τ_raw * 999.0], device=z_τ.device, dtype=torch.float32)
        
        # ★ Dual-expert switch (boundary = 0.9 from wan_i2v_A14B.py:36;
        #   image2video.py:189 uses `t >= boundary`, so use >= here for consistency)
        wan_model = wan22_high_noise_dit if τ_raw >= 0.9 else wan22_low_noise_dit
        
        # ★ Wan model input is List[Tensor [C_in, T, H, W]], no leading batch
        # ★ z_τ and y are concatenated INSIDE model (model.py:444-445)
        x_input = [z_τ.squeeze(0)]                            # List[[16, 6, 60, 104]]
        
        # ★ CFG: cond and uncond
        v_pred_cond = wan_model(
            x_input, t=t_wan,
            context=wan_cond['context'],
            seq_len=wan_cond['seq_len'],
            y=wan_cond['y'],
        )[0]                                                   # [16, 6, 60, 104]
        
        v_pred_uncond = wan_model(
            x_input, t=t_wan,
            context=wan_cond['context_null'],
            seq_len=wan_cond['seq_len'],
            y=wan_cond['y'],
        )[0]
        
        v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
        v_pred = v_pred.unsqueeze(0)                          # [1, 16, 6, 60, 104]
    
    # SDS residual (CHORD Eq. 3)
    residual = v_pred - ε + z_θ.detach()
    
    # SDS gradient through inner product
    loss = (residual.detach() * z_θ).sum() / z_θ.numel()
    return loss
```

**v3.1 关键修正汇总**：
1. `rgb_frames * 2 - 1` 归一化到 [-1, 1]（D-v3.5）
2. `wan_vae.encode([rgb_frames_neg11])`：list 输入，[0] 取
3. `t_wan = τ_raw * 999.0`：Wan 时间步 [0, 1000) **不要 ×1000**
4. Dual-expert switch `τ_raw >= 0.9` (与 image2video.py:189 一致)
5. Wan model 输入是 `List[Tensor [C, T, H, W]]`（无 batch 维）
6. `wan_cond['y']` 在 model 内部 channel-concat 到 x（model.py:444-445）
7. CFG: cond + uncond 两次 forward。`context_null` 通过 `wan_cond` 已用 `sample_neg_prompt`（D-v3.10）

### 8.2 Wan VAE latent reconstruction（**v3.1 加归一化**）

```python
def L_latent_rec_Wan(rgb_frames_3FHW_float01, z_wan_target):
    """
    rgb_frames_3FHW_float01: [3, 21, 480, 832] in [0, 1]
    z_wan_target: [16, 6, 60, 104], cached in bootstrap (already from [-1,1] input).
    """
    rgb_frames_neg11 = rgb_frames_3FHW_float01 * 2.0 - 1.0
    z_render = wan_vae.encode([rgb_frames_neg11])[0]      # grad-enabled
    return ((z_render - z_wan_target.detach()) ** 2).mean()
```

Stage B 缓存 `z_wan_target` 时也必须用归一化版：
```python
wan_video_target_float01 = wan_video_target_3FHW.float() / 255.0
z_wan_target = wan_vae.encode([to_wan_vae_input(wan_video_target_float01)])[0].detach()
```

### 8.3 RGB reconstruction（**v3.1 修 shape D-v3.13**）

shape 约定：
- **Render 输出**：`[T, 3, H, W]` = `[21, 3, 480, 832]`
- **LPIPS / L1 / first frame**：需要 `[N, 3, H, W]` 标准 batch 输入
- **Wan VAE encode**：需要 `[3, T, H, W]` （list per video）

两条路径必须显式 permute 分开：

```python
# 渲染输出
rgb_frames_T3HW = render_21_with_warp(...)              # [21, 3, 480, 832] in [0,1]

# Permute 给 Wan VAE
rgb_frames_3THW = rgb_frames_T3HW.permute(1, 0, 2, 3)   # [3, 21, 480, 832]
L_sds     = W_RFSDS_Wan(rgb_frames_3THW, wan_cond, τ_sds)
L_lat_rec = L_latent_rec_Wan(rgb_frames_3THW, bootstrap.z_wan_target)

# LPIPS / L1 用原本 shape
wan_target_T3HW = bootstrap.wan_video_target_3FHW.permute(1, 0, 2, 3).float() / 255.0
L_rgb_rec = F.l1_loss(rgb_frames_T3HW, wan_target_T3HW) + lpips_loss(rgb_frames_T3HW, wan_target_T3HW)
L_first   = (
    F.l1_loss(rgb_frames_T3HW[0:1], (s_0_clean.float() / 255.0).unsqueeze(0))
    + lpips_loss(rgb_frames_T3HW[0:1], (s_0_clean.float() / 255.0).unsqueeze(0))
)
```

### 8.4 Day-1 Direction-Based Sanity Test for W-RFSDS Residual（**v3.1 D-v3.17 重写**）

之前 v3 用 `torch.allclose(residual, v_pred_real, atol=0.1)` 验证 residual 公式 — 在 16×6×36×64 维 latent 上 element-wise allclose 没有意义，容易误判（CFG 后的 v_pred 不等于训练分布 raw velocity，z_0 不严格 clean，scheduler sigma vs τ 线性映射可能有偏差）。

**v3.1 改用 4 个 direction-based test**：

```python
@torch.no_grad()
def w_rfsds_direction_sanity_test(s_0_clean, prompt, n_iters=20):
    """
    Test A: cos(residual, z_θ - x0_pred) 的符号 — gradient direction sanity
    Test B: finite-difference descent 让 distance to x0_pred 下降
    Test C: 合成 wrong joint/mask 后 gradient 指向正确 axis/region
    Test D: CFG on/off 时 residual 尺度对照
    """
    # ---------- Setup: 跑 1 step Wan inference 拿真实参考 ----------
    wan_cond = build_wan_i2v_cond(s_0_clean.float()/255.0, prompt, ...)
    
    # 渲染一个 baseline rgb_frames (canonical from BMCSA)
    rgb_baseline_3FHW = render_canonical_video(bootstrap)
    z_baseline = wan_vae.encode([rgb_baseline_3FHW * 2.0 - 1.0])[0].unsqueeze(0)
    
    τ_test = 0.7
    ε = torch.randn_like(z_baseline)
    z_τ = (1 - τ_test) * z_baseline + τ_test * ε
    t_wan = torch.tensor([τ_test * 999.0], device=device)
    wan_model = wan22_high_noise_dit  # since τ=0.7 < 0.9, low_noise but doesn't matter for direction test
    
    v_pred_cond = wan_model([z_τ.squeeze(0)], t=t_wan,
                            context=wan_cond['context'], seq_len=wan_cond['seq_len'],
                            y=wan_cond['y'])[0].unsqueeze(0)
    v_pred_uncond = wan_model([z_τ.squeeze(0)], t=t_wan,
                              context=wan_cond['context_null'], seq_len=wan_cond['seq_len'],
                              y=wan_cond['y'])[0].unsqueeze(0)
    v_pred = v_pred_uncond + 20.0 * (v_pred_cond - v_pred_uncond)    # cfg=20
    
    # Scheduler 的 x0 prediction (Wan flow_prediction convention):
    #   x0_pred = (1 - σ_t) * z_t / (1) - σ_t * v_pred ; 简化版: x0 = z_t - τ * v_pred
    x0_pred = z_τ - τ_test * v_pred
    
    # 我们的 SDS residual
    residual = v_pred - ε + z_baseline
    
    # ========== Test A: residual 方向应指向 x0_pred 方向 ==========
    # SDS gradient 推 z_θ 向 x0_pred (model 认为更对的方向)
    # 所以 residual 与 (z_θ - x0_pred) 应负相关 (residual 是负梯度方向)
    cos_A = F.cosine_similarity(
        residual.flatten().unsqueeze(0),
        (z_baseline - x0_pred).flatten().unsqueeze(0),
        dim=1
    ).item()
    print(f"Test A: cos(residual, z_θ - x0_pred) = {cos_A:.4f}")
    assert cos_A > 0.3, f"residual direction wrong, expected > 0.3, got {cos_A}"
    
    # ========== Test B: finite-difference descent 后 distance 下降 ==========
    η = 1e-3    # small step
    z_θ_new = z_baseline - η * residual
    dist_before = (z_baseline - x0_pred).norm().item()
    dist_after = (z_θ_new - x0_pred).norm().item()
    print(f"Test B: distance to x0_pred: {dist_before:.4f} → {dist_after:.4f}")
    assert dist_after < dist_before, f"FD descent did not decrease distance"
    
    # ========== Test C: 合成 wrong joint, gradient 指向 correct axis ==========
    # 用 ψ_GT 渲染真值 video, 加 noise 扰动到 ψ_perturbed, 看 d(loss)/d(ψ) 方向
    ψ_gt = synthesize_gt_joint(s_0_clean)
    ψ_perturbed = ψ_gt + perturbation_along(some_axis, magnitude=0.1)
    
    rgb_perturbed = render_with_joint(ψ_perturbed)
    z_perturbed = wan_vae.encode([rgb_perturbed * 2 - 1])[0]
    residual_perturbed = compute_residual(z_perturbed, ...)
    
    grad_psi = autograd.grad(
        (residual_perturbed.detach() * z_perturbed).sum(), ψ_perturbed, retain_graph=True
    )[0]
    
    expected_direction = (ψ_gt - ψ_perturbed)
    cos_C = F.cosine_similarity(grad_psi, expected_direction, dim=0).item()
    print(f"Test C: cos(grad_ψ, ψ_gt - ψ_perturbed) = {cos_C:.4f}")
    assert cos_C > 0.2, f"gradient does not point back to GT, got {cos_C}"
    
    # ========== Test D: CFG on vs off — residual 尺度变化合理 ==========
    v_pred_no_cfg = v_pred_cond    # 直接用 cond 不混 uncond
    residual_no_cfg = v_pred_no_cfg - ε + z_baseline
    
    scale_with_cfg = residual.norm().item()
    scale_no_cfg = residual_no_cfg.norm().item()
    print(f"Test D: residual norm with CFG={scale_with_cfg:.4f}, no CFG={scale_no_cfg:.4f}")
    # CFG 会放大 residual, 但 direction 应该一致
    cos_D = F.cosine_similarity(
        residual.flatten().unsqueeze(0),
        residual_no_cfg.flatten().unsqueeze(0),
        dim=1
    ).item()
    assert cos_D > 0.7, f"CFG drastically changes residual direction: cos={cos_D}"
    
    print("All 4 sanity tests passed.")
```

**通过标准**：
- Test A：cos > 0.3 ✓
- Test B：distance 下降 ✓
- Test C：cos > 0.2（合成扰动指向 GT） ✓
- Test D：cos(CFG, no-CFG) > 0.7（CFG 改 scale 不改方向） ✓

若任一 fail，**回退方案**（按优先级）：
1. 反号：`residual = -(v_pred - ε + z_θ)` —— 如果 cos_A < 0 但 |cos_A| > 0.3
2. 用 Wan scheduler-consistent 形式：直接 `residual = z_θ - x0_pred = τ · v_pred`（CHORD Eq.3 的等价代数形式）
3. 检查 CFG 是否过强导致 v_pred saturate：降到 cfg=12 重试 Test D

Day-1 必做项。**未通过 sanity test 之前禁止启动主训练**。

### 8.5 τ schedule: inverse-CDF of logit-normal（★ v3.3.1 C1 修：翻转默认）

**v3.3.1 默认**：inverse-CDF of `logitNormal(mean=1.0, std=1.0)`（TRELLIS SS-DiT 训练 t-schedule，证据 `ss_flow_img_dit_L_16l8_fp16.json:60-64`）。原因：

1. CHORD §3.2 Eq.(3) 的 W-RFSDS 推导显式要求 `σ ~ ŵ(σ) = w(σ)/∫w` —— 把 w(σ) 权重折进采样分布，loss 内不再带 w；
2. CHORD §4.3 Figure 8 ablation 实证 uniform sampling 失败，inverse-CDF 才稳；
3. Wan2.2 训练 schedule 也是 logit-normal（SD3 系列约定），与 TRELLIS 一致；
4. v3.3 旧 phase-based mixture 把 τ 强行压到 [0.6, 0.98] 高噪段，**完全没有 CHORD 推荐的中段（mode≈0.5）密集采样**，理论上违反 SDS 收敛条件。

```python
def sample_tau_inverse_cdf_logit_normal(
    iter_idx, total_iters, mean=1.0, std=1.0, jitter=0.0,
):
    """
    inverse-CDF of training-time weight w(σ) for SS-DiT (LogitNormal(1, 1)).
    
    τ 单调下降 from 高分位 → 低分位; 加可选 jitter 避免完全 deterministic.
    用 scipy.stats.logistic / norm 或预算 1000-bin lookup.
    """
    quantile = 1.0 - (iter_idx + 1) / (total_iters + 1)
    if jitter > 0:
        quantile = (quantile + (torch.rand(1).item() - 0.5) * jitter).clamp(1e-3, 1 - 1e-3)
    
    # logit-normal: x = sigmoid(mean + std * Phi^{-1}(quantile)) where Phi is std normal CDF
    z = stats.norm.ppf(quantile)
    tau = 1.0 / (1.0 + math.exp(-(mean + std * z)))
    return float(tau)
```

**单调下降性质**：iter=0 时 τ ≈ 0.95（重整体 layout），iter=total 时 τ ≈ 0.05（重细节）。

**与 dual-expert 的关系（次要修正）**：v3.3 用 phase-mix 是为强制 30% 触发 Wan high_noise_expert (τ ≥ 0.9)。inverse-CDF logit-normal(1,1) 下，τ ≥ 0.9 的概率是 `P(sigmoid(N(1,1)) ≥ 0.9) = P(N(1,1) ≥ logit(0.9)) ≈ 14%`，仍能自然触发 high expert，不需要 phase-mix 强制。

**Ablation 计划**（C1 修后）：

| 实验 | τ 采样 | CFG | 预期 |
|---|---|---|---|
| **A1 default** | **inverse-CDF logit-normal(1,1)** | **25→12** | **主结果（CHORD 对齐）** |
| A2 ablation | phase-based mixture (v3.3 旧默认) | 25→12 | 应当差 (高噪段过密, 中段无信号) |
| A3 ablation | uniform U(0.05, 0.95) | 25→12 | 应当显著差 (CHORD §4.3 Figure 8 同结论) |
| A4 ablation | inverse-CDF logit-normal | 5→3 (普通 inference CFG) | 应当几何模糊 / 不收敛 (CHORD §A.1 CFG 25→12 实证) |
| A5 ablation | inverse-CDF logit-normal(0, 1) | 25→12 | mean=0 → mode=0.5; 与 mean=1 mode=0.73 对比 |

A1 vs A2/A3 直接验证 C1 修正必要性；A4 验证 CFG 范围必要性；A5 验证 schedule mean 敏感度。

### 9.1 P1 几何阶段总损失

$$
\mathcal{L}_\text{geom} = \lambda_\text{sds}\mathcal{L}_\text{SDS} + \lambda_\text{lat}\mathcal{L}_\text{latent-rec} + \lambda_\text{rgb}\mathcal{L}_\text{rgb-rec} + \lambda_\text{first}\mathcal{L}_\text{first} + \lambda_\text{contact}\mathcal{L}_\text{contact} + \lambda_\text{gate}\mathcal{L}_\text{gate} + \lambda_\text{shell}\mathcal{L}_\text{shell-sparse} + \lambda_\text{m-prior}\mathcal{L}_\text{m-prior} + \lambda_z\mathcal{L}_z
$$

各项实现见 v2 §9 + v3.1 增量：

- **L_gate**（同 v2，鼓励 g/m → 0 或 1）：`mean(σ(r)(1-σ(r)) + σ(b)(1-σ(b)))`
- **L_shell_sparse**（v3.1 新增，对 boundary / uncertain shell voxel 鼓励稀疏）：
  ```python
  shell_mask = bootstrap.is_in_uncertain_shell      # [N_obj] bool, from Stage B
  L_shell_sparse = sigmoid(r[shell_mask]).mean()
  ```
  目的：固定保守 U 容易出现 ghost geometry（shell voxel 都被 RFSDS 推到 g=1），加 shell sparsity 防止
- **L_m_prior**（v3.1 新增，仅在 Warmup 5-10% 启用）：
  ```python
  L_m_prior = F.binary_cross_entropy_with_logits(
      α_m, torch.sigmoid(logit(bootstrap.M_attn_boot_64[U_object].clamp(0.05, 0.95)))
  )
  ```
  目的：闭态 first-frame 看不见 move 边界，靠 prior anchor α_m 不偏离 BMCSA M_attn 太远

- **L_contact**（同 v2，**v3.1 注意 contact band 要 voxel_to_world**）：
  ```python
  anchors_world = voxel_to_world(bootstrap.anchors_object, res=64)
  L_contact = contact_anchor_loss(ψ_pred, anchors_world)
  ```
  ψ_pred.origin 已经在 world space，不需要再转

权重默认：
- λ_shell：0.02（弱正则）
- λ_m_prior：0.5（warmup 阶段强）→ 0（geometry phase 后关闭）

---

## 10. 训练协议

### 10.1 阶段表（★ v3.3.1 C1 修 τ_sds 改 inverse-CDF + S3 加 P1 type vote）

下表中 P1 阶段被切成 **Main G1a (10-50%)** + **Type Vote (50-55%)** + **Main G1b (55-65%)**，新增 vote 阶段是 S3 的核心。

| 阶段 | iter 比例 | `t_ss` | `τ_sds` | detach | `λ_sup, λ_part, λ_joint` | `T_g, T_m` | `λ_sds, λ_lat, λ_rgb` | `λ_m_prior` | SS-DiT? | α_m 状态 | type 状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Warmup G-** | **0–5%** | — | — | full detach | 0, 0, 0 | 1.5 | 0, 0, 0 | 0 | 跳过 | **冻结** | learnable |
| Warmup G0 | 5–10% | 固定 0.30 | 固定 0.85 | full detach | 0, 0.02, 0 | 1.5 | 1.0, 0, 0 | 0.5 | 启用 | BCE prior 启用 | learnable |
| **Main G1a** | **10–50%** | `U(0.25, 0.55)` | **inv-CDF logit-normal(1,1)** ★C1 | 0–5% / 5–15% / 15%+ full | 0→0.3, 0→0.3, 0→0.5 | 1.5→0.6 | 1.0, 0.1, 0 | 0.5→0 (10-30% 线性 decay) | 启用 | 自由学 | learnable (two-branch render) |
| **Type Vote** ★S3 | **50–55%** | — | — | — | held | held | 0 (eval only) | 0 | eval only | held | **deterministic vote 8×(t,seed)** → commit_branch ∈ {rev, pri, **dual-clone**} |
| **Main G1b** ★S3 | **55–65%** | `U(0.25, 0.55)` | inv-CDF logit-normal(1,1) | full grad | 0.3, 0.3, 0.5 | 0.6→0.5 | 1.0, 0.1, 0 | 0 | 启用 | 自由学 | **commit_branch 内 single-branch**；若 dual-clone 则两份 P1 state 各跑剩余 ~10% iter |
| Transition | 65–75% | `U(0.20, 0.40)` | inv-CDF logit-normal(1,1) | full grad | 0.3, 0.3, 0.5 | 0.5→0.2 | 0.5, 0.5, 0.1 | 0 | 启用 | 自由学 | type **committed** (dual-clone 已在 G1b 末选 loss 低者) |
| P2 Texture | 75–100% | n/a | inv-CDF logit-normal(1,1) | 几何冻 | held | 0.15 | 0.2, 1.0, 1.0 | 0 | n/a | 冻结 | committed, **single-branch render** |

**dual-clone 分支说明**：若 Type Vote 时 `confidence = max(p_type, 1-p_type) < 0.7`，则克隆 P1 state 两份；分别强制 `type_logit = -∞` (revolute) 和 `+∞` (prismatic)，各自跑 Main G1b 的 ~10% iter；最终比较两份的 final SDS+rgb loss，选低者 commit。这是 S3 的核心机制。compute cost ≈ +10% iter 但通常只在 boundary case 触发 (~20% 样本)。

### 10.2 Warmup G- (0–5%) 跳过 SS-DiT + 冻 α_m 的好处

- 节约 ~5% iter × 一次 SS-DiT forward 的开销
- adapter 还没有有效梯度，跑 SS-DiT 是浪费
- α_m 在闭态 first-frame 渲染下无监督信号，让它先冻在 BMCSA 初始值
- 0–5% 只训：`Δz_s` (decoder path) + `α_g`，BMCSA prior 周围的 presence gate 先稳定

### 10.3 τ_sds 采样：inverse-CDF of logit-normal(1, 1)（★ v3.3.1 C1 修）

旧 v3.3 phase-based mixture 在 [0.6, 0.98] 高噪段聚集，违反 CHORD W-RFSDS 的 `σ ~ ŵ(σ)` 要求。改为按 TRELLIS SS-DiT 训练 schedule `logitNormal(mean=1.0, std=1.0)` 的归一化权重反 CDF 采样，全程使用同一个 sampler：

```python
import math
from scipy import stats

def sample_tau_inverse_cdf_logit_normal(
    iter_idx, total_iters, mean=1.0, std=1.0, jitter=0.02,
):
    """
    σ_i = inv_CDF_{logitNormal}(quantile_i),  quantile_i = 1 - (i+1)/(I+1)
    
    iter=0 → quantile≈1 → σ≈sigmoid(mean + std·Phi^-1(1)) ≈ 0.95 (重 layout)
    iter=I → quantile≈0 → σ≈sigmoid(mean + std·Phi^-1(0)) ≈ 0.05 (重细节)
    
    全过程单调下降, 自然 coarse-to-fine; jitter 加小量随机避免完全 deterministic.
    """
    quantile = 1.0 - (iter_idx + 1) / (total_iters + 1)
    if jitter > 0:
        quantile = max(min(quantile + (torch.rand(1).item() - 0.5) * jitter, 1 - 1e-3), 1e-3)
    z = stats.norm.ppf(quantile)                  # Φ^{-1}
    tau = 1.0 / (1.0 + math.exp(-(mean + std * z)))
    return float(tau)
```

**与 dual-expert 的关系**：logitNormal(1,1) 下 `P(τ ≥ 0.9) ≈ 14%`，自然触发 Wan high_noise_expert；不再需要 phase-mix 强制 30%。

**为什么删 phase-mix**：
1. CHORD §3.2 Eq.(3) 显式要求 `σ ~ ŵ(σ)`，否则 SDS gradient 期望偏；
2. CHORD §4.3 Figure 8 实证 uniform 失败、inverse-CDF 才稳；
3. phase-mix 在 [0.6, 0.98] 聚集，**完全缺失 logit-normal 的 mode 附近 (≈0.73) 的密集采样**，把训练 schedule 的核心区域空了；
4. inverse-CDF 给出自然的 coarse-to-fine 退火，不需要手动按训练阶段切 τ 范围。

---

## 11. 纹理阶段（P2）— **v3.3.1 S4 修：tanh-reparameterized canonical SLAT 残差**

### 11.1 核心机制

- 唯一可学参数：`delta_z = nn.Parameter(torch.zeros_like(z_slat_init))`，shape `[N_obj, 8]`
- 实际 forward 用：`z_slat = z_slat_init + 3.0 * slat_std.view(1, -1) * torch.tanh(delta_z)`
- D_GS 每 iter 解码一次 canonical Gaussians（**不冻结输出**，让所有通道梯度都能回流到 delta_z）
- 每个 state k 的 SDS / RGB 损失通过 `rasterize.backward → D_GS.backward → tanh.backward → delta_z[i].grad` 自动回流
- **21 个 state 的梯度在 PyTorch autograd sum 阶段自动累加到同一份 delta_z**，无需手动同步

**为什么用 tanh reparameterization 替代直接 nn.Parameter(z_slat)** — 见 §4.3 详细说明。一句话：硬截断会破坏 AdamW momentum 状态 + box≠manifold；tanh 给 manifold-aware 软上界 + 全程可微 + AdamW 跑在 delta_z 上无破坏。

### 11.2 与导师反馈对应：为什么是 SLAT 而不是 per-Gaussian SH₀

| 设计 | 训练时 | 导出 mesh 时 | manifold 约束 |
|---|---|---|---|
| v3.1 `Δ_features_dc`（GS 输出端 residual） | render 看到 base color + 残差 | `D_Mesh(z_slat0)` 看不到这个残差 ⇒ **训练-导出不一致** | — |
| v3.2/v3.3 `z_slat = nn.Parameter` | `D_GS(z_slat)` render | `D_Mesh(z_slat)` 看到同一份 latent ⇒ **训练-导出一致** | 无硬上界, 仅 L_base_anchor 软正则; 可能 drift off-manifold |
| **v3.3.1 (S4)** `z_slat = z_init + 3·std·tanh(delta_z)` | `D_GS(z_slat)` render | `D_Mesh(z_slat)` 用同一份公式 ⇒ **训练-导出一致** | **manifold-aware 3-σ 硬上界 + 全程可微** |

### 11.3 Move voxel 梯度回流（你 Q2 关心的）

例：canonical voxel A 索引 `i_A`（move），canonical 位置 p_can，state-1 warp 后 p_s1。

- forward: `gauss_can = d_gs(SparseTensor(z_slat, U))[0]`；`means_move[i_A] = T_1 @ gauss_can.get_xyz[i_A]`
- backward state 1: `∂L_s1 / ∂z_slat[i_A]` 通过两条路径汇聚：
  - **SH₀ 颜色路径**：`L_s1 → rgb_s1 → rasterize → sh[i_A] → D_GS.backward → z_slat[i_A]`（不过 T_k，直接）
  - **位置/scale/rotation 路径**：`L_s1 → rgb_s1 → rasterize → means_move[i_A] → T_1 @ canonical_xyz[i_A] → D_GS.backward → z_slat[i_A]`（通过 T_1 chain rule，T_1 在 P2 frozen 但 chain rule 仍执行）
- 21 state 的梯度 PyTorch 自动 sum

### 11.4 P1 → P2 固化的变量（★ v3.3.1 S3 修：type 必须已 committed）

P2 入口约定：**type 已经在 P1 末 Type Vote (50-55%) + Main G1b (55-65%) 中 commit**，传给 P2 一个 boolean `type_hard`。不再有"P2 中再决定 type"的代码路径，不再有 `type_uncertain / get_p2_render_mode / two_branch_soft / single_branch_hard` 分支。

P2 init 必须 assert `type_hard` 已 committed：

```python
# P1 学完后固化（detach 后存 bootstrap）
g_soft_p1     = torch.sigmoid(r_p1).detach()                # [N_obj]
m_soft_p1     = torch.sigmoid(b_p1).detach()                # [N_obj]
ψ_pred_p1     = bootstrap.ψ_pred_final                       # frozen joint
phi_render_p1 = bootstrap.phi_render_final                  # [21] frozen

# ★ v3.3.1 S3: type 已 committed (在 P1 末 type vote + G1b 中决定)
# 来源:
#   (a) confidence ≥ 0.7 → 直接 commit 由 sigmoid(ψ_pred_p1.type_logit) > 0.5 决定
#   (b) confidence < 0.7 → dual-clone 跑剩余 ~10% iter, 选 final SDS+rgb loss 低者
# 进 P2 时 type_hard 已是 bool 不再变.
assert hasattr(bootstrap, 'type_hard'), "type must be committed before P2 entry"
type_hard = bootstrap.type_hard                              # bool: True=prismatic, False=revolute

# P2 hard threshold gates (P1 学完后 g, m 应已接近 binary)
g_per_voxel = (g_soft_p1 > 0.5).float()
m_per_voxel = (m_soft_p1 > 0.5).float()

# Per Gaussian
g_per_gauss = g_per_voxel.repeat_interleave(32)
m_per_gauss = m_per_voxel.repeat_interleave(32)
```

**为什么删 `get_p2_render_mode` 二分**：见 S3 critique — P2 几何全冻，soft blend 两条 branch 不能让 type 变对（type 已被 ψ_pred_p1 冻结），反而让纹理朝两条不兼容的轨迹同时妥协，后 80% 切 single-branch 时若 type_hard 选错则纹理已歪。正确做法是在 P1 末就用 deterministic vote + dual-clone 比 loss 选定 type。

**type vote + dual-clone 实现接口**（在 P1 末调用，写入 bootstrap.type_hard）：

```python
@torch.no_grad()
def p1_type_vote(p1_state, learnable, bootstrap, n_samples=8):
    """P1 进度 ~85% 时调用. n_samples 个 (t_ss, seed) 平均 type_logit."""
    t_list = [0.30, 0.35, 0.40, 0.45]
    seed_list = [42, 1337]
    logit_acc = 0.0
    for t in t_list:
        for s in seed_list:
            torch.manual_seed(s)
            ε = torch.randn_like(bootstrap.z_s0)
            z_t = q_sample(p1_state.z_s_base, t, ε)
            _, captured = ss_dit_w.forward_capture(z_t, t, bootstrap.trellis_cond_can)
            feat, _ = sample_hidden_at_U(captured, bootstrap.U_object, ...)
            m_soft = torch.sigmoid(learnable.α_m + ... * learnable.H_part(feat).squeeze(-1))
            weights = m_soft / (m_soft.sum() + 1e-6)
            F_pool = (weights.unsqueeze(-1) * feat).sum(dim=0)
            ψ_logit = (learnable.ψ_param + ... * learnable.H_joint(F_pool)).type_logit
            logit_acc += ψ_logit.item()
    logit_avg = logit_acc / (len(t_list) * len(seed_list))
    p_type_avg = 1.0 / (1.0 + math.exp(-logit_avg))
    confidence = max(p_type_avg, 1 - p_type_avg)
    return p_type_avg, confidence

def commit_type(p_type_avg, confidence, p1_state, learnable_cur):
    """
    confidence ≥ 0.7 → 直接 commit
    confidence < 0.7 → 克隆两份分别强制 type, 跑剩余 ~10% iter, 比较 final loss 选低者
    """
    if confidence >= 0.7:
        bootstrap.type_hard = bool(p_type_avg > 0.5)
        return 'direct_commit', None
    else:
        # dual-clone 模式
        state_rev = deepcopy(p1_state)
        state_pri = deepcopy(p1_state)
        # 强制 type_logit 极端值, 让 ψ_pred 输出 hard 选定 type
        state_rev.learnable.force_type_logit = float('-inf')   # revolute
        state_pri.learnable.force_type_logit = float('+inf')   # prismatic
        # 各跑 ~10% iter (Main G1b)
        loss_rev = run_p1_g1b(state_rev, n_iter=int(0.10 * total_iters))
        loss_pri = run_p1_g1b(state_pri, n_iter=int(0.10 * total_iters))
        if loss_rev <= loss_pri:
            bootstrap.type_hard = False     # revolute
            return 'dual_clone_committed_revolute', state_rev
        else:
            bootstrap.type_hard = True      # prismatic
            return 'dual_clone_committed_prismatic', state_pri
```

### 11.5 Inner loop（★ v3.3.1 S3+S4 修：单 branch + tanh reparam）

```python
# ★ v3.3.1 S4: delta_z 是 nn.Parameter, z_slat 由 tanh reparam 派生
z_slat_init   = bootstrap.z_slat0.clone().detach()               # [N_obj, 8] frozen reference
slat_std_safe = bootstrap.slat_std.clamp_min(1e-3).to(device)    # [8]
delta_z       = nn.Parameter(torch.zeros_like(z_slat_init))      # [N_obj, 8], delta_z=0 → z_slat=z_init

optimizer = torch.optim.AdamW([delta_z], lr=1e-3, weight_decay=0)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=N_p2, eta_min=1e-5)

# ★ v3.3.1 S3: type 已 committed, 单 branch render
T_list = [SE3_rollout(ψ_pred_p1, phi_render_p1[k], type=type_hard) for k in range(21)]
# T_list 在整个 P2 期间不变 (几何全冻), 可 cache 而不是每 iter 算

for it in range(N_p2):
    f_global = 0.75 + 0.25 * (it / N_p2)
    
    # ----- ★ S4: reparameterize z_slat (manifold-bounded) -----
    z_slat = z_slat_init + 3.0 * slat_std_safe.view(1, -1) * torch.tanh(delta_z)
    
    # ----- forward: D_GS once, NOT detach geometry channels -----
    sparse_in = SparseTensor(z_slat, bootstrap.U_object_with_batch)
    gauss_can = d_gs(sparse_in)[0]
    # 所有 channel 都参与 backward (xyz / scale / rotation / opacity / sh)
    
    # ----- 21-frame render with frozen joint, single branch (★ S3 简化) -----
    rgb_T3HW = render_21_with_warp(
        gauss_can, T_list, g_per_gauss, m_per_gauss, cfg_warp=cfg.warp
    )                                                          # [21, 3, 480, 832] in [0, 1]
    rgb_3THW = rgb_T3HW.permute(1, 0, 2, 3)                    # for Wan VAE
    
    # ----- losses -----
    τ_low = sample_tau_inverse_cdf_logit_normal(it, N_p2, mean=0.0, std=1.0)
    # 注: P2 用 mean=0 (mode≈0.5) 比 P1 的 mean=1 更聚焦低-中噪段 (CHORD §A.1 texture stage)
    
    L_sds = W_RFSDS_Wan(
        rgb_3THW, bootstrap.wan_cond_cached, τ_low,
        cfg_scale=12.0,                                        # CHORD texture stage 实测
    )
    L_lat_rec = L_latent_rec_Wan(rgb_3THW, bootstrap.z_wan_target)
    
    wan_target_T3HW = bootstrap.wan_video_target_3FHW.permute(1, 0, 2, 3).float() / 255.0
    L_rgb_rec = F.l1_loss(rgb_T3HW, wan_target_T3HW) + lpips_loss(rgb_T3HW, wan_target_T3HW)
    
    s_0_norm = (bootstrap.s_0_with_carpet.float() / 255.0).unsqueeze(0)
    L_first = F.l1_loss(rgb_T3HW[0:1], s_0_norm) + lpips_loss(rgb_T3HW[0:1], s_0_norm)
    
    # ----- ★ S4: anchor 改在 delta_z 上 (tanh 已限上界, 软正则鼓励 base 接近 0) -----
    # 注: tanh 已硬约束 |z_slat - z_init| ≤ 3·std, 所以 anchor 权重可降; 
    # 仍保留 confidence-aware 三段以鼓励 base voxel 的 delta_z 接近 0
    base_conf = ((0.5 - m_soft_p1) / 0.5).clamp(0, 1)          # m=0 → 1, m=0.5 → 0
    move_conf = ((m_soft_p1 - 0.5) / 0.5).clamp(0, 1)          # m=1 → 1, m=0.5 → 0
    uncertain = (1 - base_conf - move_conf).clamp_min(0)        # m=0.5 → 1
    
    delta_z_sq_per_voxel = delta_z.pow(2).mean(dim=-1)          # [N_obj], 注: 直接在 delta_z 上算
    L_base_anchor      = (base_conf  * delta_z_sq_per_voxel).mean()
    L_move_smooth      = (move_conf  * delta_z_sq_per_voxel).mean()
    L_uncertain_anchor = (uncertain  * delta_z_sq_per_voxel).mean()
    
    # ----- total (anchor 权重比 v3.3 降低: tanh 已提供硬上界) -----
    loss = (
        0.2 * L_sds
      + 1.0 * L_lat_rec
      + 1.0 * L_rgb_rec
      + cfg.λ_first * L_first
      + 3.0  * L_base_anchor                                    # ★ v3.3.1: 10.0 → 3.0
      + 0.05 * L_move_smooth                                    # ★ v3.3.1: 0.1 → 0.05
      + 0.3  * L_uncertain_anchor                               # ★ v3.3.1: 1.0 → 0.3
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
```

**S3+S4 修正对 P2 的影响**：
1. 删除 `type_uncertain / render_mode / two_branch_soft` 分支 → 只有 single branch render；
2. `T_list` 在整个 P2 不变，可 cache（21×SE(3) 矩阵），节省每 iter 的 rollout 时间；
3. 学的是 `delta_z` 不是 `z_slat`；z_slat 在 forward 时由 tanh reparam 派生；
4. anchor 权重显著降低（10.0→3.0、0.1→0.05、1.0→0.3）—— tanh 已提供 manifold-aware 硬上界，软正则只是次要 prior。

### 11.6 Decoded-geometry drift monitor（★ v3.3.1 S4 修：tanh 已硬约束 → 仅 logging）

P2 在 `delta_z` 上优化，`z_slat = z_init + 3·std·tanh(delta_z)` 已经把 z_slat 严格限制在 `z_init ± 3·std` 范围内（manifold-aware 3-σ 上界）。但 D_GS 解码后 Gaussian 几何属性的实际 drift 仍需监控，作为 sanity log 与 ablation 报告。

**与旧 v3.3 的差异**：旧设计 z_slat 是 nn.Parameter，drift 超阈值时**需要动态调高 anchor 权重**抢救；新设计 tanh 已提供硬上界，drift 不会爆炸，**monitor 只作 logging + 报告**，不再需要 emergency intervention。

```python
@torch.no_grad()
def compute_decoded_drift(delta_z_current, z_slat_init, slat_std_safe, U_object_with_batch):
    """
    每 100 iter 调一次. 报告 D_GS 解码后 Gaussian 几何属性的 drift.
    期望: xyz_drift < 1 voxel = 1/64 ≈ 0.016 world unit
         scale_drift < 0.01
         opacity_drift < 0.05
    """
    z_slat_now = z_slat_init + 3.0 * slat_std_safe.view(1, -1) * torch.tanh(delta_z_current)
    gauss_init = d_gs(SparseTensor(z_slat_init, U_object_with_batch))[0]
    gauss_now  = d_gs(SparseTensor(z_slat_now,  U_object_with_batch))[0]
    
    return {
        'xyz_drift_rmse':     (gauss_now.get_xyz - gauss_init.get_xyz).pow(2).mean().sqrt().item(),
        'scale_drift_rmse':   (gauss_now.get_scaling - gauss_init.get_scaling).pow(2).mean().sqrt().item(),
        'opacity_drift_rmse': (gauss_now.get_opacity - gauss_init.get_opacity).pow(2).mean().sqrt().item(),
        'rotation_drift_cos': F.cosine_similarity(gauss_now.get_rotation, gauss_init.get_rotation, dim=-1).mean().item(),
        'delta_z_l2':         delta_z_current.pow(2).mean().sqrt().item(),
        'tanh_saturation':    (delta_z_current.abs() > 2.0).float().mean().item(),
    }

# 在 P2 inner loop 末尾 (每 100 iter):
if it % 100 == 0:
    drift = compute_decoded_drift(delta_z, z_slat_init, slat_std_safe, bootstrap.U_object_with_batch)
    logger.info(f"P2 it={it} drift: xyz={drift['xyz_drift_rmse']:.4f}, "
                f"scale={drift['scale_drift_rmse']:.4f}, "
                f"opacity={drift['opacity_drift_rmse']:.4f}, "
                f"rot_cos={drift['rotation_drift_cos']:.4f}, "
                f"|delta_z|={drift['delta_z_l2']:.3f}, "
                f"tanh_sat={drift['tanh_saturation']:.2%}")
    
    # ★ S4 后: 仅 warning, 不 intervene (tanh 硬约束已托底)
    if drift['xyz_drift_rmse'] > 0.05:
        logger.warning(
            f"P2 it={it} xyz_drift={drift['xyz_drift_rmse']:.4f} > 0.05 (>3 voxel). "
            f"虽 tanh 限上界但解码端仍 drift 明显, 可能 SLAT 局部 manifold 不光滑. "
            f"参考 ablation: 调小 tanh scale (3.0 → 2.0) 或减 lr (1e-3 → 5e-4)."
        )
    if drift['tanh_saturation'] > 0.30:
        logger.warning(
            f"P2 it={it} tanh_saturation={drift['tanh_saturation']:.2%} > 30%. "
            f"大量 voxel 的 delta_z 已饱和到 ±2 附近, 说明 3σ 上界可能太紧; "
            f"参考 ablation: tanh scale 3.0 → 4.0 放宽."
        )
```

**ablation：tanh scale (3.0 vs 2.0 vs 4.0)**：
- 3.0 (默认)：99.7% manifold 支撑，对齐 SLAT VAE training distribution
- 2.0：95% 支撑，更保守；可能限制纹理学习
- 4.0：99.99% 支撑，几乎不约束；可能漂出 manifold 致 D_GS 解码退化

AAAI reviewer 若质疑"P2 优化 SLAT 会不会破坏 geometry"，drift monitor + tanh 硬上界 + ablation 三件套是直接回应。

### 11.8 Supervision Provenance map（v3.3 改名 + 收窄 claim）

**关键改动**：v3.2 的 `texture_provenance` 命名暗示纹理来源 (donor source)，但我们没做 donor projection，做不到精确 source 判定。v3.3 改名 `supervision_provenance`，类别只描述 **"该 Gaussian 在多少 state 中可见，因此 supervision 信号来自哪里"**，**不**声称颜色来自某 frame。

```python
@torch.no_grad()
def compute_supervision_provenance(bootstrap, z_slat_final, ψ_pred_p1, phi_render_p1):
    """
    Per-Gaussian classification of which rendered states the Gaussian was visible in.
    NOT a texture source/donor claim — just visibility-based supervision provenance.
    """
    sparse_in = SparseTensor(z_slat_final, bootstrap.U_object_with_batch)
    gauss_can = d_gs(sparse_in)[0]
    N_gauss = gauss_can._xyz.shape[0]
    
    visibility = torch.zeros(N_gauss, 21, dtype=torch.bool)
    for k in range(21):
        T_k = SE3_rollout(ψ_pred_p1, phi_render_p1[k], type_hard)
        visibility[:, k] = compute_visibility_via_rasterize_aux(
            gauss_can, T_k, m_per_gauss, camera_locked
        )
    
    parent_idx = torch.arange(len(bootstrap.U_object)).repeat_interleave(32)
    is_base_gauss = (m_per_voxel[parent_idx] < 0.5)
    
    provenance = torch.empty(N_gauss, dtype=torch.long)
    for i in range(N_gauss):
        if is_base_gauss[i] and visibility[i].all():
            provenance[i] = 0    # visible_in_all_states
        elif visibility[i].any():
            provenance[i] = 1    # visible_in_open_states
        else:
            provenance[i] = 2    # never_visible
    
    return provenance
```

`supervision_provenance.json` 报告：

```json
{
    "visible_in_all_states_ratio": 0.45,
    "visible_in_open_states_ratio": 0.32,
    "never_visible_ratio": 0.23,
    "note": "These ratios describe supervision visibility across rendered states, NOT donor texture source. Texels labeled 'never_visible' have no direct image supervision and rely on the Wan2.2 video prior; quality is bounded by the prior's hallucination accuracy."
}
```

**论文表述（v3.3 收窄）**：
> "We report supervision provenance per Gaussian, indicating in how many rendered states each Gaussian was visible to the camera. Texels labeled `never_visible` (deep interior surfaces never exposed in any rendered state) have no direct image supervision and rely solely on the Wan2.2 video prior. We do not claim donor-based texture reconstruction; the Wan prior's hallucination quality bounds this category."

**不写**：
- ~~"texture source from frame X"~~（没做 projection）
- ~~"donor fusion"~~（没做 visibility / view-angle / depth 加权）
- ~~"hidden surface recovered from open-state observation"~~（recovered 太强，实际是 W-RFSDS prior）

### 11.9 Optimizer 设置

```python
# ★ v3.3.1 S4: optimizer 跑在 delta_z 上 (不在 z_slat 上)
optimizer = torch.optim.AdamW([delta_z], lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

# Cosine decay over P2
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_p2, eta_min=1e-5
)
```

无需 lr 分组（只有 delta_z 一个 Parameter）。weight_decay=0 因为：(a) tanh reparameterization 已提供 manifold-aware 硬上界；(b) L_base_anchor 软正则鼓励 base voxel delta_z 接近 0。

---

## 12. 导出（**v3 新加 deterministic gate protocol**）

### 12.1 Deterministic gate logits

```python
@torch.no_grad()
def deterministic_gate_export(learnable, bootstrap, n_samples=8):
    """
    训练时 r, b 依赖 t_ss / ε / adapter random，不可复现导出.
    Export 时用固定 t 和 fixed seed eps 多样本平均.
    """
    fixed_t_list = [0.25, 0.30, 0.35, 0.45]
    fixed_seeds  = [42, 1337]
    
    r_acc = torch.zeros(N_obj, device=device)
    b_acc = torch.zeros(N_obj, device=device)
    n_total = 0
    
    for t_ss in fixed_t_list:
        for seed in fixed_seeds:
            torch.manual_seed(seed)
            ε = torch.randn_like(bootstrap.z_s0)
            
            z_s_base = bootstrap.z_s0 + learnable.Δz_s
            σ_min = 1e-5
            z_t = (1 - t_ss) * z_s_base + (σ_min + (1 - σ_min) * t_ss) * ε
            
            pred_v, captured = ss_dit_w.forward_capture(
                z_t, t_ss, bootstrap.trellis_cond_can
            )
            occ_logits = ss_vae_decoder(z_s_base)
            feat, occ_at_U = sample_hidden_at_U(
                captured, bootstrap.U_object, occ_logits
            )
            
            r = occ_at_U.squeeze(-1) + learnable.α_g + learnable.λ_sup_final * learnable.H_sup(feat).squeeze(-1)
            b = learnable.α_m + learnable.λ_part_final * learnable.H_part(feat).squeeze(-1)
            
            r_acc += r
            b_acc += b
            n_total += 1
    
    r_final = r_acc / n_total
    b_final = b_acc / n_total
    
    g_hard = (torch.sigmoid(r_final) > 0.5)
    m_hard = (torch.sigmoid(b_final) > 0.5)
    
    return g_hard, m_hard
```

### 12.2 Mesh + atlas + URDF
（同 v2 §12.2-12.3，但 SparseTensor coords 必须用 [N,4] 含 batch 列）

---

## 13. 与 CHORD 的关系与差异
（同 v2 §13，但加注：v3 修了 W-RFSDS 调用 Wan API 的 6 个具体接口 bug）

---

## 14. 终极一句话（★ v3.3.1 更新）

**用 grad-enabled 的 one-step SS-DiT structural refiner（带 patchify + post-block adapter + final layer_norm + unpatchify）把 W-RFSDS 视频蒸馏的梯度（按 TRELLIS logit-normal(1,1) 训练 schedule 反 CDF 采样 τ，CFG 25→12 linear decay，Wan I2V cond 严格按 VAE-latent + 4ch mask channel-concat + T5 text 构造，Wan timestep 范围 `[0, 1000)`，Wan VAE 输入归一化至 `[-1, 1]`），从渲染像素端（P1 21 帧双 branch soft blend → P1 末 deterministic type vote + 必要时 dual-clone 选 loss 低者 commit；P2 21 帧单 branch render），经可微 Gaussian 渲染（base + warped move 双贡献，move Gaussian 的位置和 quaternion 同步旋转）、解析 SE(3) rollout（5 个学习 φ 累积到 21 帧）、D_GS 输出端的 BinaryConcrete opacity gate（不改 `_opacity`，乘 `get_opacity`）、SS-VAE decoder 的 dense occupancy logit 桥、trilinear+PE hidden 坐标映射，端到端地回到 SS-DiT block 14/16/18 之后新增的 zero-init residual adapter；在冻结 TRELLIS 主干的前提下，在固定的 canonical support `U_object`（Stage B 12 步严格顺序构造 + Stage C.5 每 1000 iter silhouette consistency check，IoU < 0.85 触发一次性 expand + SLAT 重采）上学习 base/move 分割、single-DoF joint 参数；carpet 全程含在 video / render / target，仅 Export 时按 is_carpet_mask 剔除；纹理阶段冻结所有几何，唯一可学参数 `delta_z` 通过 tanh reparameterization 派生 `z_slat = z_init + 3·std·tanh(delta_z)` 给 D_GS 和 D_Mesh 共享读取（manifold-aware 3-σ 硬上界 + 全程可微）；导出阶段用 deterministic gate protocol（固定 t/seed 多样本平均）确保可复现。**
