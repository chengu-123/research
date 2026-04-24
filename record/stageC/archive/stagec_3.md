# 以 state-0 为锚的 screw-manifold EM：一份可落地的 Stage C 设计

**核心判决先行**。Stage C 的最优架构是一条四节流水线：(i) 跨状态 voxel count 分布直接给出 `base / boundary / move` 三类硬先验；(ii) state 5 冷启动用 moment-based 粗估 + Chasles 分解投到 5-DoF screw manifold，再做 IRLS 精化；(iii) EM 以 **anchor-then-relax** 三阶段调度展开——先在 state 5 收敛 T₅，再沿 k=4…0 逆序传播，最后全局松弛；(iv) swept volume 用 canonical_move 扫，前 N−1 轮只做**警戒**（软 penalty），最后一轮才 **late-commit carving**。整条链路拒绝 loss soup、拒绝 per-voxel matching、拒绝 Theil-Sen 后处理——它们在本新 formulation 下均冗余。canonical_move 通过 residual-weighted soft vote + 体积守恒阈值构造；**M_attn 退出主信号链**，仅作对称件剪枝的 tiebreaker；**SLAT/SS-latent 亦不入主链**，只作可选 ablation。这份设计在三点上与既有工作划清边界：K>2 的硬 count-indexed part partition、从最大打开态冷启动的非对称 EM、canonical-part swept volume 作为末轮 commit。

Stage B 产出的 6 个 state 已通过真实数据验证：count=6 的 4699 voxel 是几何基线近乎完美一致的 base，count=1 的 3301 voxel 是各状态独占的纯 move 证据，IoU 相邻态 0.75–0.84。这种"干净的票型"是整套 formulation 成立的数据前提，也是为什么下游算法能拒绝过参数化。

## Stage B 的 count 分布作为贝叶斯先验

把跨 6 state 的 voxel count c(v) ∈ {0,…,6} 视为最强的**组合式先验**，是整份设计的起点。c(v)=6 的 voxel 对应\"6 个 state 都判定占据\"；在 TRELLIS ~5% 量级的表面噪声下，这等价于 base 的 MAP 估计。c(v)=1 对应\"仅一个 state 独占\"，在 1-DOF 运动下这只能是那个 state 的 move 所占据的一个轨迹点。c(v)∈{2,…,5} 是 **partial-sweep 带**：base-move 接触层、TRELLIS 在不同 state 下对同一 move voxel 的略微飘移、以及真 base 的轻微重采样误差都落在这里。

真实数据给出的参数默认值从此处直接锁死。**base 阈值取 c(v)=6**（4699 voxel，干净支持集）；**纯 move 种子取 c(v)≤1**（3301 voxel，高置信度运动证据）；**boundary uncertain 区间取 2≤c(v)≤5**（共 603 voxel，约占总 move 类的 15%）。这个 15% 是整份设计对 downstream carving 容忍度的主要驱动。

一个与既有直觉反向的事实需要被正面处理：**closed state 体素数（5862）比 open state（5401）多约 8.5%**，与"打开后表面更多 → voxel 更多"的朴素假设相反。最可信的机制解释是 TRELLIS 的 sparse structure VAE 在接触面处对 feature 不一致性做了 soft over-prediction——closed 态基-move 接触层同时接收\"这是 base 顶面\"和\"这是 move 底面\"两种 DINOv2 feature 信号，VAE 的 Dice loss 对正例 false positive 惩罚较弱，于是激活了额外一层 voxel。这种 bias 对设计**有利**而非有害：它意味着 state 0 下接触带被保留得比真实几何更厚，这些 voxel 在跨状态投票里会恰好落在 c∈{2,…,5} 的 boundary 区段，正好被主链路当作\"不确定、交给 swept volume 后处理\"的对象。

## Canonical 空间的精确定义与两条证据链

**Canonical 空间严格等于 state 0 的 occupied 空间**，即 Ω_c = {v : O₀(v)=1}。这条约束贯穿整个设计——canonical_base ⊆ Ω_c，canonical_move 虽然通过跨状态 warp 聚合，但**最终裁剪回 Ω_c**。没有任何一个 voxel 可以\"从其他 state 补进来\"。这种 asymmetric canonical 选择的精神与 Articulate-your-NeRF 冻结 state-0 NeRF、ArticulatedGS 以 start state 为锚的路线一致，但它们都在 neural field / Gaussian 空间实现，本设计**在离散 voxel 空间首次 instantiate**。

State 0 下 base 与 move 的划分不来自 state 0 自身——而是由**其他 5 个 state 的证据反推**。具体分两条证据链：(a) 跨状态一致性链，count(v)=6 的 voxel 以极高置信度归 base；(b) canonical 重投影链，一旦 {T_k} 估出，canonical_move 被 warp 回 state 0 得到 M_0_reconstructed，其与 O_0 的交集是 state 0 下 move 的几何证据。两条链在 boundary 带（c∈{2,…,5}）上可能冲突，由 swept volume 的末轮 carving 做最终仲裁。

这是整份设计唯一需要\"两路证据综合\"的环节，严格避免了 per-voxel 逐点 matching loss 带来的优化噩梦。

## 从 state 5 冷启动的 move 初分割

State 5 作为冷启动点的理由来自信息论：它是最打开态，move 与 base 在物理空间里分离最远，遮挡最小，几何分辨率最清晰。具体分割算法采用**分层 multi-cue 融合**而非单一信号。

第一层是**强种子** S_hard = {v ∈ O₅ : c(v) ≤ 2}，这些 voxel 只在 state 5（及最多再一个邻近开态）出现，几乎必然属于 move。第二层是**弱约束** S_soft = {v ∈ O₅ : c(v) ≤ 5}，排除所有 state 共占的 base。第三层是形态学+最大连通分量：move_5_init = LargestCC(MorphClose(S_hard ∪ S_soft))。M_attn 仅在 S_soft\S_hard 区间做 tiebreaker，**不进入 S_hard 的构造**——这是对\"长抽屉 trajectory 中段 attention 污染\"这一已知 failure 的直接防御。

这套冷启动的关键失败模式是**极短冲程 move**：若 state 5 的位移不足 2–3 voxel，则 S_hard 为空，整个分割退化。检测器是 |S_hard|/|O_5| < 0.05，触发时算法切换到"选位移最大的 state 作为 anchor"的自适应模式，而非固定 state 5。真实数据里 state 0 独占 voxel 数（841）已经比 state 5（518）还多，暗示这个自适应 anchor 选择在某些物体类上确实可能触发——但本数据集的 state 5 仍有 518 个独占 voxel，远超阈值，默认路径安全。

## 初始 T₅ 估计与 1-DOF 约束的 screw manifold 搜索

初始 T₅ 是整条 EM 的种子，其质量决定了后续传播的成败。主方法是三步 closed-form 加一步精化。**Step 1** 计算两端 centroid μ_0、μ_5 与 inertia tensor Σ_0、Σ_5，通过特征向量对齐得到初始旋转 R₀（手性修正 det=+1）与平移 t₀。**Step 2** 做对称性检测：若 Σ_5 的 top-2 特征值比 λ₂/λ₁ > 0.95，标记为绕长轴对称，保留 N=8 个均匀分布于对称轴上的候选假设，推迟到后续 state 打破。**Step 3** 用 Chasles 分解把 (R₀, t₀) 投影到 5-DoF screw 参数 (ω, q, θ)——旋转轴方向 ω ∈ S²、轴上 pivot q ∈ R³（沿 ω 方向的自由度通过取 q 到 move 质心最近点消去）、旋转角 θ。**Step 4** 在 screw manifold 上跑 5 维 IRLS：argmin_{ω,q,θ} Σ_v move₅_init(v) · SDF(O_0, T(ω,q,θ)·v)²。

**关键论点**：直接在 screw manifold 上搜索比通用 SE(3) ICP 鲁棒得多，这已被 FreeArt3D 的 ablation（移除 joint estimation 导致 77% → 崩溃）、PARIS 的两阶段 config（se3.yaml warmup → revolute.yaml refine）和 Artykov ICCVW 2025 的 RANSAC screw sampling 同时印证。1-DOF 的物理约束把搜索空间从 6 维压到 5 维（revolute）或 4 维（prismatic），基本消除了 symmetric object 的 cost-surface plateau 问题——**只要 θ_5 足够大**。这就是从最大打开态冷启动的第二个硬理由：θ 越大，screw 轴的可识别性越强；当 θ → 0（比如 state 1）screw 轴数值上退化，那些 state 不能作为冷启动点。

Revolute/prismatic 二选一仲裁在 IRLS 后做：分别拟合两种模型，用 residual ratio（两者 RMSE 之比）超过 1.5 判定主导类型；否则保留两个假设进入 Phase 2。

## 跨状态 majority voting 构造 canonical_move

给定 {T_k}，每个 state 的候选 move 是 M_k_cand = O_k \ base_mask_k（base_mask 由 c(v)=6 和 T_k 反演共同给出）。warp 回 canonical 空间得到 W_k(v) = trilinear(M_k_cand, T_k⁻¹·v)。聚合为 canonical_move 的精确公式：

**canonical_move(v) = 1 ⟺ [ Σ_k w_k · W_k(v) / Σ_k w_k ] > τ\***，其中 w_k = exp(−β · r_k²)，r_k 是 state k 的 screw-manifold 拟合残差 RMSE；阈值 τ\* 由**体积守恒**自适应选取：τ\* = argmin_τ ||{v : score(v)>τ}| − median_k |W_k|_binarized|。

选 weighted soft vote 而非 hard majority vote 的理由是 TRELLIS 明显违反 multi-atlas 文献中的\"同分布 rater\"假设——closed state 系统性致密，open state 系统性稀疏。残差加权能自动压低拟合差的 state 的话语权。但在 **EM 前 1–2 轮**，T_k 本身尚未收敛，残差估计不可靠，此时退回 hard MV（≥⌈K/2⌉=3 票）+ morphological closing 作为稳定器。这种 hard→soft 的切换与 ArtGS 的 joint prediction warmup 哲学一致：early-stage 用 rigid 规则、late-stage 解除。

体积守恒阈值的理由是 ROC / Youden 需要 GT 而你没有；Otsu 需要双峰而 score 分布单峰；固定 0.5 在加权投票下不保 Bayesian 最优。median_k |W_k| 给出的是一个 data-driven 的体积 oracle。真实数据里各 state move 独占 voxel 数 median ≈ 540（541 为 {841,563,612,434,333,518} 的中位数），这就是 canonical_move 最终体积的目标 anchor。

最终 canonical_move **被裁剪回 Ω_c**：canonical_move_final = canonical_move_raw ∩ O_0。这条裁剪是整份设计的架构性硬约束，确保没有 voxel 从其他 state 泄漏进 canonical。

## EM 外循环的三阶段 anchor-then-relax 调度

**Phase 1（anchor，state 5 独立收敛）**：在 state 5 上固定其他 T_k 不存在（它们还没被估计），只跑 T₅ ↔ canonical_move 的 EM 内循环 5–10 轮。每轮 E-step 把 canonical_move warp 到 state 5 坐标系得到预测 mask，M-step 重估 T₅ 使之与 O₅ 的 overlap 最大。symmetric 假设并行追踪。此 phase 输出是一个高置信度 anchor T₅。

**Phase 2（sequential propagation，逆序 k=4→0）**：每个 k 的 warm-start 用沿 screw 轴按 θ 比例插值 T_k⁽⁰⁾ = (ω, q, θ₅ · k/5)；对 prismatic 则为 d_k = d_5 · k/5。然后固定 T_{k+1…5} 不动，只优化 T_k 在 screw manifold 上的 fit。每进一个 state，对 symmetric hypotheses 做 likelihood ratio 剪枝——非对称件通常在 state 4 或 3 就打破简并，真对称件可能一路保留到 state 0，此时最终报告"轴方向已定、绕轴角度无法辨识"而非强行选一个。canonical_move 在每一步后用 soft vote 更新。

**Phase 3（global relax，全松弛）**：所有 T_0…T_5 同时优化，目标 L = Σ_k BCE(trilinear(canonical_move, T_k⁻¹·v), Ô_k(v)) + λ·Σ_k max(0, θ_k − θ_{k+1})²。后一项强制 θ_k 沿 k 单调（closed→open），是物理合理性的最低约束，λ 取 0.1 即可。此 phase 也是 carving 真正发生的地方——前两 phase 只做警戒。

这三阶段调度针对 PARIS（PAOLI 2025 明确批评"unstable when initialization is noisy"）、纯 sequential（早期 state 错误不可逆）、ArtGS 的 warmup（只在单一 state 暖身）三种失败模式各打一针。Phase 1 保证冷启动种子够硬，Phase 2 防止早期错误传播，Phase 3 消除 sequential bias。

## Swept volume 的精确触发条件与对象

Swept volume 由 **canonical_move（不是 M_k）** 沿 {T_k} 扫出：S_move = ⋃_k T_k(canonical_move)。选 canonical_move 作扫体的根本理由是它是所有 state 投票后的\"共识部件\"——用它扫消除了 per-state TRELLIS 重建噪声向 carving 结果的泄漏。这与 M_k 各自扫再 union 的方案相比，对 T_k 微小误差的敏感度显著更低，因为误差被 majority vote 稀释过一次，不会在每个 state 上独立放大。

**触发时机严格二分**。EM Phase 1 与 Phase 2 全程**不做 carving**，只做警戒：L_carve = Σ_v base(v) · S_move(v) 作为软 penalty 加入 total loss，系数极小（~0.01），仅为优化轨迹提供\"互穿不好\"的梯度信号。Phase 3 的最后一轮（等价于 EM 外循环收敛后）**才执行硬 carving**：base_final = O_0 ⊙ (1 − S_move)，并强制下界 |base_final| ≥ α · |O_0|，α=0.3。下界保护是对"revolute 角度过大 swept 扫过绝大部分 O_0"这一 failure 的最后防御。

Theil-Sen 尺寸修正在新 formulation 下**确实冗余**——majority vote 的体积守恒阈值已经自然剔除了各 state 过大的尾部，不存在需要\"统计斜率\"修正的 linear bias。真正需要警惕的是 isotropic shrinkage，若末轮 swept 后 canonical_move 体积比 median_k |M_k| 缩减超过 30%，触发 spatial deconvolution（Wang TPAMI 2013）一次性修正，而不是 Theil-Sen。

非凸运动（如翻盖 180°）的 swept self-intersection 用 max 而非 sum 聚合：S_move(v) = max_k T_k(canonical_move)(v)，防止重复累计导致 penalty 量纲失衡。

## M_attn 的最终使用边界

主信号链**不依赖 M_attn**。用户已提醒长抽屉 trajectory 中段会出现 attention 污染，而文献里（TRELLIS 作为 backbone 的工作如 FreeArt3D）也确认 DiT cross-attention 在 diffuse 物体上定位能力有限。M_attn 在本设计中只保留三处极有限的使用。

第一处是 **state 5 冷启动的第三层仲裁**：在 S_soft\S_hard 的 uncertain 区间里，用 M_attn > 0.5 做 tiebreaker。这里若 M_attn 污染，最多就是多收或少收几个 boundary voxel，不影响 S_hard 种子和最终 largest CC。第二处是 **symmetric hypotheses 的弱先验**：如果 M_attn 在对称轴某个方向有明显 bias，可用它给候选假设打一个 soft prior 分，但这个 bias 不能 dominate 几何残差。第三处是 **监督下的 sanity check**：如果最终 canonical_move 与 upsampled M_attn 的 Dice 低于 0.3，打印 warning 但不改变输出——这是给论文 reproducibility 用的诊断，不是 loss。

DiT 内部更丰富的信号（per-layer、per-head、per-timestep cross-attention、self-attention affinity）在原则上可用，且 diffusion-based segmentation 文献有 head-level probing 的标准做法。但它们都属于\"可做但未被验证对本 pipeline 有增益\"的范畴，**明确推迟到 ablation 章节**，不进入主链路。

## SLAT / SS-latent 是否进入主链

**不进入**。这里有一个必须在论文中澄清的事实：用户手里的 z_final 形状 (K=6, 8, 16, 16, 16) 极有可能是 TRELLIS 的 **sparse-structure VAE latent（SS-latent）**，不是 SLAT 本身。真正的 SLAT 是稀疏 list {(z_i, p_i)}，每个 active voxel 附带 8 维 latent，典型 L≈20K。用户的 dense (8,16³) tensor 与 TRELLIS config 文件 `ss_enc_conv3d_16l8_fp16_64`（16³ 空间 × 8 channel）完全对应，是 SS-VAE 的产物，只编码 coarse occupancy 的 latent 压缩，**不携带 appearance / semantic 通道分解信息**。论文如果要引用 z_final 做任何 part-level 论证，必须先 double-check 这一点，否则会被审稿人立刻识破。

即使用真正的 SLAT，它的 8 维 latent 是 KL-regularized 的 entangled bottleneck，**没有任何 per-channel 语义设计**；训练目标全部是重建 + KL + Dice + rectified flow CFM，没有 part supervision。文献里 **PartField**（ICCV 2025）和 **SegviGen**（arXiv 2603.16869）都绕开了\"直接 clustering SLAT\"路径，前者训练专门的 feature field，后者 fine-tune 整个 Tex-SLAT flow 预测 part color。这提供了强证据：**SLAT 不是一个现成可聚类的 part feature**。

本设计中 SLAT/SS-latent 的唯一出场是一次 ablation：K=2 KMeans on SLAT ⊕ 空间坐标，与主链路的跨状态 count-based 分割对比 Dice，预期可行性评分 3/10（纯 SS-latent 则 1/10）。这个 ablation 的作用是在论文里\"显式关闭这条路\"，回答审稿人的\"为什么不用 TRELLIS 内部特征\"问题。

## 主要 failure mode 与防御机制总览

| Failure mode | 触发条件 | 防御 |
|---|---|---|
| 极短冲程 move | \|S_hard\|/\|O₅\| < 0.05 | 切换 adaptive anchor（选 max-displacement state） |
| 对称件轴方向歧义 | λ₂/λ₁ > 0.95 on Σ₅ | 保留 N=8 screw hypotheses，Phase 2 剪枝；否则报告\"角度 aliased\" |
| T_k 早期错误 → canonical_move 污染 | EM 前 2 轮 | 强制 hard MV（≥3 票），关闭 soft weighting |
| Swept 扫过大多数 O₀ | \|base_final\| < 0.3·\|O₀\| | 下界保护 α=0.3，拒绝 commit |
| canonical_move 被 state 6 过度稀释 | residual 分布双峰 | 对高残差 state 权重清零，不是只降权 |
| M_attn 长抽屉污染 | M_attn 均值 > 0.5 且分布 flat | M_attn 完全退出，只用 count-based |
| TRELLIS closed-state over-prediction | c∈{2,…,5} 带异常厚 | 交给末轮 carving，不在 segmentation 阶段 commit |

这张表覆盖了从 Stage B 质量、对称件病态到 EM 优化失稳的所有已知路径，每一条都有可触发的定量阈值与自动 fallback。

## 与相关工作的 head-to-head 定位

与 **PARIS**（ICCV 2023）的核心区别在输入模态与 canonical 定义：PARIS 要多视角 RGB 双 state + 学习的双 NeRF canonical，本方法要单图 → Stage B voxel → 离散 state-0 anchor canonical。PAOLI 2025 明确批评 PARIS"unstable when init is noisy"，本方法的 anchor-then-relax 直接针对该问题。

与 **Ditto**（CVPR 2022）的区别在 voting 对象：Ditto 用 PointNet++ regression + per-point **joint parameter** voting（SE(3) 参数空间投票），本方法用 per-voxel **occupancy** voting（几何空间投票）。两者互补但不重叠，且 Ditto 对未见类崩溃。

与 **DTA / DigitalTwinArt**（CVPR 2024）的区别在 state 数量与 canonical 结构：DTA 要 RGBD 双 scan + 学习的 part-segmentation field，本方法要 K=6 voxel + state-0 锚定。DTA 有 collision loss 与本方法的 swept volume 精神相近，但它作用于双 state，本方法作用于 canonical-part。

与 **ArtGS**（ICLR 2025）的区别在 canonical 位置：ArtGS 把 canonical Gaussians 放在 t=0.5 mid-state 追求时间对称，本方法锚在 t=0（closed），牺牲对称性换鲁棒性。ArtGS 需要 Chamfer-based motion clustering 识别 movable Gaussians，本方法用 count-indexed partition 直接给出，对部件数增长更稳定。

与 **ArticulatedGS**（CVPR 2025）的区别在实现基底：ArticulatedGS 用 3DGS + DeformNet，本方法用 voxel + screw-manifold IRLS。两者都 state-0 锚定（最接近的先例），但 ArticulatedGS 依赖 DeformNet 学习双 state 对应，本方法用 K=6 的组合式 count 规则**不需要任何学习**地确定 part partition。

与 **FreeArt3D**（SIGGRAPH Asia 2025）是最直接竞品，两者都基于 TRELLIS。FreeArt3D 用 SDS per-iter gradient 优化两个 hash-grid MLP 外加 auxiliary \"disk\" 归一化尺度，本方法跳过 SDS 与 hash grid，直接在 Stage B 的离散 voxel 上做 closed-form 投票 + IRLS 精化，**计算成本低几个数量级**，对 disk 依赖归零，但牺牲了 FreeArt3D 的连续表达能力（纹理、细节）。论文应明确定位为"FreeArt3D 补充而非替代"：本方法产出精确的 joint kinematics + URDF，FreeArt3D 产出 photorealistic articulated surface。

与 **GEAR**（2026）的区别是 EM 调度：GEAR 的 EM 是 state-symmetric（E-step 用 SAM 多视图弱监督），本方法的 EM 是 **state-asymmetric**（从最大打开态 state 5 冷启动、逆序传播），这是**最强的独立新颖点**，在所有调研工作中无直接先例。

与 **Articulate-your-NeRF**（NeurIPS 2024）是 state-0 anchor 最直接的 neural-field 先例。它用 voxel-grid 初始化 seg+articulation，但从单 state 的 NeRF density 抽取；本方法用跨 K=6 state 的 occupancy 组合规则抽取，信息源更干净且不需要 NeRF 训练。

## 预期的 CVPR/AAAI reviewer tough questions

**Q：你的 state-0 anchor 与 Articulate-your-NeRF / ArticulatedGS 有什么本质区别？** 答：他们在 neural field / Gaussian 空间，依赖单 state observation + 学习的 deformation network；本方法在离散 voxel 空间，依赖 K≥3 state 的组合式 occupancy 规则。ArticulatedGS 的 DeformNet 本质是学习一个 2-state 对应，本方法的 count partition 是 closed-form 6-state agreement，对部件数与 state 数增长更鲁棒。

**Q：voxel-level hard majority vote 是不是就是软投票 + 阈值？为什么不直接做 soft？** 答：做了。EM 稳定前用 hard（≥3 票）启动因为 T_k 残差未收敛，无法定义合理 weight；收敛后切 residual-weighted soft vote + 体积守恒阈值。Ablation 会显示 pure hard 比 pure soft 高 0.04–0.08 Dice，而 hybrid 再比 pure hard 高 0.03 左右（此处需要实验验证，不做无证据断言）。

**Q：从 state 5 冷启动是不是 cheating？如果 state 5 不是最大打开态呢？** 答：设计里有 adaptive anchor——检测 |S_hard|/|O_k| 选开得最大的 state，state 5 只是 K=6 序列下的名字。关键不是\"state 5\"这个编号，而是\"θ 最大的 state\"这个信息论选择：θ 大 ⇒ screw 轴可识别性高，冷启动误差小。GEAR 和 FreeArt3D 的 state-symmetric 策略在 θ 极小的 state 上同样要 pay the cost。

**Q：swept volume 用 canonical_move 扫而不是 M_k 扫，失败会怎样？** 答：若 canonical_move 本身被污染（例如含 base voxel 泄漏），swept 会 carve 掉 base 自己。防御是 |base_final|≥0.3·|O_0| 下界、前 N−1 轮只警戒不 carve 的 late commit、以及 canonical_move ⊆ Ω_c 的硬裁剪。这三道保险中任一条触发即阻止错误 carving。

**Q：IoU 只有 0.75–0.84，majority vote 的信噪比够吗？** 答：IoU 指的是 state 间差异，不是 vote 可靠性。真实数据里 c=6 的 voxel 数（4699）与 c=1 的 voxel 数（3301）分别占总 union ≈53% 和 ≈37%，即 vote 的主导票型覆盖 90% 的相关 voxel，只有 ≈10% 落在 c∈{2,…,5} 的 boundary 带——这个数字比 multi-atlas label fusion 文献里的典型场景（30–50% boundary）干净得多。

**Q：为什么不用 TRELLIS 自己的 SLAT 做 part clustering？** 答：因为 (a) 用户手里的 (8,16³) 是 SS-latent 不是 SLAT，SS-latent 只编码 occupancy 没有 appearance；(b) 真 SLAT 的 8 维是 entangled VAE bottleneck，训练目标无 part 监督；(c) PartField（ICCV 2025）、SegviGen（2026）等直接采用 TRELLIS 做 part 任务的工作都证明需要额外的 feature field 或 fine-tune，不能 zero-shot。Ablation 一次关掉这条路。

**Q：对称件（旋钮、圆柱抽屉把手）怎么办？** 答：λ₂/λ₁>0.95 检测 + N=8 hypothesis multi-tracking + sequential Phase 2 剪枝；真对称件最终报告\"轴方向 identified、绕轴角度 aliased\"，canonical_move 在对称不变量意义下仍然正确（正如对称件的 grasping pose 也必须在对称群下定义）。

**Q：K=6 是必需的还是可变？K=3 或 K=10 怎样？** 答：K=6 只是 Stage B 的设置。count partition 对 K 的鲁棒性是一个关键 ablation：理论上 K≥3 即可让 count∈{0,1,K} 的票型可分辨，K=6 给出 boundary 带的中间票型用于 swept volume 仲裁。K 越大，c(v)=K 的 base 支持集越干净，但 TRELLIS 预算增长线性。建议 Stage B 的 K 作为 pipeline-level hyperparameter 在论文里给消融。

**Q：URDF 最终怎么导出？** 答：URDF origin 设为 state 0 的 base frame 重心；base link 对应 base_final voxel 的 marching-cubes 网格；child link 对应 canonical_move 的网格；joint 类型由 Phase 2 结尾的 revolute/prismatic 仲裁决定；joint axis = ω，joint origin = q（revolute 的 pivot）或仅 direction（prismatic）；joint limits = [min_k φ_k − δ, max_k φ_k + δ]，δ 取运动范围的 5% 作 safety margin。contact_region 定义为 base_final ∩ dilate(canonical_move, r=2) ∩ axis_neighborhood(q, ρ=5 voxel)，用于下游抓取规划。

## 参数默认值（从真实数据直接锁定）

| 参数 | 默认值 | 真实数据依据 |
|---|---|---|
| base 阈值 | c(v)=6 | 4699 voxel 清洁支持集 |
| move 强种子阈值 | c(v)≤1 | 3301 voxel，各 state 独占 |
| boundary 带 | 2≤c(v)≤5 | 603 voxel，~15% |
| 体积守恒目标 | median_k \|M_k_exclusive\| = 518 | 各 state 独占数中位数 |
| 对称检测阈值 λ₂/λ₁ | 0.95 | 文献标准 |
| hypothesis 数 N | 8 | 经验 |
| base 下界 α | 0.3 | 从 4699/5862≈0.80 反推 |
| EM 内环迭代数 | 5–10 | IoU 0.75–0.84 意味着收敛较快 |
| soft vote 权重 β | 2/median(r_k²) | 数据自适应 |
| monotonicity 正则 λ | 0.1 | 弱物理约束 |
| swept warning 系数 | 0.01 | 仅梯度 hint |

这些默认值不是随便选的——每一个都能被论文里一句话解释清楚，这是对"拒绝过参数化"这条设计宣言的兑现。

## 工程实现的关键细节

64³ × 6 state × trilinear sampling 的 warp 操作总计 ~1.5M × 6 次 trilinear lookup，在 GPU 上单轮 EM 内环 <1 秒，全流程预期 30 秒量级，完全可负担。Screw manifold 的 5 维 IRLS 在 PyTorch/JAX 上数值稳定，不需要特殊求解器。关键数值风险在长 prismatic——完全抽出的抽屉 state 5 的 move 可能越出 O_0 的 bounding box，trilinear 采样出界的默认 0-fill 会让 canonical_move 缺失末端，需要显式把 canonical grid 扩到 128³ 或沿 ω 方向 padding。这是唯一的 resolution 限制点。

另一个实现注意点是 Phase 3 全局松弛的收敛性——ArtGS / PARIS 的经验是 joint 松弛容易震荡，应加 Levenberg-Marquardt 阻尼、line search 与 early-stopping（canonical_move 的 Dice 变化 < 0.005 停止）。iteration 预算建议 outer EM 6–8 轮，每轮 inner 解析求解 screw params + 一次 canonical vote 更新。

最后一个细节是 label fusion 文献里的 **spatial bias**：Wang TPAMI 2013 证明 weighted voting 在凸形 part 上产生 under-segmentation。本方法的 canonical_move 在 revolute 场景下是近凸的，因此最后一轮 swept commit 前应做一次 1-voxel morphological dilation 作为 deconvolution 补偿，然后再 carve base。这是本方法唯一需要"小规模后处理"的地方，既不是 loss 也不是单独 stage。

## 最后一段

这套设计的真正价值不在于单个 step 的新颖，而在于**组合式决策**：从 count 分布的贝叶斯先验到 canonical-move 的跨状态 vote，再到 screw-manifold 5-DoF 搜索与 anchor-then-relax EM，最后以 canonical-part swept volume 的 late-commit 收官——每一步都只用"刚刚够用"的复杂度，每一步的失败都被下一步或更晚的阶段回补。拒绝 loss soup 之所以可能，是因为组合式 count 规则把 part partition 这个本来需要训练监督或 SDS 的问题**降格为确定性几何查询**；拒绝 per-voxel matching 之所以可能，是因为 majority vote 在 K=6 state 上的冗余足以覆盖 TRELLIS 的 ~5% 表面噪声；拒绝过参数化之所以可能，是因为所有参数都有 Stage B 真实数据或 label fusion / screw geometry 文献的 closed-form 依据。审稿人对本方法最可能的正面评价不是"novel idea"，而是**"principled end-to-end formulation with minimal moving parts"**——这正是 CVPR/AAAI 近年奖项论文的典型文风。最大的剩余风险是对称件的轴 aliasing 与 极短冲程的冷启动退化，两者都已经有 deterministic 检测器与 graceful fallback，不会造成 silent failure。这份设计可以直接进入实现。