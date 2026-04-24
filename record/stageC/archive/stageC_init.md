# Stage C 重设计：从 occupancy signature 到 screw-manifold bundle adjustment

**核心结论先行（BLUF）。** 当前 SegMatch v3 失败的根因是"把硬阈值分割 → 学到的 per-voxel 描述符 → pairwise 配准 → graph-cut"这条链路上每一环都是**噪声源**，而没有一个环节利用了强物理约束。文献综述后我建议的新设计 **AOF (Articulated Occupancy Flow)** 抛弃 learned descriptor，转而依赖三个**闭合形式 / 强物理**信号：(1) K=6 状态下的 per-voxel 时间签名（bit-pattern）直接给出 base/move 初分割；(2) 移动部分的**惯性张量主轴轨迹**给出转轴 ω 的 closed-form 估计，与 **ContactArt / Part²GS 风格的 contact-anchor SVD** 互相冗余；(3) 最终在 **1-DOF screw manifold** 上做 Theseus / Lie-algebra Gauss-Newton 优化 TSDF 匹配 loss，把分割与关节拟合作为**耦合 EM**。这一设计能直接回答你列出的 8 个核心问题，并在与 FreeArt3D / PARIS / DTA / ScrewSplat 等 SOTA 的差异化上占住"体素原生 + K>2 + 显式物理先验 + 无需 SDS"这个空位。

下面给出完整的综述 + 方案设计。

---

## 文献景观的三块拼图

**第一块：training-free articulated fitting 的现状。** 过去三年这条线经历了从 2-state NeRF → 2-state 3D-GS → 多 state 生成式先验的跃迁。**PARIS (ICCV 2023, arXiv 2308.07391)** 是 training-free baseline：用两组多视图 RGB 学一对 NeRF（static + movable），通过一个由 MLP 预测的 per-point **segmentation field** s(x)∈[0,1] 决定颜色/密度的贡献权重，motion params (ω, q, θ) 作为可学习变量，loss 只有 RGB + silhouette + 一个鼓励 s 二值化的熵正则。它的痛点恰好是**只用 2 state**且不利用任何显式物理先验；其在对称件（旋转盘、圆形铰链）上的失败率在 DTA、ArtGS、VideoArtGS 等后续工作中被反复引用。**Ditto (CVPR 2022, arXiv 2202.08227)** 的公式结构不同：ConvONet 编码前后两个点云 → 两个 decoder 头分别预测 per-point segmentation 与 joint params (type, ω, q, θ)，损失是 BCE + 回归，**完全需要训练**，训练域外失效。**DTA / DigitalTwinArt (CVPR 2024, 2404.01440)** 从 2 次 RGBD 扫描出发，通过一个 segmentation field + per-part rigid transform 的耦合优化，并引入**collision loss**（part 之间不能穿插）——这是最早强调"contact/collision 应作为硬约束"的工作之一。

**ScrewSplat (CoRL 2025, arXiv 2508.02146)** 是目前最接近"显式 screw axis 优化"的 SOTA：它把 screw axis (ω, q, pitch) 连同置信度 γⱼ 一起作为 3D-GS 的可学习变量，从 nₐ≥2 configs 联合优化；其消融显示 axis-angle 误差从 nₐ=2 到 nₐ=6 近乎减半——这是 K>2 带来信号增益的**直接定量证据**，对我们 K=6 的设定是最有力的支持。**Part²GS (arXiv 2506.17212)** 加入 repel-point 场作为接触/碰撞先验，与物理约束的"轴必须穿过接触带"紧密对齐。**ArticulatedGS (CVPR 2025)** 与 **ArtGS** 继续走 2-state GS 路线但仍然是 2 state。**Watch-It-Move (CVPR 2022, 2112.11347)** 是早期少见的**多帧**方法（~100 帧视频 → 椭球 part + 静态邻接分析），但走的是 ellipsoid 分解，不直接给 axis。

**最关键的直接竞争者是 FreeArt3D (SIGGRAPH Asia 2025, arXiv 2510.25765)**。它与你的 pipeline 共享两个核心元素：**TRELLIS 作为 3D diffusion prior** + **training-free per-instance optimization** + **多 state 图像输入**。它的机制是：把 articulated 表示（static body + movable part + joint params + per-image joint state θ_k）作为可学习变量，每次 iter 根据当前 θ_k 把 movable part 变换后与 static body 拼合，渲染出一张"articulated 姿态下的 occupancy"，然后送进 **frozen TRELLIS 的 rectified-flow decoder** 做 SDS 3D-to-4D 梯度。这是很聪明但也很**贵**的公式——它本质上让 TRELLIS 告诉优化器"这个姿态下整个形状像不像真实的 3D"，但**没有显式利用 cross-state voxel 一致性信号**，也没有 inertia-tensor / contact-anchor 这类几何闭式解作为 warm-start；同时它在 PAct (arXiv 2602.14965) 的对比图中被指出 "often produces noisy or distorted part geometries and inconsistent part boundaries"——**这正是 SDS 的已知病：梯度 noisy，part 边界漂移**。我们 Stage C 的差异化定位因此非常清晰：**把 SDS 换成 cross-state occupancy signature + 物理闭式解 + screw-manifold 轻量优化**，把 FreeArt3D 用 SDS 做的部分切成"几何"+"先验"两块，用更便宜、更可验证的信号替换 SDS。

**第二块：moment-based / 刚体 / 对应无关 registration。** 这是你之前方案里缺的一大块。当 base/move 没有点对点对应时，有三条可用的闭式路径：(a) **Kabsch / Umeyama**（Umeyama 1991 PAMI 13:376）需要对应，不直接可用；(b) **无对应配准** ICP（Besl-McKay 1992）、**CPD** (Myronenko-Song 2010 PAMI)、**TEASER++** (Yang-Shi-Carlone 2020 T-RO, arXiv 2001.07715)、**Go-ICP** (Yang 2016 PAMI)——在没有好的初值时它们都容易陷入局部极小，尤其在旋转对称件上存在连续姿态模糊；(c) 对我们最关键的 **moment / inertia-tensor 轨迹方法**。后者的核心命题是：如果 M_k = R_k · M_0，则协方差 C_k = R_k C_0 R_kᵀ，**三个主轴随 R_k 旋转而三个特征值不变**。因此每个主轴 e_i^(k) 追踪一条圆弧，**圆所在平面垂直于 ω**。最直接的估计：ω 就是矩阵 A_i = Σ_k e_i^(k) e_i^(k)ᵀ 的最小特征值方向，或者等价地用 Chang-Pollard 2007 的 marker-circle-fit 推广到"主轴轨迹"。这在 biomechanics (Chang-Pollard 2007 J. Biomech., Gamage-Lasenby 2002) 里有先例但**从未被用于体素 articulated object fitting**——这正好是一块可以打的科学贡献点。

**screw theory / Chasles–Mozzi 定理**给出另一条闭式捷径：任何 SE(3) 都可写成绕某轴 (ω, q) 的旋转 θ + 沿同轴平移 h·θ。给定**先通过 RANSAC-ICP 得到的**一堆 {T_k}，可以在 **每对 (R_k, t_k) 上闭式抽出 ω̂_k = (R_k − R_kᵀ)∨ / (2 sin θ_k)、q_k = ½t_{k,⊥} + ½cot(θ_k/2)(ω × t_{k,⊥})**；然后对 {ω̂_k} 做 SVD 取主方向，{q_k} 对 ω 平面内取均值。这是**"从 SE(3) 样本反推单一 screw"的闭式 motion averaging**（Govindu 2001；Hartley-Trumpf-Dai-Li IJCV 2013），在文献里被充分研究但在 articulated fitting 社区几乎没被利用——**又是一个空位**。**revolute vs prismatic 的判别**可以做 AIC：设 RSS 为体素 Hamming / TSDF L2 残差，令 p_R = 4 + (K−1)、p_P = 2 + (K−1)，计算 AIC_model = 2·p + N log(RSS/N)，取小者；这种残差比分类器比你现在暗含的"靠旋转角大小拍脑袋"鲁棒得多。

**第三块：cross-state signature segmentation 与耦合优化。** 这块的综述最让我振奋，因为我们的问题结构**极其适合 signature-based seg**。Agent 3 的检索明确显示：同类信号在 LiDAR MOS、scene flow rigid clustering（**MultiBodySync** CVPR 2021 2101.06605；**Rigid3DSceneFlow** CVPR 2021；**OGC** NeurIPS 2022）、动静态 NeRF 分解（Nerfies/HyperNeRF/D-NeRF）、以及**unsupervised kinematic motion detection on shape collections** (Xu-Ruan-Sridhar-Ritchie SIGGRAPH 2022, UKMD, arXiv 2206.08497) 这些工作里都有先例——但**没有一个直接把 per-voxel K-bit signature 作为分割信号**。Yan-Pollefeys 2005/2008 PAMI 的**articulated factorization**是精神上的祖先：把 feature trajectory 矩阵按秩分解出不同 rigid subspace。把这个思路下沉到 voxel：每个 voxel 是一个 K-bit 向量，base voxels 向量接近 **(1,1,1,1,1,1)**，move voxels 向量随 k 变化，空 voxels 是 0——**这是一个在 {0,1}^K 上的聚类问题**，天然适合 MRF / graph-cut。用 TRELLIS 的 z_final (K,8,16,16,16) 作为额外的 unary 证据，而**不是** trilinear-upsample 到 64³ 再算 pairwise NN（你现在做法的 fatal flaw 是 8 维 latent 在 upsample 到 64³ 后信息已经坍塌，NN match 的信噪比很差）。

**耦合 seg+fit 的收敛理论。** 这是 CVPR reviewer 必问的。经典结论：Torr 1998 / Vidal 2004 的 EM-style multi-body rigid motion 算法在任意初值下**单调下降**但只保证**局部收敛**；ICP 本身（Besl-McKay 1992）也只有局部收敛。一个经验上 robust 的做法是"**先闭式 warm-start → 再 alternating EM → 最后 joint bundle adjustment**"三段式，每一段监控残差指标（Dice on M_k, axis-angle Δω, segmentation flip rate）。PARIS 没有做这种三段式，所以它在小 K 时极其敏感初值；我们的 K=6 + 物理闭式 warm-start + bundle adjustment 可以在这一点上显著胜过它。

---

## 对你 8 个核心问题的逐条回答

**问题 1：SOTA 与 limitation。** 按"我们应该直接比较"的相关度排序，5 个最重要的前作是：**FreeArt3D** (TRELLIS+SDS+多 state，training-free，但 SDS noisy、无物理先验、part boundary drift)；**ScrewSplat** (显式 screw axis + nₐ>2，training-free 式优化，但依赖 GS 和多视图渲染，不是体素原生)；**PARIS** (2-state，NeRF，有 seg field，对称件失败)；**DTA** (2 RGBD，含 collision 损失，强 baseline，2 state)；**Ditto** (监督，2 state 点云)。次要参考：Part²GS (repel-point 接触先验)、Watch-It-Move (多 state 但非 axis-centric)、ArtiLatent (arXiv 2510.21432，用 TRELLIS sparse voxel 作为 articulated 生成的 latent，与我们最贴近但它是 generative 不是 per-instance fitting)。**共同 limitation**：几乎所有方法的 seg 和 joint 要么都是 learned（Ditto/URDFormer），要么都依赖 NeRF/GS 视角差（PARIS/DTA/ScrewSplat/FreeArt3D），**没有一个**是"体素原生 + 多 state signature + 显式惯性张量/接触闭式 warm-start"的组合。这就是 AOF 的生态位。

**问题 2：cross-state occupancy signature 能否直接做 base/move seg？**可以，而且应该是主干信号。令每个体素 v 的签名 s(v) = (O_1(v),...,O_K(v)) ∈ {0,1}^K。**严格论述**：在刚体假设下，一个 base 体素在所有 state 中占据值不变（除了接触处的一圈薄壳），一个 move 体素则随 φ_k 改变位置。因此：**(i)** `count(v) = Σ_k O_k(v)` 天然是 base-likeness 指标，count=K 是 pure base，count∈{1,...,K−1} 是 move，count=0 是空；**(ii)** 但纯 count 有歧义：某些 move 体素在多个 state 出现（如 hinge 附近运动幅度小的体素会长期被占），所以需要第二个量—— **persistence run-length**：把 signature 视为序列，统计最大连续 1 的长度 ℓ(v)。对 base 的期望是 ℓ(v)=K，对 move 期望是 < K/2。**(iii)** 加入空间平滑的 MRF：定义 unary E_data(v) = αₓ·f(count, ℓ) + β·g(M_attn(v)) + γ·h(z_feat consistency across k)，pairwise E_smooth(v,u) = wₕ·|s(v)⊕s(u)|（Hamming + 空间近邻），用 α-expansion 最小化。文献里**没有一个方法**做过这种"bit-pattern + 空间正则"的 base/move seg；你现在用 M_attn<0.3 硬阈值是把最强的信号 (count/persistence) 完全浪费了。

**问题 3：moment-based 判型是否有先例？**有，但不在 articulated 社区。Chang-Pollard 2007 在 biomechanics 里用 marker 轨迹的圆弧拟合估 rotation axis，Gamage-Lasenby 2002 做 plane-fitting for hinge axes，机器人 kinematic calibration (Hollerbach, Khalil) 里也有类似闭式公式。在 articulated object 文献里我没有找到把**惯性张量主轴 × K-state 轨迹**做 ω 估计的工作——这是一个清晰的 novelty 窗口。但要诚实说明 **degeneracy**：当两个 eigenvalue 相等（薄板、圆盘、柱体）主轴在 2D 子空间里不唯一，这条方法会失败；当三个 eigenvalue 全相等（球状件）完全失败。补救是用 eigengap g = (λ_3 − λ_1)/λ_3 作为置信度，g<ε 时退回 contact-anchor SVD 或 TEASER++ 路径。

**问题 4：1-DOF manifold constrained optimization 怎么写最好？**最干净的写法是直接参数化而非罚项。设 ξ = (ω, v) 是单位 screw，其中 v = −ω × q（纯旋转）或 ω = 0, v 单位（prismatic），则 T_k = exp(φ_k · ξ̂)。变量 θ = (ω ∈ S², q ∈ ℝ³/ℝω（即 q 垂直 ω 的投影）, {φ_k}_{k=1..K−1}, B, M)。正则 (‖ω‖−1)² + (ω·(q×ω))² 把变量限制在流形上。loss 用 TSDF L2 而非 hard occupancy L2（Curless-Levoy 1996 / KinectFusion Newcombe 2011）：L_fit = Σ_k ‖TSDF(B) ⊕ warp(TSDF(M), T_k) − TSDF(O_k)‖²_{Huber}；soft-Dice 作 auxiliary（MONAI Dice 的 IoU=True 版本）。关键工程决定：**在 tangent-space 做 Adam 或 Gauss-Newton**（Solà *Micro Lie Theory*, arXiv 1812.01537；jaxlie/Theseus 2207.09442），每步 retract 回流形；绝不在原始参数上无约束 Adam，否则 ω 会漂出 S²。Theseus 的 LM + 稀疏 Jacobian 是 2026 年的最佳选择，备选 PyTorch3D SE(3) + 手写 Adam。

**问题 5：耦合 seg+fit alternating 的 prior 与收敛性。** 有 Torr-Murray 1993、Torr 1998、Vidal-Ma 2004 GPCA、Tron-Vidal 2007 的多刚体 EM 收敛分析：**任意 E-step + M-step 都单调降低 EM 似然**，但只能保证**局部极小**。ICP（Besl-McKay 1992）同理。实践上的 robust pipeline 必须包括：(a) **多 restart**（不同 ω 初始化）或 (b) **warm-start 闭式解** 把初值塞到全局极小的吸引域里——这正是惯性张量 + contact anchor 的价值。CVPR reviewer 会追问"你为什么确信 local minimum 足够好"，标准回答是：ablation 多 init、report convergence basin、show fraction of cases where pose distance between 10 random inits drops below ε。**另一个重要的保护措施**：每次 E-step 后强制 seg 与当前 {T_k} 一致（即 move 体素必须满足 warp(M_0, T_k) 与 O_k \ B 重合），这个一致性约束是 ICP 的类 Picard 迭代。收敛性正式保证来自 Lloyd-style monotone decrease 论证。

**问题 6：contact-anchor 作为硬约束的先例。** 核心前例是 **ContactArt (ICCV 2023, arXiv 2305.01618)**（在 contact map 上做 diffusion prior）、**Ditto-in-the-House (ICRA 2023)**（把 contact heatmap 作为输入）、**Part²GS (2506.17212)**（repel-point）、以及 robotics 的 joint-calibration-from-contact 文献。**但把 contact 做成"轴线必须穿过 anchor set"的几何硬约束**非常干净：给定 anchor 集合 C = argtop-k M_attn（或者 base 与 move 签名从 1 突变到 0 的边界体素），**点到轴线的距离平方和最小化有闭式解**（Eckart-Young / SVD）：q* = mean(C)，ω* = C 去中心后的 top-1 right singular vector。这给出 ω 和 q 的**第一个 warm-start**，且与 inertia-tensor 主轴法**几何独立**——两个冗余的闭式解可以互相 cross-validate。Plücker 坐标 (ω, m = q×ω) 在 LM 里的残差 ‖ω×c − m‖² 是理想软约束。**文献里至今没人把这种 contact-Plücker-SVD 作为 articulated fitting 的 warm-start**，这是第二个 novelty 位。

**问题 7：K>2 vs K=2 的额外信号。** 三点显著增益：**(i) DOF 过定度**：revolute joint 有 4 DOF（ω: 2 + q⊥: 2）+ K 个 φ_k + 2 个形状 field。K=2 给出 12 个 rigid observables 对 6 个未知参数；K=6 给出 36 对 10，**过定比从 2× 跃升到 3.6×**，AIC/BIC 残差检验才有统计意义。(ii) **motion averaging 鲁棒性**：{T_k} 中若有一个因为 ICP 陷入局部极小而错了，在 K=2 下你完全检测不到，K=6 下可以用 RANSAC-on-screws 或 median-on-rotation-log 排除。(iii) **signature segmentation 本身依赖 K**：K=2 的 signature 只有 4 种 pattern 信息太少，K=6 有 64 种足以区分 base/边界/move/暂态。ScrewSplat 的 ablation 显示 axis-angle 误差从 nₐ=2 到 nₐ=6 近乎减半，这是最直接的经验证据。我们应该做 K∈{2,3,4,5,6,8} 的 ablation——reviewer 肯定会问。

**问题 8：sparse voxel SE(3) 的 robust 方法。** 硬 occupancy 的梯度死区是核心陷阱。四条 fix：(a) **trilinear-interp soft occupancy**（标准，PARIS/Ditto/ScrewSplat 都用）；(b) **TSDF + truncation**（KinectFusion Newcombe 2011, BundleFusion），梯度在距离 ≤ τ 的 shell 内都有效，**这是 pose 最 robust 的信号**；(c) **morphological blur / Gaussian blur**（σ≈1.5 voxel）作轻量替代；(d) **coarse-to-fine multi-resolution**（32³→64³→128³）先求 global 再 refine。对于 K=6, 64³ 的规模（1.5 MB/state 的 float），我们完全可以直接存 TSDF 并用 Theseus 的 LM。**SE(3)-equivariant sparse conv**（SS-Conv NeurIPS 2021 2111.07383）在 training-free setting 下不是必须。

---

## AOF (Articulated Occupancy Flow) 方案设计

**设计哲学**：三段式 **Signature → Closed-form warm-start → Coupled bundle adjustment**，每一段都有可独立 ablate 的输出，且每一段都有物理意义而非 learned descriptor。下面逐段给出公式级细节。

### 第 1 段：signature-based initial segmentation

输入 O_stack ∈ {0,1}^{K×64³}, M_attn ∈ [0,1]^{64³}, z_final ∈ ℝ^{K×8×16×16³} (trilinear-upsample 到 64³ 作 z_64)。对每个体素 v：

```
count(v)  = Σ_k O_k(v)                               # base-likeness, ∈ {0,...,K}
persist(v)= max_k run_length_of_ones(s(v), start=k)  # persistence run
var_z(v)  = var_k ‖z_64(k,v)‖²                       # TRELLIS cross-state semantic var
```

定义三类初始标签：**pure-base** 若 count=K 且 persist=K（硬约束，无噪声时精确）；**pure-empty** 若 count=0；**candidate-move** 其它。然后在 candidate-move 上解 MRF:

```
E(y) = Σ_v [λ₁·(1 − y_v)·φ_base(v) + λ₂·y_v·φ_move(v)] + λ_s·Σ_{(u,v)∈N6} 𝟙[y_u≠y_v]
φ_base(v) = count(v)/K − 0.3·M_attn(v) − 0.1·var_z(v)  # higher = more base-like
φ_move(v) = 1 − φ_base(v)
```

用 α-expansion / graph-cut 求解（PyMaxflow 或 gco_python）。输出 B⁰（canonical base 初估，= all pure-base ∪ y=0 voxels）、M_k⁰（每个 state 的 move seg = O_k \ B⁰）。**关键区别于 v3**：(i) 主信号是 cross-state bit-pattern 而非单帧 z_final descriptor；(ii) M_attn 与 z_var 作**辅助**而非主；(iii) 硬 pure-base 约束保证 base 不会漂。

### 第 2 段：闭式 warm-start（两条冗余路径）

**Path A — inertia-tensor 主轴轨迹**。对每个 k，用 M_k⁰ 计算质心 c_k 与协方差 C_k，特征分解 C_k = Σ λ_i^(k) e_i^(k) e_i^(k)ᵀ。用 eigengap g_k = (λ_3 − λ_1)/λ_3 标记对称性，对 g_k < 0.05 的轴跳过。对每个非退化 i，构造 A_i = Σ_k e_i^(k) e_i^(k)ᵀ；ω 的候选是 A_i 最小特征值方向。跨 i 聚合：Ã = Σ_i (A_i / tr A_i)，ω̂_A = smallest-eigvec(Ã)。**符号规范化**：强制 (ω̂_A · (c̄_late − c̄_early)) 为固定符号。

**Path B — contact-anchor Plücker-SVD**。定义 anchor 集合 C = {v : 0 < count(v) < K AND v ∈ 3-voxel shell around ∂B⁰}（也就是 base-move 的接触带）。闭式：q̂_B = mean(C), ω̂_B = top-right-singular-vec((C − q̂_B)ᵀ)（3×|C| 矩阵）。此路径在 Path A 因对称退化时仍可用。

**融合 ω** = SLERP(ω̂_A, ω̂_B; w = g_A / (g_A + g_B))，**q** = 取 Path B 的 q̂_B 投影到 ω 的垂直子空间。

**Path C — pairwise ICP + screw motion averaging**（第三冗余，cost 几秒）。对每对 (M_0⁰, M_k⁰) 跑 TEASER++（对称件）或 point-to-plane ICP（非对称件），拿到 {T̂_k}。逐对闭式抽 screw：
```
θ_k = arccos((tr R̂_k − 1)/2), ω̂_k = (R̂_k − R̂_kᵀ)∨ / (2 sin θ_k)
q_k = ½·t_{k,⊥} + ½ cot(θ_k/2) (ω̂_k × t_{k,⊥}),  t_{k,⊥} = t̂_k − (ω̂_k·t̂_k)ω̂_k
```
然后 ω̂_C = top-left-singular-vec of [ω̂_1, ..., ω̂_{K−1}]（符号对齐后）, q̂_C = 投影均值。**最终 ω₀, q₀** = 三路 weighted average by 各自 confidence（g, residual, inlier ratio）。这种**冗余闭式 warm-start** 是 AOF 相对 PARIS/FreeArt3D 最干净的鲁棒性改进。

**关节类型判别（AIC）**。用当前 {T̂_k}（从 Path C）计算 R_θ = max_k θ_k 与 R_t = max_k ‖t̂_k‖。Fit 两个模型得 RSS_R, RSS_P，N = K·|move voxels|。比较
```
AIC_R = 2(4+K) + N log(RSS_R/N),   AIC_P = 2(2+K) + N log(RSS_P/N)
```
选小者；当 AIC_R − AIC_P 落在 [−10, 10] 内要求 R_θ > 3° 才宣布 revolute，避免 screw-like 边界歧义。

### 第 3 段：1-DOF manifold coupled bundle adjustment

变量集合 θ = (ω ∈ S², q ∈ ℝ³ constrained ω·q=0, pitch h, {φ_k}_{k=1..K−1}, 软 canonical base B̃ ∈ [0,1]^{64³}, 软 canonical move M̃ ∈ [0,1]^{64³}, 软 per-state assignment a_k(v) ∈ [0,1]）。T_k = exp(φ_k · ξ̂) with ξ̂ = [ω̂ (−ω×q + hω); 0 0]。total loss：

```
L = λ_fit · Σ_k ‖TSDF(B̃) ⊕ warp(TSDF(M̃), T_k) − TSDF(O_k)‖²_Huber        (a) volumetric fit, Curless-Levoy 1996
  + λ_dice · Σ_k [1 − Dice(B̃ ∪ warp(M̃, T_k), O_k)]                        (b) Dice aux, MONAI
  + λ_axis · Σ_{c∈C} ‖(c − q) − ((c−q)·ω)ω‖²                              (c) point-to-axis, ContactArt 风格
  + λ_plk  · Σ_{c∈C} ‖ω × c − (q × ω)‖²                                   (d) Plücker residual
  + λ_repel·Σ_{x∈∂B̃, y∈∂warp(M̃,T_k), ‖x−y‖<r} max(0, r − ‖x−y‖)²          (e) repel / no-interpenetration, Part²GS
  + λ_consist · Σ_{k,v} ‖a_k(v) − σ(B̃(v) − warp(M̃, T_k)(v))‖²            (f) seg-motion coupling
  + λ_bin  · Σ_v [a_k(v)(1−a_k(v))]                                       (g) binary prior, PARIS-style entropy
  + λ_contact · ‖pool_contact(B̃, M̃, T_k) − M_attn‖²                       (h) M_attn 作为弱 unary，不作硬阈
  + λ_smooth · Σ_k (φ_{k+1} − 2φ_k + φ_{k−1})²                            (i) state-smooth, Watch-It-Move
  + λ_unit · (‖ω‖−1)²  +  λ_ortho · (ω·q)²                                (j) gauge
```

**建议权重**（初值，需验证集扫描）：λ_fit=1.0, λ_dice=0.3, λ_axis=0.2, λ_plk=0.05, λ_repel=0.1, λ_consist=0.3, λ_bin=0.01→0.2 anneal, λ_contact=0.2, λ_smooth=0.02, λ_unit=10, λ_ortho=10。

**优化器**：Theseus (arXiv 2207.09442) 的 LM + 稀疏 Jacobian，在 ω ∈ S²、q ∈ ℝ²（in ω-⊥-plane）、φ_k ∈ ℝ 上做 tangent-space retraction；B̃, M̃, a_k 用 Adam（lr=1e-2，500 step）；交替更新：固定 (ω,q,{φ_k}) 更新 (B̃, M̃, {a_k}) 5 步，然后固定 (B̃, M̃, {a_k}) 更新 (ω,q,{φ_k}) 1 步 LM。整个优化 < 3 min on 4090。

**输出**：(joint_type, ω, q, {T_k=exp(φ_k ξ̂)}, canonical_base=thresh(B̃, 0.5), canonical_move=thresh(M̃, 0.5), contact_region = argtop-k M_attn ∩ anchor refined, per_state_assignment=argmax a_k)。URDF 由 (ω, q, type, φ_min=min φ_k, φ_max=max φ_k) 直接写出。

---

## 五个 reviewer 最可能提出的 tough question 与预先反驳

**Q1："相对 FreeArt3D 的 novelty 是什么？都用 TRELLIS + 多 state + training-free。"** A: FreeArt3D 把整个 articulation 耦合到 SDS 梯度里；我们**完全不用 SDS**，把几何信号拆成"cross-state signature"、"inertia-tensor moment"、"contact anchor"三个**显式的、物理可解释的、闭合形式的**先验，然后只在第 3 段用 differentiable loss 精调。FreeArt3D 论文自己在 PAct 的对比里被指出 part boundary 不稳定；我们的 signature-based seg 给出**硬 pure-base 约束**直接解决这个问题。此外我们做 K=6 ablation vs FreeArt3D 的默认 3-4 state。

**Q2："inertia-tensor 主轴法在对称件上就会崩——你的对称件效果怎么样？"** A: 我们有 eigengap 阈值自动退化检测（g < 0.05 跳过），并有 contact-anchor SVD（Path B）和 ICP+screw averaging（Path C）作冗余；同时在 PartNet-Mobility 的对称类 category（如 rotary knob, circular hinge）上专门做 ablation。Paper 里 explicitly report eigengap distribution 作为诊断。

**Q3："M_attn 错了怎么办？你依赖它做 contact anchor。"** A: M_attn 只在 (h) 项作**弱 unary**（λ_contact=0.2）和 anchor 候选初选，真正的 anchor 是由 signature 的"count 从 K 到 <K 的空间突变 + 3-voxel shell"几何决定的；ablation 要 show M_attn 加 0%/20%/50% 噪声时最终 axis error 的变化（我们预计应该 robust，因为 signature 本身就能定位接触带）。

**Q4："K=6 必要吗？6 张 Wan2.2 视频帧 vs 2 张的代价怎么算？"** A: 做 K∈{2,3,4,5,6,8} ablation。基于 ScrewSplat 的先例，axis-angle 误差随 K 下降 ~√K；我们期望 K=6 是 accuracy/compute sweet spot。K=2 会直接退化到 PARIS 的 regime 并展示我们优于 PARIS 因为 signature/inertia 仍可工作（PARIS 纯靠 SDS 渲染-loss 耦合优化）。

**Q5："soft-Dice / TSDF loss 在 disjoint 时梯度消失——你的 warm-start 保证重合吗？"** A: 三路闭式 warm-start（inertia + contact + ICP-screw）被设计成在 **signature-based init seg** 后已经把 {T_k} 放到正确的吸引域；即使如此第 3 段在前 100 step 用 σ=1.5 voxel 的 Gaussian blur 缓解 disjoint 问题，并 anneal 到原分辨率。失败 case（初始 IoU<0.3）用 Theseus 的 trust-region + restart 兜底。

---

## 次要 reviewer 问题的简短预案

**(a) "只处理单关节是不是太弱？"** — MVP 论文惯例；在 future work / 附录里示范 AOF 扩展到 k-joint 通过 EM-style joint segmentation + per-joint AOF 的 straightforward 扩展（跟 DTA 和 Ditto-in-the-House 的 kinematic tree 处理一样）。

**(b) "TRELLIS 的 z_final 用得不够？"** — 在 signature-MRF 里 var_z(v) 作 auxiliary unary，在 (h) 项 M_attn 是 z_final 的 DiT attention 派生量；我们**刻意**让几何信号主导，因为 learned descriptor 在 cross-state fitting 里已被 SegMatch v3 证明不可靠——这是一个**消极结果驱动的设计选择**，要在论文里明写。

**(c) "跟 ScrewSplat 的差异？"** — ScrewSplat 是 GS 渲染 loss，需要多视图图像；AOF 是体素 TSDF loss，直接在 Stage B 产物上跑，不需要回到像素。

**(d) "运行时间？"** — 三段加起来 <5 min/instance on 4090；FreeArt3D 也是 minutes 级，PARIS 是几十分钟。

**(e) "benchmark？"** — PARIS Two-Part Dataset、PartNet-Mobility 12 类、ACD (Iliash 2024)。指标：joint axis angular error、axis position (closest distance)、joint type acc、part Chamfer-L1、full-shape Chamfer-L1。baselines: PARIS、Ditto、DTA、ArtGS、ScrewSplat、FreeArt3D（以其公开的结果）。

---

## 结论与新洞察

**最大的洞察是**：你当前 v3 方案失败不是因为"分割 + 配准"这条思路错了，而是因为**每一步都用了最弱的信号**——硬阈值分割丢掉了 cross-state bit-pattern，learned descriptor 在 upsample 后信噪比低，pairwise NN matching 对几何对称完全敏感。文献综述明确显示：**cross-state voxel signature 是被所有相关工作忽视但对你设置最 "free" 的信号**（K=6 几乎白送），**moment-based axis 估计在 biomechanics 有强先例但从未进入 articulated fitting 社区**，**contact-anchor SVD 是 5 行代码的几何硬约束但没有 SOTA 论文采用**。把这三个几乎"被文献遗忘"的闭合形式信号叠加起来，在第 3 段用 screw-manifold bundle adjustment 做耦合精调，就能得到一个**物理可解释、训练无关、对 K 单调收益、在 reviewer 面前可全面 defend**的 Stage C。

AOF 的论文定位应该是"**从 learned heuristic 回到物理先验**"而非"又一个 SDS 变体"——这对 CVPR/AAAI 的审稿 trend（2025 年之后显著青睐 interpretable/training-free/physics-grounded）也更友好。最后要做的一件具体的事：立刻写一个 10-objects 的 pilot，只实现第 1 段（signature MRF）+ Path B（contact SVD），不跑第 3 段，看 ω 误差能到多少；如果已经能打过 PARIS 2-state baseline，那第 3 段只是锦上添花，论文骨架就已经成立。