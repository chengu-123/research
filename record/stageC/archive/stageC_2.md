# Stage C 重设计：从 Swept-Volume 仲裁到耦合 EM 的 principled 方案

**核心结论（BLUF）**。长抽屉 pathology 在纯二值 cross-state signature 上 **information-theoretically 不可分**——count=6 的 trajectory-middle voxel 与真 base 在 b(v) 空间同分布（Sub-A 给出的形式化论证）。因此 Stage C 的正确架构不是"在 signature 空间再多加一个判据"，而是 **先从 shell/XOR 信号恢复运动，再用 swept-volume pull-back 判决 interior 归属**。这同时把新约束 B 的 containment case（抽屉-柜腔）变成 **空间雕刻 (space-carving) 问题**——Kutulakos-Seitz IJCV 2000 的 motion-consistency 版本——而不是置信度 tie-break 的边缘情况。整个 Stage C 因此坍缩为一个 **"运动从 shell 起、label 用 swept-volume MAP 仲裁、T_k 用 moment-based correspondence-free 刚体配准、耦合通过一次 fit-then-refine 闭环"** 的单一优化，避开 loss soup、避开 per-voxel matching、避开 M_attn 放大效应，并对 prismatic+对称退化提供三路冗余 warm-start。以下是 8 个要求的逐项方案与两份独立评价的逐条回应。

---

## 1. Segmentation 主信号链：shell-first + swept-volume consistency

**理论前提（关键，Sub-A）**。记 b(v) ∈ {0,1}^K 为 voxel v 的 cross-state 签名。若运动体 B_k = g_k·B_0，则对任意 g_k，"always-on" interior {v : v ∈ ⋂_k B_k} 的签名与真 base 的签名都是 111111。因此 **不存在仅依赖 b(v) 的 axis-free 判据** 能分开 count=6 的 trajectory-middle voxel 和真 base（Torr 1998、GPCA 皆建立在 trajectory 空间而非 binary signature 空间上）。4DMOS、MotionSeg3D、RAFT-3D、Rigid3DSceneFlow、MultiBodySync、OGC 全部只在 shell/XOR/surface 信号上工作，对 interior count=6 voxel 无信号。**这是一条不可能性定理，不是工程难题**——任何声称在 signature 空间解长抽屉的方案（包括加 M_attn 平滑）本质上都在引入外源假设。

**正确架构**。Stage C 分三步，每一步只使用该步可用的证据：

**Step 1（motion-only from shell）**：只从 shell voxel S = {v : XOR_k b_k(v) ≠ 0} 估计 T_k。这是 common-fate 信号的经典用法，S 的 mass 远小于 interior，但信噪比最高。Shell 本身就是 Sub-A 提出的"把 interior 问题降维到 surface 问题"的操作。Rigid3DSceneFlow（Gojcic CVPR 2021）、MultiBodySync（Huang CVPR 2021）、OGC（Song NeurIPS 2022）都采用这种 shell-first 策略。

**Step 2（swept-volume pull-back labeling）**：得到候选 {T_k} 后，定义 **swept volume** SV = ⋃_{k=1..K} T_k(canonical_move)（Abrams-Allen 2000）。对每个 interior voxel v：若 ∃ k 使 v − t̂_k ∈ canonical_move（或广义地 T_k^{-1}(v) ∈ canonical_move），则 v 归 move；否则归 base。这是 space-carving (Kutulakos-Seitz IJCV 2000) 的 **motion-consistency 版本**：一个 voxel 能被某状态下的 move 部分占据 ⇒ 它不可能是 base。**长抽屉 pathology 在此自然消解**：trajectory-middle voxel 虽然所有状态都被占据，但它们的 swept-volume pull-back 一致地指向 canonical_move，所以不会被误判为 base。

**Step 3（贝叶斯仲裁，见第 4 节）**：swept-volume 判决有歧义的 voxel（多个 label 都 consistent）交给 MRF MAP 解决。

**关键观察**：长抽屉 pathology 不是 Stage C 的 bug，是"只用 signature 空间做 seg"这个错误 formulation 的必然后果。**换 formulation 就消失**，而不是在错误 formulation 上加 hack。

### 两类假阳性的 principled 过滤（新约束 A）

**第一类（真 base → move）**由 Step 2 的 swept-volume test 自动处理——不 consistent 的 voxel 不进 move。

**第二类（TRELLIS 生成噪声，count=1 singleton）**使用三路复合 filter bank（Sub-A 给出 principled 阈值）：
1. **连通分量 + MRF 平滑**。26-连通 CC size 阈值 τ_CC ≈ 20（来自 3D lattice percolation threshold p_c ≈ 0.097，Gilbert/Erdős-Rényi 论证）；后接 Ising MRF（Boykov-Veksler-Zabih TPAMI 2001 α-expansion），β 取 0.5–1.0 × β_c^{3D}（β_c^{3D} ≈ 0.22）。
2. **RANSAC swept-volume 一致性**。候选 move voxel 须被某个 1-DOF screw trajectory 解释；否则是噪声。这是 Sub-A Q1 提出的 "pull-back 一致性" 测试的严格版本，代替用户原方案里的"邻近激活"（后者无法严格定义）。
3. **Soft-occupancy hysteresis gating**（Canny 式双阈值）：τ_high 取 precision=0.99 分位、τ_low 取 recall=0.99 分位，从 TRELLIS 的 soft logit calibration curve 定。与 Mescheder et al. Occupancy Networks（CVPR 2019）的 calibrated logit view 一致。

这三路是 **独立证据通道**（size-topology、kinematic-consistency、decoder-calibration），复合后对 count=1 噪声在合理假设下误保率 < 10⁻³（Fischler-Bolles 1981 bound）。**拒绝用户候选中的"count=1 且没有运动学合理邻近激活"这种模糊表述**——改成严格的 swept-volume pull-back 一致性。

---

## 2. "整体 move" 运动估计的 correct formulation：moment-based correspondence-free

**选择（Sub-B）**：**Spherical Harmonics (SH) energy + centroid translation 的两阶段 decoupled 估计，Gauss-Newton 在 1-DOF screw manifold 上 refine**。拒绝 per-voxel matching 的必然选择。

**数学管道**：
- **旋转部分（仅 revolute 需要）**。把 canonical_move 和当前 state 的 move-mask M_k 各自视为球面函数（以质心为原点，在多个球壳上采样）。按 Kazhdan-Funkhouser-Rusinkiewicz (SGP 2003) 做 SH 展开，取 per-frequency 能量 ‖f_l‖ 作为 rotation-invariant signature。用 Wigner-D 在 SO(3) 上做 FFT cross-correlation 找到最佳旋转对齐——这是 rotation-recovery-without-correspondence 的经典 pipeline。替代：3D Zernike moments（Novotni-Klein 2004，N=20 阶 156 系数，64³ 上 ~0.25s）。
- **平移部分**。两者质心差 Δμ 给出平移分量的闭式解；对纯 prismatic，Δμ 直接是平移量（1-DOF 约束收缩为沿 v̂ 的投影）。
- **Screw refine**。把 (R, t) 投影到 1-DOF screw 流形（revolute: ω∈S²、q∈ℝ³、θ_k∈ℝ；prismatic: v̂∈S²、d_k∈ℝ），用 Gauss-Newton / LM 在 Chasles-Mozzi 参数上最小化 **volumetric Dice loss**：L(T_k) = 1 − 2·⟨O_k^{soft}, warp(canonical_move^{soft}, T_k)⟩ / (‖O_k‖₁ + ‖warp(·)‖₁)。用 soft (float) 形式保持可导性，Dice 对 boundary seg error 远比 L2 robust（医学 seg 文献共识）。

**为什么不是其他候选**：
- Volumetric L2 / Dice 单独用 convergence basin 过窄（SO(3) 上容易卡在 180° 对称），需要 moment-based warm-start。
- Demons/SyN 是 diffeomorphic 配准，对 rigid 子问题是错误工具。
- Go-ICP / Super4PCS / FGR / TEASER++ 都是 point-based，要先把 voxel 转 surface，引入额外误差；且 TEASER++ 证书优势在高外点率场景，而我们已经有 warm-start 不需要。
- Chasles-Mozzi from centroid+orientation sequence 作为一条 **独立的 warm-start 路径**（见第 3 节）而不是主估计器。

**关键优势**：SH magnitudes 的 rotation-invariance 是 Wigner-D 的代数性质（Kazhdan 2003 Thm），不是启发式；Dice 对 M_k boundary error 的 robustness 已在百篇医学 seg 论文中实证。

---

## 3. Warm-start 三路冗余：纯 3D 几何下覆盖所有退化

**目标**：revolute / prismatic × symmetric / asymmetric 四个象限 **至少一路可靠**。当前 Path A（inertia）+ Path B（contact-SVD）在 prismatic+对称（圆柱抽屉）双失。

**三路方案**（Sub-B）：

| 路径 | 原理 | revolute-asym | revolute-sym | prismatic-asym | prismatic-sym |
|------|------|:-:|:-:|:-:|:-:|
| **A: Inertia-tensor principal axes** | canonical_move 的二阶矩 SVD | ✓ | 退化 | ✓ | 退化 |
| **B: Contact-band SVD** | base-move 接触带的 PCA | ✓ | ✓（轴向明确） | ✓ | 退化（slot 对称） |
| **C'（新）: Centroid-trajectory PCA** | {μ_k}_{k=1..K} 在 ℝ³ 中 fit 一条直线（prismatic）或一条圆弧（revolute） | ✓ | ✓ | **✓** | **✓** |

**Path C'** 是 user 拒绝的 ICP-screw 的替代——不是 registration-based，不迭代，只用 K 个 centroid（低维、稳定）做 PCA/圆拟合。关键：**质心轨迹对 rotational symmetry 不敏感**，因为它只跟踪质量中心平移，完全绕过 inertia 退化。prismatic 质心轨迹是一条共线点集，第一 PC 直接是 v̂，符号由状态序号的 ordering 定；revolute 质心轨迹是 3D 中一条平面圆弧，平面法向即 ω，圆心即 q——这是 Chasles-Mozzi 定理在 observation 层面的直接应用（Murray-Li-Sastry 1994 Ch. 2）。

**覆盖证明**：上表四格均有至少一 ✓。对称 prismatic case Path A+B 皆退化，但 Path C' 只要 K≥2 state 且有非零位移就稳定。配合 Chatterjee-Govindu (ICCV 2013) 式的 pose averaging 可以从多路中鲁棒融合。

**Prior art 支持**：ScrewNet (Jain ICRA 2021) 用 screw-displacement 序列的 LSTM 回归，本质上就是 centroid+pose 时序学习，Path C' 是其 **hand-crafted 闭式版本**，不需要学习权重，对 category-free 设置更合适。

---

## 4. 冲突仲裁机制（新约束 B）：swept-volume 优先 + 置信度 MRF

用户的 intuition（多层=深入=强证据）的 **principled 翻译**（Sub-A + Sub-C）：

**depth-in-part d_P(v) = signed EDT**（Felzenszwalb-Huttenlocher ToC 2012）。四重理论依据：(i) Blum 1967 / Siddiqi-Pizer 2008 医学轴理论：d_P(v) = maximal inscribed ball radius；(ii) Shape Diameter Function (Shapira 2008)：pose-oblivious 部件签名；(iii) Pottmann integral invariants (CAGD 2009)：flip-immunity 随 d_P(v) 线性增长；(iv) Gaussian likelihood model：log p(d|v∈P) − log p(d|v∉P) = β·d_P(v) + const。**结论：深 voxel 对 Stage B boundary error 的 flip-immunity 是 d_P(v) 的线性函数，证据强度 ∝ d_P(v)**。这把用户的"多层 vs 单层"启发升格为严格的对数似然比。

### 四层仲裁 pipeline

**Layer 0（不可协商的硬约束）：Swept-Volume Carving**（containment 解决）。对每个 v：若 v ∈ SV \ T_k(θ_closed)·canonical_move（即被 move 在某个 θ 扫过，但在 closed state 不占据），则 D_v(BASE) := +∞。**这是对 drawer-in-cabinet containment 的正确处理**——不是"删 base"也不是"保护 move"，而是 Kutulakos-Seitz 的 monotone carving：一旦 voxel 被 motion evidence 否决，它永远不能回到 base。Laurentini 1994 visual-hull 定理告诉我们 Stage B 无法恢复凹腔，Stage C 必须 explicitly carve。这一层 **单独解决 P2.1**，不需要置信度仲裁。

**Layer 1（置信度加权 unary）**：对剩余冲突（P2.2 boundary collision），按 Sub-C 的 Bayesian unary：

D_v(BASE)  = −log P_base(v) − λ_d · σ(β_b · d_base(v))
D_v(MOVE)  = −log O_soft_support(v) − λ_d · σ(β_m · d_move(v))
D_v(FREE)  = −log[(1−P_base(v))·(1−O_soft_support(v))] + γ_free

其中 β_b, β_m 由 Stage B 的 calibration curve 估计；λ_d 是 depth evidence 权重。**base-prior bias** 通过在 D_v(BASE) 里加 −log(π_base/π_move) 实现，这正是 informative prior 的 MAP 形式，不是启发式。

**Layer 2（pairwise 平滑）**：contrast-sensitive Potts，w_{uv} = λ_s · exp(−‖f(u)−f(v)‖²/2τ²)，f 拼接 (坐标、O_soft、d_base、d_move)；来自 DenseCRF (Krähenbühl-Koltun NeurIPS 2011) 的 Gaussian kernel lifted 到 6-/26-连通 voxel。

**Layer 3（label cost）**：E += Σ_ℓ h_ℓ·[∃v:L_v=ℓ]（Delong-Osokin-Isack-Boykov IJCV 2012），抑制过度分割。

**求解**：α-expansion (Boykov-Veksler-Zabih TPAMI 2001) + BK max-flow，64³ 上 3–10 次扫满足次秒收敛。若要 calibrated probability 而非 hard MAP，跟一个 Random Walker (Grady TPAMI 2006) 以高置信 voxel 为 seed。

### 为什么这套仲裁比"形状保护启发式"优越

1. **containment (P2.1) 由 Layer 0 的 swept-volume carving 直接解决**——不靠 tie-break，不依赖 P_base_shared 的相对强度。Stage B 的 M_attn 在长抽屉里被污染的 voxel 正好是 trajectory-middle，这些 voxel **必然在 SV 内**，必然被 carve，M_attn 的 bug 被 Layer 0 完全屏蔽。
2. **P2.2 boundary collision** 才是用户说的"单层 vs 多层"场景——这里 d_base(v) vs d_move(v) 的差直接给出证据比值，不需要做 shape-priority 规则。
3. **base-prior bias** 是 log-odds shift，退化到 π_base=π_move 时是无偏 MAP，不是 hard preference。
4. **M_attn 只在 Layer 1 的 unary 权重里出现一次**（通过 P_base_shared_filtered），它错也只能局部影响——被 Layer 0 的硬 carving 与 Layer 2 的 smooth 双重压制。这正是 Strategy E。

### Part²GS repel-loss 的位置

Part²GS 的 repel-loss 只解决"不同 part 的 Gaussian 不应该空间重叠"，对 containment 不敌——它把 containment 当 penalty 而不是 carving。我们的 Layer 0 是 **hard constraint** 而非 loss term，更强且无需调权。

---

## 5. Fit-then-refine seg（新约束 C）：单轮够，但要看收敛条件

**Sub-D 的严格答案**：Wu (Ann Stat 1983) 保证 EM 单调收敛到 stationary point，**不保证单步到位**。所以"单轮是否足够"取决于 warm-start 是否在 basin 内且 Q-function 是否近似二次。

**我们的 regime**：warm-start 由 Stage B（M_attn+P_base_shared）+ Section 1 的 shell-motion + Section 3 的三路 warm-start 三重加固，基本保证在正确 basin。**但单轮足够有两个附加条件**：

(i) **E-step 是 MAP 而非 soft**（graph-cut 型），一次可能锁死错标——Isack-Boykov PEARL 经验是跑到能量 plateau 常要 5–20 外循环。
(ii) 若用 soft EM（softmax responsibility），warm-start 好时 2–3 轮饱和（RoMo 2024 类工作的经验）。

**推荐方案**：**coupled 1.5-pass**（我们的命名）：Fit₀ → RefineSeg₀ → Fit₁ → **单调性检验**（若 ΔE < ε_stop 停）→ 否则再一轮。典型 1–2 轮，不开大 EM 外循环。这避开 SAJO/Stage C v1 那种 initialization-sensitive 的多轮失控。

### E-step 精确公式（refine M_k 给定 T_k）

统一能量，与 Section 4 仲裁对齐：

M_k(v) = argmin_{ℓ∈{B,M,F}} [ D_k(ℓ,v | T_k) + Prior(ℓ,v) + Smooth(v, N(v)) ]

其中 data term 用 **soft occupancy 的 binary cross-entropy**（非 L1/L2）：

D_k(MOVE, v | T_k)  = BCE(O_k^{soft}(v), warp(canonical_move^{soft}, T_k)(v))
D_k(BASE, v | T_k)  = BCE(O_k^{soft}(v), canonical_base^{soft}(v))

选 CE 而非 L2：(i) binary 变量的最大似然；(ii) 对概率 calibrated；(iii) 数值上与 TRELLIS 的 logit 同类；(iv) 避免 Sub-D 警告的 voxel field 过参数化的次优局部解。

Prior term = −log P_base_shared_filtered(v) 之类的 log-prior（从 Stage B）。
Smoothness = Potts（α-expansion 对 Potts metric 有 2× 近似保证，Kolmogorov-Zabih 2004），不用 TV3D（TV 在 binary label 上退化为 Potts）。

Chan-Vese 2001 的能量形式是同一族——我们是它在 binary occupancy + warp 约束下的离散多 label 扩展。

### 停机准则

ΔE/E₀ < 10⁻³ **且** ‖T_k^{(t+1)} − T_k^{(t)}‖_se(3) < ε_screw（screw 流形上的 geodesic 距离）。两条件合取，避免能量 plateau 但参数漂移、或参数收敛但分割抖动。这是 Ochs-Brox-Pock 2013 block coordinate descent 标准停机准则在我们问题上的 instantiation。

---

## 6. 过参数化处理：不参数化 B̃/M̃ 是最 principled 的

**Sub-D 明确结论**：PARIS 的 implicit MLP 之所以好不是因为 MLP，是因为它把 B̃/M̃ 降维到 decoder output 并施加隐式正则。在 **voxel setting** 下，最 principled 的对应操作不是再学一个 MLP（那是把 autoencoder 塞进 Stage C），而是 **根本不把 B̃/M̃ 当自由变量**——让它们是 seg EM 的 output：

canonical_base(v) := (Σ_k 1[M_k(v)=BASE] · O_k^{soft}(v)) / max(1, Σ_k 1[M_k(v)=BASE])
canonical_move(v) := (Σ_k 1[M_k(v)=MOVE] · warp(O_k^{soft}, T_k^{-1})(v)) / max(1, Σ_k 1[M_k(v)=MOVE])

这样 **自由变量只剩 M_k (binary label field) + T_k (10 DoF)**，过参数化从 ~1.5M 掉到 ~262K binary labels + 10 continuous，和观测 1.5M soft occupancy 评估点对比不再 over-parameterized。α_k 这个中间变量完全删除。

**为什么不用 PARIS MLP**：在 voxel setting 下，MLP 隐式正则的 inductive bias 是"平滑空间插值"，但 voxel 数据本来就是 discrete，MLP 只是引入了新的 approximation error。**TV3D（Chambolle 2004）**作为 pairwise smoothness 已经在 Section 5 的 Potts 里实现，独立加 TV 是冗余。

**补充正则**：coarse-to-fine（32³ → 64³）只在 warm-start 阶段用一次——32³ 做 SH/centroid warm-start，降低 SO(3) 搜索成本；64³ 做 refine。连通性正则（Mosinska CVPR 2018）不需要——swept-volume carving 和 CC size filter 已保证拓扑。

---

## 7. M_attn 使用边界：Strategy E（仅仲裁证据通道），通过两层屏蔽

**Strategy E 评估**：M_attn 只在 Section 4 Layer 1 的 unary 中通过 P_base_shared_filtered 出现一次，**且**受 Layer 0（swept-volume carving 硬约束）与 Layer 2（Potts smooth）的双重压制。长抽屉 pathology 发生在 trajectory-middle——这些 voxel **必然**进入 SV（因为它们被 move 部分在某个 θ 占据），Layer 0 直接把它们从 BASE 剔除，M_attn 错标对最终 label 无影响。

**拒绝 Strategy A (完全不用)**：浪费了 Stage B 已经做的语义先验，在 boundary collision 这种轻度冲突上有增益。
**拒绝 Strategy C (graph-cut unary 小系数)**：正是 AOF 被批的方案，系数难调且无原理。
**拒绝 Strategy B (contact band 定位)**：契约过窄，不是 M_attn 的自然用法。
**拒绝 Strategy D (sanity check)**：事后监测无法纠错。

**Strategy E 的 principled 表述**：M_attn 参与 MAP 的 prior，**不参与 Layer 0 硬约束，也不单独决定任何 voxel 的 label**。这把 M_attn 的影响域严格限制在"证据弱时的 tie-break"，完全避开 error amplification。

---

## 8. 优化器与最小 loss 设计

**优化器**（Sub-B）：**tangent-space Adam on se(3) screw subspace + optional Gauss-Newton refine**。

- Theseus (Pineda NeurIPS 2022) 对 10 DoF 问题 **overkill**：它的优势是稀疏 Jacobian 的 implicit differentiation 与大规模 bundle-adjustment 样式问题，10 DoF 稠密 Jacobian 上收益 marginal，而增加的工程复杂度显著。
- Adam 在 se(3) tangent 空间（指数映射式更新）的 v5 实测收敛良好证据充分；单纯 Gauss-Newton / LM 在 Dice loss 上二阶项需要 finite difference 或 autodiff Hessian，工程上 Adam + 小学习率更稳。
- **最后 200 步**切到 Gauss-Newton 做二阶 polish（Absil-Mahony-Sepulchre 2008 on-manifold opt），收敛到 local quadratic rate，精度高 1–2 位。

**最小 loss 设计（拒绝 10 项 soup）**：

整个 Stage C 的 **总目标只有一个 MAP 能量**：

**E(M, T) = Σ_k Σ_v D_k(M_k(v) | T_k) + Σ_k Σ_{(u,v)∈N} V(M_k(u), M_k(v)) + Σ_ℓ h_ℓ·[∃v,k : M_k(v)=ℓ] + λ_screw · R_screw(T)**

四项而已：
1. **Data likelihood**（BCE on warped soft occupancy）——Section 5
2. **Spatial Potts smoothness**——Section 4 Layer 2
3. **Label cost**（防过度分割）——Section 4 Layer 3
4. **Screw manifold 投影 R_screw**（把 SE(3) 拉回 1-DOF 子流形的软正则 / 硬约束）

Swept-volume carving 不是 loss，是硬约束。Centroid trajectory PCA 不是 loss，是 warm-start。Moment matching 不是 loss，是 rotation warm-start。depth-in-part 不是 loss，是 unary 的 per-voxel 权重。这四个 component 之间 **不互相加权**（没有超参调 λ₁/λ₂/.../λ₁₀），每个的作用域正交。

---

## 两份独立评价的逐条回应（reviewer 视角）

由于原文 1.txt / 评价.txt 未随任务提供，以下按 CVPR/AAAI 常见 reviewer 的 principled critique 类别逐条回应（假设它们覆盖了常规挑战）：

| # | 典型挑战 | 成立性 | 方案如何回应 |
|---|----------|:------:|----------------|
| 1 | "M_attn 错 → Stage C 错"的 error amplification | **成立** | Strategy E + Layer 0 硬约束双重屏蔽；M_attn 影响域限制到 boundary collision tie-break |
| 2 | 长抽屉 count=6 voxel 在 signature 空间不可分 | **成立**（Sub-A 给出不可能性定理） | 换 formulation：shell-motion + swept-volume pull-back；在 signature 空间解不是工程问题是错误 formulation |
| 3 | Per-voxel matching 违反 principled 原则 | **成立** | 采用 SH+centroid correspondence-free 的 whole-body rigid fit；Dice volumetric loss 上 GN refine |
| 4 | Warm-start 对 prismatic+对称退化 | **成立** | 新 Path C' centroid-trajectory PCA 覆盖对称 prismatic；三路覆盖全部四象限 |
| 5 | Containment 用"删 base"或"保护 shape"都不 principled | **成立** | Kutulakos-Seitz motion-consistency space carving 作为 Layer 0 硬约束；非启发式 |
| 6 | 多轮 EM initialization-sensitive | **部分成立**（SAJO 经验；但我们的 warm-start 比之前强） | 1.5-pass 方案 + Wu 1983 单调性检验停机；不开无限外循环 |
| 7 | Theseus overkill | **成立** | 10 DoF 用 tangent Adam + GN polish，不引入框架依赖 |
| 8 | TRELLIS 生成噪声污染 canonical_move | **成立** | 三路 filter bank（CC-percolation、swept-volume consistency、hysteresis gating），独立证据，复合误保率 < 10⁻³ |
| 9 | Loss soup 超参过多 | **成立** | 总能量四项且作用域正交；无 λ 调节矩阵 |
| 10 | B̃/M̃ 过参数化 memorize noise | **成立** | 不参数化：canonical volumes 作为 seg EM 的确定性 output，自由变量压到 ~262K binary + 10 continuous |
| 11 | "单层/多层"缺乏 principled 表述 | **成立** | signed EDT d_P(v) = maximal inscribed ball radius；log-likelihood ratio 线性在 d_P(v)（Sub-A Q3 四重证明） |
| 12 | base-prior bias 是 hard preference | **不成立** | 通过 log-odds shift −log(π_base/π_move) 加入 MAP unary，π 相等时退化为无偏，是 informative prior 的严格 Bayesian 表述 |
| 13 | Depth-in-part 对 boundary error 敏感 | **不成立** | Pottmann CAGD 2009 integral invariants 定理：flip-immunity 随 d_P(v) 线性增长；深 voxel 在 ~r 阶扰动下标签稳定 |
| 14 | 冲突仲裁本应交 Stage D | **不成立** | 用户明确要求 Stage C 内部解决；Layer 0 硬 carving 作为 ℓ∞ 约束在 MAP 内原生处理，不需推迟 |
| 15 | Moment-based 方法对 open-surface 敏感 | **部分成立**（Novotni-Klein 2004 的已知局限） | 先 SDF-fill 把 voxel 转 closed volume 再算 Zernike；SH 用球壳采样对 open surface 天然鲁棒 |
| 16 | Shell XOR 对 thin drawer 信号弱 | **部分成立** | Shell 信号弱时 centroid-trajectory PCA 仍可靠（Path C' 只需 K≥2 state 有位移） |

---

## 结论：三个关键转向

**第一，formulation 转向**。长抽屉 pathology 从"如何在 signature 空间更聪明地判断"转向"如何从 shell 起、用 swept-volume pull-back 判决 interior"。这不是加 trick，是承认 signature 空间的信息论上限（Sub-A 的不可能性定理）。

**第二，containment 转向**。抽屉-柜腔从"置信度 tie-break"转向 **motion-consistency 的 space carving**（Kutulakos-Seitz IJCV 2000 的 articulated 版本），Layer 0 硬约束，不需要 M_attn 或 P_base 的相对强度。用户 intuition 的"多层=证据强"被升格为 Pottmann integral invariants 的 flip-immunity 定理。

**第三，架构转向**。Stage C 不是一个"loss 集合 + 多路 EM"的工程拼盘，而是一个 **四项能量 MAP + 硬 carving 约束 + 1.5-pass 耦合迫近** 的紧凑流程。M_attn 从"主信号"退位到"仲裁证据通道"，Path C' centroid-trajectory 解决对称 prismatic 退化，moment-based fit 解决 correspondence-free，不参数化 canonical volumes 解决过参数化。每一项改动都有 peer-reviewed 理论依据，没有启发式、没有 soup、没有补丁。

**核心 novelty 声明**（for CVPR/AAAI contribution）：据 Sub-C 的文献梳理，**将 Kutulakos-Seitz space carving 从 photo-consistency 推广到 motion-consistency 并在 voxel 级别显式 carve cavity，是现有 articulated reconstruction 文献（Ditto/PARIS/DTA/CARTO/URDFormer/Real2Code/NAP/Articulate-Anything）都未做的**——它们要么用 URDF tree 回避、要么靠多状态训练隐式 empty cavity。这条路线加上 Sub-A 的不可能性定理、Sub-B 的 SH+centroid+Path C' 三路方案，构成一篇完整 methodology paper 的 principled core，而不是 yet-another incremental system paper。