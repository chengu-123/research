# pipeline.md — ArtRASA-v8: StageB/BMCSA-informed Staged RFSDS Pipeline

**目标**：单张闭态图 `I0` → URDF-ready articulated 3D asset  
**版本**：ArtRASA-v8 / AAAI-oriented pipeline  
**核心原则**：

1. 不使用 SAM2 / Particulate / GIM-DKM / 外部 learned articulation predictor 作为主线。
2. 旧 StageB / BMCSA 不是废弃项，而是 **topology prior generator**。
3. 最终优化对象只有一个 canonical articulated asset：`B`、`M`、`T(q_k)`、canonical SLAT。
4. 不把所有 loss 混在一个大 `L_total` 中；采用 staged dual-path RFSDS。
5. TRELLIS / Wan2.2 / DINOv2 均冻结，只优化新增层与显式结构变量。
6. Wan2.2 pseudo-state frames 是 **conditions / priors**，不是真实多状态观测，也不是 GT。

---

## 0. One-line thesis

> Given a single closed-state image, we first use Wan2.2 to synthesize K fixed-camera pseudo-state frames, then use the old StageB/BMCSA pipeline only as a no-gradient interior-topology prior generator. The final optimization maintains a single canonical articulated representation, decomposed into a static base and a movable part. High/mid-noise W-RFSDS optimizes geometry, part masks, and single-DoF trajectory; low-noise W-RFSDS later optimizes state-consistent SLAT donor fusion and texture. Base consistency is guaranteed by construction, not by alignment loss.

---

## 1. Research objective

### 1.1 Input

```text
I0: single closed-state RGB image
prompt: locked-camera articulation prompt
joint prior: prismatic or revolute, preferably specified by text/prompt for MVP
```

### 1.2 Output

```text
canonical_base.glb
canonical_move.glb
joint.json
object.urdf
optimized_video.mp4
diagnostics/
  support_report.json
  part_mask.npy
  contact_anchor.npy
  texture_provenance.json
```

### 1.3 Scope

```text
single object
single-DoF articulation
prismatic / revolute
single closed input image
training-free with respect to articulated 3D datasets
per-instance test-time optimization
```

Multi-joint and non-rigid articulation are future work.

---

## 2. Papers / systems borrowed and how

### 2.1 TRELLIS

**Borrowed idea**:

- Two-stage generation: sparse structure → SLAT.
- SLAT is a structured local latent that supports mesh, 3D Gaussian, and radiance-field decoding.
- Local manipulation / asset variants imply that TRELLIS internal latent space has spatial locality and editable region structure.

**Used in ArtRASA**:

```text
SS stage:
  canonical support / base-move geometry / part mask

SLAT stage:
  canonical texture / exposed surface features / donor fusion
```

### 2.2 Wan2.2 I2V-A14B

**Borrowed idea**:

- Frozen image-to-video model as motion and temporal prior.
- High-noise expert is better aligned with coarse layout / motion.
- Low-noise expert is better aligned with texture / detail.

**Used in ArtRASA**:

```text
Phase G:
  high/mid-noise W-RFSDS for geometry and trajectory

Phase T:
  low-noise W-RFSDS for SLAT texture and detail
```

### 2.3 CHORD / RFSDS

**Borrowed idea**:

- Frozen rectified-flow video model supplies a velocity residual.
- The residual is backpropagated through rendered video into a 3D/4D representation.

**Used in ArtRASA**:

```text
Geometry RFSDS:
  rendered gray/depth/normal video → Wan2.2 residual → voxel/part/joint

Texture RFSDS:
  rendered RGB video → Wan2.2 residual → SLAT-RASA/texture adapter
```

### 2.4 MorphAny3D

**Borrowed idea**:

- SLAT latent-space attention fusion, especially MCA and TFSA.
- The useful part is not 2D texture back-projection, but attention-level feature mixing inside SLAT-DiT.

**Used in ArtRASA**:

```text
K-way SLAT donor fusion:
  each donor is sampled in its own state-consistent coords
  donors are canonicalized into one canonical SLAT
```

### 2.5 FreeArt3D

**Borrowed idea**:

- Per-instance, training-free articulated 3D optimization.
- Stage-wise decomposition is preferable to one loss soup.
- Classical initialization of joint / motion is useful.
- Only a small number of active losses should be used at any stage.

**Difference**:

```text
FreeArt3D:
  uses real multi-state images

ArtRASA:
  uses one closed image + Wan2.2 pseudo-state conditions
  uses old StageB/BMCSA as topology prior
```

### 2.6 NANO3D

**Borrowed idea only**:

- Region-aware preservation: edited region changes, unedited region remains stable.
- Voxel/SLAT merge philosophy motivates base preservation.

**Not used directly**:

```text
No post-hoc Nano3D-style merge as final geometry.
Base consistency is enforced by single-canonical representation.
```

### 2.7 Old StageB / BMCSA from our repo

**Core role**:

```text
Old StageB/BMCSA is not final generator.
It is a no-gradient topology prior generator.
```

It provides:

```text
BMCSA_s0 support
open-state union support
soft occupancy/logit evidence
coarse move candidate
coarse joint warm-start
optional hidden features for weak distillation
```

It does not provide:

```text
final K-state geometry
hard voxel outputs
final base/move segmentation
final joint
```

---

## 3. Final architecture overview

```text
Input I0
  ↓
Wan2.2 I2V fixed-camera generation
  ↓
K pseudo-state frames I0...I5
  ↓
Old StageB/BMCSA no-grad pre-pass
  ↓
Interior-aware SUPPORT_COORDS + coarse joint init
  ↓
PartHead bootstrap
  ↓
Geometry / Voxel / Joint RFSDS path
  ↓
Contact-anchored joint polish
  ↓
SLAT / Texture RFSDS path
  ↓
Mesh + texture + URDF export
```

---

## 4. Phase 0 — Wan2.2 pseudo-state generation

### 4.1 Purpose

Generate K pseudo-state frames to expose motion and hidden surfaces that are not visible in the closed image.

### 4.2 Input

```text
I0
locked-camera prompt
```

### 4.3 Prompt template

```text
Positive:
A locked-off single-camera video. The first frame is exactly the provided image.
The camera is static, with no zoom, no pan, and no orbit.
Only the movable part opens with a physically plausible single-DoF motion.
The object body remains stationary.
Material, color, lighting, and texture remain consistent across frames.

Negative:
camera movement, zoom, pan, orbit,
object morphing, identity drift,
new object, background change,
texture flicker, floating parts,
non-rigid deformation
```

### 4.4 Output

```text
I0, I1, ..., I5
cond_K = DINOv2(I0...I5)
```

### 4.5 Important note

These frames are **pseudo-state conditions**, not ground-truth observations. They must never be described as real multi-state input images.

---

## 5. Phase 1 — Old StageB/BMCSA topology pre-pass

### 5.1 Purpose

Use the old StageB/BMCSA pipeline to discover hidden topology, especially:

```text
cabinet cavity
drawer interior
door back side
base inner wall
open-state-only surfaces
```

### 5.2 Execution

Run old StageB / BMCSA once under `torch.no_grad()`:

```text
I0...I5
  → K-batched SS-DiT / SCAR / BMCSA / SDEdit-style pipeline
  → O_stack_soft
  → BMCSA_s0
  → open-state union
  → hidden feature evidence
```

### 5.3 Cached artifacts

```text
bmcsa_O_stack_soft.npy
bmcsa_s0_soft.npy
bmcsa_open_union.npy
bmcsa_move_prior.npy
bmcsa_hidden.pt
coarse_motion_report.json
```

### 5.4 What is kept

```text
soft occupancy
logits
union support
coarse move candidate
coarse joint initialization
optional soft feature prior
```

### 5.5 What is discarded

```text
K hard voxel outputs
K final state geometries
post-hoc StageC final geometry
```

### 5.6 Rationale

Old StageB/BMCSA has two sides:

```text
Useful:
  can discover hidden topology via cross-state evidence

Harmful:
  K independent hard voxel outputs suffer from argwhere / threshold jitter
```

ArtRASA keeps the useful soft topology evidence and discards the harmful discrete outputs.

---

## 6. Phase 2 — Interior-aware support and coarse joint initialization

### 6.1 Interior-aware support

Construct the canonical optimization canvas:

```text
SUPPORT_COORDS =
    BMCSA_s0_support
  ∪ open_state_union_support
  ∪ inverse_warped_move_support
  ∪ motion_sweep_support
  ∪ dilation
  ∪ topK_uncertain
```

Pseudo-code:

```python
S0 = bmcsa_s0_soft > tau_s0
Sopen = max_k(bmcsa_O_stack_soft[k]) > tau_open
Smove = inverse_warp_union(move_candidates, T_k_coarse)
Ssweep = motion_sweep(move_candidates, T_k_coarse, n=20)
Suncertain = topK_uncertain(bmcsa_logits, K=2000)

SUPPORT_COORDS = S0 | Sopen | Smove | Ssweep | dilation(S0 | Sopen, r=1) | Suncertain
```

Recommended defaults:

```text
tau_s0 = 0.4
tau_open = 0.4
dilation radius = 1
topK_uncertain = 2000
motion sweep samples = 20
```

### 6.2 Support confidence

Each support voxel gets a confidence label:

```text
high confidence:
  BMCSA_s0 stable support
  closed-visible exterior

medium confidence:
  open-state union
  inverse-warped move support

low confidence:
  dilation
  topK uncertain
  motion sweep boundary
```

Low-confidence voxels initialize as `uncertain`.

### 6.3 Coarse joint initialization

No RFSDS here. Use classical geometry:

```text
BMCSA O_stack_soft
  → stable base / move candidate
  → connected components
  → centroid trajectory
  → PCA / inertia / least-squares
  → T_k_coarse
```

Output:

```text
axis_init
pivot_init
q_k_init
joint_type_init
```

### 6.4 Why not random init

RFSDS is not a reliable 3D joint solver from random initialization. It should refine a plausible joint, not discover one from zero.

---

## 7. Phase 3 — PartHead bootstrap

### 7.1 Trainable variables

```text
PartHead
support gate
```

### 7.2 Frozen variables

```text
SS-RASA
JointHead
SLAT-RASA
texture adapter
Wan2.2
TRELLIS
DINOv2
```

### 7.3 Loss

\[
L_{boot}
=
\lambda_d L_{distill}
+
\lambda_p L_{part-simple}
\]

where:

```text
L_distill:
  soft BCE / KL with BMCSA move prior

L_part-simple:
  move volume ratio + weak overlap
```

### 7.4 Recommended settings

```text
iters = 100–300
lambda_distill = 0.03–0.05
soft target only
no hard pseudo-label
```

### 7.5 Purpose

Give PartHead a good start. Do not let BMCSA bias dominate final masks.

---

## 8. Phase 4 — Geometry / Voxel / Part RFSDS path

### 8.1 Purpose

Optimize:

```text
canonical occupancy
support gate
base / move split
coarse motion plausibility
```

Do **not** optimize texture.

### 8.2 Trainable variables

```text
SS-RASA
PartHead
support gate
boundary head
```

### 8.3 Frozen / low-LR variables

```text
SLAT-RASA: frozen
texture adapter: frozen
donor weights: frozen
JointHead: fixed at T_k_coarse or very low LR
```

### 8.4 Render type

Use geometry-only rendering:

```text
silhouette
depth
normal
gray albedo
soft volume / Gaussian occupancy proxy
```

### 8.5 Objective

\[
L_{geom}
=
L^{geo}_{W\text{-}RFSDS}
+
\lambda_{id}L^{geo}_{first}
+
\lambda_{sup}L_{support}
+
\lambda_{part}L_{part-simple}
\]

### 8.6 Explanation

- `L_RFSDS_geo`: high/mid-noise Wan2.2 W-RFSDS.
- `L_first_geo`: first-frame silhouette / mask / feature anchor.
- `L_support`: support sparsity and uncertain voxel suppression.
- `L_part_simple`: prevents empty move or all-move collapse.

### 8.7 Variables not allowed to move

```text
SLAT texture
color residual
texture donor weights
full RGB appearance
```

This prevents texture shortcuts from absorbing geometry errors.

---

## 9. Phase 5 — Joint / Contact polishing path

### 9.1 Purpose

Refine the joint after geometry is mostly stable.

### 9.2 Trainable variables

```text
JointHead
axis
pivot / origin
q_k
optional move boundary low LR
```

### 9.3 Frozen variables

```text
SS-RASA
PartHead core
support high-confidence core
SLAT-RASA
texture adapter
```

### 9.4 Objective

\[
L_{joint}
=
\eta_r L^{weak}_{W\text{-}RFSDS}
+
\eta_c L_{contact-bundle}
+
\eta_o L_{collision}
\]

### 9.5 Contact bundle

\[
L_{contact-bundle}
=
L_{contact-mass}
+
L_{axis-anchor}
+
L_{pivot-near}
+
L_{axis-dir}
\]

This is one physical prior bundle, not four independent paper losses.

### 9.6 Contact field

Let:

\[
P_B(x)=P_{occ}(x)P_{base}(x)
\]

\[
P_M(x)=P_{occ}(x)P_{move}(x)
\]

Define:

\[
C(x)
=
\exp
\left(
-\frac{d_B(x)^2+d_M(x)^2}{\sigma_c^2}
\right)
\]

Contact existence:

\[
L_{contact-mass}
=
\max(0,\tau_m-\sum_x C(x))^2
\]

### 9.7 Anchor set

Every N iterations:

```python
with torch.no_grad():
    A = topM(C)
    A = largest_connected_component(A)
    anchors = farthest_point_sample(A, M_anchor)
```

No gradient through connected components.

### 9.8 Revolute joint anchor

Axis line:

\[
\mathcal{L}(o,a)=\{o+\lambda a\}
\]

Point-to-line distance:

\[
d_{line}(p;o,a)=\|(p-o)\times a\|_2
\]

Axis-anchor loss:

\[
L_{axis-anchor}
=
\operatorname{softmin}_{p\in A}
d_{line}(p;o,a)^2
\]

Pivot-near:

\[
L_{pivot-near}
=
\min_{p\in A}\|o-p\|^2
\]

Axis direction:

\[
L_{axis-dir}
=
1-(a^\top u_1)^2
\]

where \(u_1\) is the PCA principal direction of the contact anchor strip.

### 9.9 Prismatic joint prior

For prismatic joints:

\[
L_{pris-dir}
=
1-(a^\top \hat d_{motion})^2
\]

Collision:

\[
L_{collision}
=
\sum_{k,x}
P_B(x)P_M(T_k^{-1}x)
\]

---

## 10. Phase 6 — SLAT / Texture RFSDS path

### 10.1 Purpose

Optimize:

```text
texture
SLAT donor fusion
newly exposed surfaces
material consistency
appearance detail
```

Geometry and joint are frozen.

### 10.2 Trainable variables

```text
SLAT-RASA
texture adapter
donor weights
SLAT residual
color residual
Gaussian color / opacity residual
```

### 10.3 Frozen variables

```text
B, M geometry
PartHead
JointHead
support gate core
SS-RASA
```

### 10.4 State-consistent SLAT donor sampling

Use Option A:

```python
for k in range(K):
    coords_k = B ∪ T_k(M)
    s_k = SLAT_DiT(cond=I_k, coords=coords_k)
```

### 10.5 Canonicalization

For base:

\[
s_B^*(p)
=
\sum_k w_{k,p}^{B}s_k(p)
\]

For move:

\[
p_k=T_k(p)
\]

\[
\tilde{s}_k(p)=\operatorname{Lookup}(s_k,p_k)
\]

\[
s_M^*(p)
=
\sum_k w_{k,p}^{M}\tilde{s}_k(p)
\]

Final:

```text
one canonical SLAT
one base texture
one move texture
```

No K state-specific final texture.

### 10.6 SLAT-RASA / MorphAny3D-style fusion

K-way MCA:

\[
h^{MCA}_{k,i}
=
\sum_j w_{i,j}
\operatorname{CrossAttn}(Q=h_{k,i},K/V=cond(I_j))
\]

K-way TFSA:

\[
h^{TFSA}_{k,i}
=
\sum_j \tilde w_{i,j}
\operatorname{SelfAttn}(Q=h_{k,i},K/V=cache_j)
\]

Donor weights use:

```text
visibility
view angle
part compatibility
frame confidence
state distance
RFSDS residual reliability
```

### 10.7 Objective

\[
L_{tex}
=
L^{tex}_{W\text{-}RFSDS}
+
\mu_{id}L^{rgb}_{first}
+
\mu_{can}L_{canonical-SLAT}
\]

- `L_RFSDS_tex`: low-noise RFSDS, texture/detail.
- `L_first_rgb`: visible exterior RGB anchor.
- `L_canonical-SLAT`: donor consistency after canonicalization.

---

## 11. Phase 7 — Export

No training.

Steps:

```text
1. hard threshold occupancy
2. hard threshold base / move
3. connected component cleanup
4. decode canonical_base mesh
5. decode canonical_move mesh
6. bake texture from canonical SLAT
7. export joint.json
8. write URDF
9. run PyBullet / MuJoCo simulation diagnostics
```

Use mesh / FlexiCubes for final export. Use Gaussian / soft render during optimization.

---

## 12. Final loss budget

| Phase | Trainable variables | Active losses | Count |
|---|---|---|---:|
| Wan + BMCSA pre-pass | none | none | 0 |
| support / joint init | classical only | least-squares / PCA | 0 |
| PartHead bootstrap | PartHead, support gate | distill, part-simple | 2 |
| geometry RFSDS | SS-RASA, PartHead, support gate | RFSDS_geo, first_geo, support, part-simple | 4 |
| joint polish | JointHead / axis / pivot / q | weak RFSDS, contact_bundle, collision | 3 |
| SLAT texture | SLAT-RASA, texture, donor weights | RFSDS_tex, first_rgb, canonical_SLAT | 3 |
| export | none | none | 0 |

---

## 13. Main contributions

### C1 — StageB/BMCSA-derived interior topology prior

Old StageB/BMCSA discovers hidden topology but produces jitter-prone K-state discrete outputs. ArtRASA-v8 retains its soft topology evidence as canonical support and discards hard K-state geometry.

### C2 — Single-canonical staged RFSDS

The final optimization maintains one canonical base and one canonical movable part. All states are derived by analytic single-DoF transform, guaranteeing base consistency by construction.

### C3 — State-consistent SLAT donor canonicalization

Each SLAT donor is sampled in matched state coordinates and then canonicalized into one SLAT. This avoids condition-coordinate mismatch and preserves physical texture consistency.

### C4 — Contact-anchored URDF joint

Base–move contact anchors constrain revolute axis / pivot and prismatic direction, improving physical validity of exported URDF joints.

### C5 — Dual-path Wan2.2 W-RFSDS

High/mid-noise RFSDS optimizes geometry and trajectory; low-noise RFSDS optimizes SLAT texture after geometry is fixed.

---

## 14. Ablation matrix

Essential ablations:

```text
A1: no BMCSA support, I0-only support
A2: BMCSA support only
A3: BMCSA support + weak distill
A4: staged optimization vs single-loop loss soup
A5: with / without phantom-KV
A6: velocity correction vs support-gated RASA
A7: Option A SLAT vs Option B SLAT
A8: no contact-anchor
A9: no classical coarse joint init
A10: geometry RFSDS only vs geometry+texture RFSDS
A11: K=2 vs K=6 pseudo states
A12: no old StageB/BMCSA pre-pass
```

Critical ablations:

```text
A1 / A12:
  prove StageB/BMCSA topology prior is necessary

A4:
  prove staged optimization is necessary

A7:
  prove state-consistent SLAT donor canonicalization is better

A8:
  prove contact-anchor improves URDF joint validity
```

---

## 15. Safe claims and unsafe claims

### Safe claims

```text
Wan2.2 pseudo-state frames are conditions, not GT observations.
TRELLIS and Wan2.2 are frozen.
We optimize only inserted modules and explicit structure variables.
Old StageB/BMCSA is reused only as a topology prior generator.
K hard voxel outputs from StageB are not used as final geometry.
Argwhere is not made differentiable; fixed support + differentiable gate bypasses it.
K states are derived from a single canonical articulated representation.
Geometry and texture RFSDS are separated to reduce credit assignment ambiguity.
```

### Unsafe claims

```text
Wan2.2 gives segmentation labels.
SAM2 / external predictors solve part segmentation.
MorphAny3D proves articulated K-state SLAT fusion.
Argwhere is differentiable.
The method recovers true hidden geometry.
The method solves arbitrary multi-joint articulation.
K Wan frames are real multi-state observations.
```

---

## 16. Implementation priority

### Week 1 minimal target

```text
Day 1:
  Run Wan2.2 pseudo-state generation.
  Run old StageB/BMCSA pre-pass.
  Build BMCSA-derived support.

Day 2:
  Implement support gate + PartHead bootstrap.
  Verify support contains interior voxel.

Day 3:
  Implement geometry RFSDS on gray/silhouette render.
  Verify gradient reaches SS-RASA / support gate.

Day 4:
  Implement coarse joint init from BMCSA soft voxels.
  Verify deterministic K geometry.

Day 5:
  Implement contact-anchor joint polish.

Day 6:
  Implement Option A SLAT donor canonicalization.

Day 7:
  Run 1-object full pipeline and compare:
    I0-only support vs BMCSA support
    staged vs single-loop
```

### Go / no-go tests

```text
G1: BMCSA support contains open-state interior voxel.
G2: geometry RFSDS can activate/deactivate support gate.
G3: joint polish reduces contact-anchor distance.
G4: Option A SLAT produces more stable texture than Option B.
G5: staged optimization is more stable than all-loss single loop.
```

---

## 17. Final recommendation

Adopt **ArtRASA-v8: StageB/BMCSA-informed staged RFSDS**.

Specifically:

```text
1. Remove SAM2 / Particulate / GIM-DKM from the main pipeline.
2. Keep old StageB/BMCSA as the central topology-prior contribution.
3. Keep K=6 Wan pseudo-state conditions.
4. Use BMCSA-derived interior-aware support.
5. Use classical coarse joint init from BMCSA soft voxels.
6. Demote phantom-KV to ablation.
7. Use staged optimization with ≤4 active losses per stage.
8. Use high/mid-noise RFSDS for geometry.
9. Use low-noise RFSDS for SLAT/texture.
10. Use contact-anchor physical prior for URDF joint validity.
11. Use Option A state-consistent SLAT donor canonicalization.
```

This version best preserves the originality of the old StageB/BMCSA line while making the optimization substantially more stable and easier to defend at AAAI.
