# Stage C v4 redesign plan

> Status: PROPOSAL, pending user confirmation. Do NOT edit method.md / pipeline.md / stageB.md until this plan is approved.
> Author: Claude (deep-research mode), 2026-05-25.
> Scope: replace the failing centroid-based Stage C with a cardinal-axis voxel-projection scheme.

---

## 0. One-paragraph summary

The current Stage C reads the 3D centroid trajectory of per-state move voxel sets V_k and runs a line-vs-arc fit on it. For drawer-with-occlusion the V_k set composition changes more than the centroid moves, so the centroid signal is degenerate; for revolute samples the centroid arc is contaminated by hallucination noise and yields non-cardinal continuous axes. The v4 redesign drops the centroid as the primary signal. It enumerates 6 cardinal directions for prismatic and 3 cardinal axes for revolute, and discriminates by the perpendicular-plane projection invariant: prismatic preserves the projection across states (rigid translation along the cardinal is in-plane invariant); revolute traces a 2D arc of projected centroids in the perpendicular plane. Phi is read off the leading edge along the candidate direction (prismatic) or the angular position of the projected 2D centroid (revolute), both of which survive per-state occlusion-induced shape changes.

---

## 1. Failure analysis (concrete)

### 1.1 Output evidence

`outputs/30857/stage_c/stage_c_joint_init.json` (ground truth: prismatic drawer):

```
joint_type:         "revolute"     (WRONG)
axis:               (0.555, 0.422, -0.717)   (non-cardinal, continuous)
residual_line:      1.86e-5
residual_arc:       1.28e-5        (arc wins by tiny overfit margin)
phi_0:              [-0.4, -0.2, 0, 0.2, 0.4, 0.6]    (linspace fallback)
delta_u_init:       [NaN, NaN, NaN, NaN, NaN]
axis_fit_source:    "centroid_circle"
```

7201 and 7128 (ground truth: revolute): type correct, but same phi linspace fallback and NaN delta_u_init, and axes are non-cardinal continuous values.

### 1.2 Root causes

- RC-1: line-vs-arc discriminator overfits arc in 3D (one more DoF) on nearly-collinear centroids; type_margin=0.15 too lax.
- RC-2: 3D centroid is dead signal for drawer-with-occlusion. V_k set composition (thin face slab in s_0 vs thick body in s_5) shifts the geometric centroid almost as much as the motion does, so the net centroid traversal can be < 1e-9.
- RC-3: code-state mismatch — v3 cardinal module exists at `mine/pipelines/stage_c/joint_type_detect.py` but production output shows `axis_fit_source: "centroid_circle"`, meaning the v2 legacy path actually ran. v3 never sets `arc_center/arc_normal/arc_radius`; v2 axis_fit gates on those non-None.
- RC-4: anchor band only contributes to confidence, not to axis selection as a hard constraint.

---

## 2. What Stage B v3.3.6 provides (the inputs)

Per stageB.md §3.1:

| Field | Shape | Role in v4 |
|---|---|---|
| `O_base_canonical` | (64,64,64) uint8 | hard subtractor + revolute-axis contact gate |
| `O_move_per_state` | (K=6,64,64,64) uint8 | hard move evidence (per state) |
| `P_base_canonical` | (64,64,64) float | UNUSED (Stage B doc §4.2.3: soft has multiplicative attenuation false-negatives at edges) |
| `P_move_evidence_per_state` | (K,64,64,64) float | soft union with hard for V_k_raw |
| `M_motion_corridor_64` | (64,64,64) float | anchor band source + swept-corridor whitelist |
| `is_carpet_mask` | (64^3,) bool | mandatory subtract |

K = 6 (`[0, 4, 8, 12, 16, 20]` from Stage A 21-frame video).

---

## 3. The discriminating physical signature

For each cardinal candidate axis `a ∈ {±X, ±Y, ±Z}`:

- **Prismatic along `a`**: rigid translation. The projection `Π_a(V_k)` onto the plane perpendicular to `a` is INVARIANT across k (drawer cross-section preserved). `IoU(Π_a(V_k), Π_a(V_j))` is high for all (k, j).
- **Revolute around `a`**: rotation in the perpendicular plane. The projection ROTATES. The 2D centroid `c_k = mean(Π_a(V_k))` traces a CIRCULAR ARC around the projected hinge. Voxel count `|V_k|` stays roughly constant across k (rigid body rotates).

Both signatures are read off the SHAPE of `Π_a(V_k)`, not its 3D centroid. They both survive per-state count/shape changes caused by occlusion (because the perpendicular projection ignores the motion direction for prismatic, and because the arc fit is in 2D and bounded by stable feature-corners for revolute).

---

## 4. Algorithm v4 — Cardinal Voxel-Projection Analysis

### Step 1 — Clean per-state move sets

```
For each k:
    M_k = (O_move_per_state[k] > 0) | (P_move_evidence_per_state[k] > tau_soft=0.15)
    M_k &= ~O_base_canonical_bool       # subtract hard base
    M_k &= ~is_carpet_3d                # subtract carpet
    V_k_raw = nonzero(M_k)              # (N_k, 3) int

# Temporal consistency filter
presence_count[v] = sum_k [v in V_k_raw]
keep_persistent = {v : presence_count[v] >= min_observers=2}
keep_corridor   = {v : M_motion_corridor_64[v] > tau_corridor=0.1}    # swept-corridor whitelist
keep_set        = keep_persistent UNION keep_corridor

V_k_clean = V_k_raw INTERSECT keep_set

# Mark states with too few voxels as invalid (interpolate later)
valid_mask[k] = (|V_k_clean| >= min_voxels=30)
V_union = union over valid k of V_k_clean
```

The corridor whitelist Q2 saves singleton voxels that fall inside the Stage B swept corridor (real drawer-back voxels that only appear in s_5).

### Step 2 — Per-axis perpendicular projection

For each `a ∈ {+X, +Y, +Z}` (axis name, unsigned):

```
For each valid k:
    # 2D mask in perpendicular plane (drop a-coord)
    Π_grid_k = scatter V_k_clean onto (64, 64) bool grid using the two non-a coords

    # 2D centroid in perp plane (world-unit coords)
    c_2d_k = mean(voxel_to_world(V_k_clean projected to non-a plane))

    # Extent ALONG a (for revolute stability check)
    e_k(a) = max(V_k · a_unit) - min(V_k · a_unit)
```

### Step 3 — Prismatic scoring (6 signed directions)

For each `d ∈ {+X, -X, +Y, -Y, +Z, -Z}`:

```
a = |d| (the cardinal name)
d_unit = signed cardinal unit vector

# (a) perpendicular-plane IoU (THE discriminator)
iou_perp = mean over pairs (k, j) of IoU(Π_a_grid_k, Π_a_grid_j)

# (b) front-edge advance along d (user's idea)
f_k = quantile(V_k_clean · d_unit, q=0.95)
total_advance = f_{K-1} - f_0
monotone_frac = #{k : f_{k+1} - f_k > tau_adv=0.008} / (K-1)

# (c) newly-appearing voxels direction (user's idea)
align_score = 0
for k in 0..K-2:
    new = V_{k+1}_clean \ V_k_clean
    if |new| >= min_new=20:
        offset = (centroid(new) - centroid(V_k_clean)) · d_unit
        align_score += clip(offset, -extent_scale, extent_scale)
align_score /= max(K-1, 1)

# (d) uniformity of advance
advances = [f_{k+1} - f_k for k in 0..K-2]
uniformity = 1 - clip(std(advances) / max(mean(advances), eps), 0, 1)

# composite
prismatic_score(d) =
    iou_perp
  * clip(total_advance / extent_scale=0.5, 0, 1)
  * monotone_frac
  * clip(align_score / extent_scale, 0, 1)
  * (0.5 + 0.5 * uniformity)
```

`best_pris = argmax_d prismatic_score(d)`.

### Step 4 — Revolute scoring (3 unsigned axes; sign comes from right-hand rule on data)

For each `a ∈ {+X, +Y, +Z}`:

```
# (a) voxel-count stability (rigid rotation preserves count)
n_k = |V_k_clean|
count_cv = std(n_k) / max(mean(n_k), 1)
count_stability = exp(-count_cv / 0.3)

# (b) extent-along-a stability
extent_cv = std(e_k(a)) / max(mean(e_k(a)), eps)
extent_stability_a = exp(-extent_cv / 0.3)

# (c) 2D arc fit on perp-plane centroids
(c2d_center, r, arc_residual) = pratt_circle_fit({c_2d_k : valid_mask[k]})
arc_quality = exp(-arc_residual / (0.025)**2)

# (d) hinge-contact compatibility — HARD constraint
anchor_band_voxels = (M_motion_corridor_64 > tau_corridor=0.1)
                   & (O_base_canonical > 0)
                   & ~is_carpet_3d
anchor_centroid_3d = mean(voxel_to_world(anchor_band_voxels))

# Build candidate 3D axis line: passes through (c2d_center lifted to 3D at
# anchor_centroid's a-coord) with direction a_unit
axis_line_point = lift_2d(c2d_center, a_coord=anchor_centroid_3d · a_unit, axis_name=a)

# Min distance from anchors to axis line
diffs = anchor_band_world - axis_line_point
perp  = diffs - (diffs · a_unit) * a_unit
dists = ||perp||
min_dist = dists.min()

if min_dist < contact_radius=4/64=0.0625:
    contact_compat = 1.0
else:
    contact_compat = exp(-(min_dist - contact_radius) / contact_radius)

# composite
revolute_score(a) = count_stability * extent_stability_a * arc_quality * contact_compat
```

Sign: determine right-hand-rule sign from cross-product of consecutive 2D centroid vectors around c2d_center; flip a_unit if needed so phi advances positive with k. `best_rev = argmax_a revolute_score(a)`.

### Step 5 — Type decision + dual-clone preparation

```
type_logit = log(best_pris.score / best_rev.score)
type_margin = 0.5

if type_logit > type_margin:
    primary = build_jointinit(prismatic, best_pris)
    secondary = build_jointinit(revolute, best_rev)
    type_confidence = clip(type_logit / 2.0, 0, 1)
elif type_logit < -type_margin:
    primary = build_jointinit(revolute, best_rev)
    secondary = build_jointinit(prismatic, best_pris)
    type_confidence = clip(-type_logit / 2.0, 0, 1)
else:
    # uncertain — primary = higher score, low confidence
    type_confidence = 0.3
    primary = whichever has higher score
    secondary = the other
```

`JointInit.secondary` already exists in `io_contract.py` but is never populated by current code.

### Step 6 — phi_0 from envelope (NOT centroid)

PRISMATIC primary (axis = `d_unit`):

```
u_raw_k = quantile(V_k_clean · d_unit, q=0.95)   # leading edge (front of drawer)
# For invalid k: interpolate from neighbors

# Robust normalisation
u_raw -= u_raw.min()
u_norm = u_raw / max(u_raw.max(), eps)
u_norm = PAV(u_norm)                       # monotone smoothing
# Enforce min gap
for i in 1..K-1:
    u_norm[i] = max(u_norm[i], u_norm[i-1] + phi_min_gap=1e-3)
u_norm = (u_norm - u_norm[0]) / (u_norm[-1] - u_norm[0])

c = canonical_state_idx = 2
phi_0 = u_norm - u_norm[c]                  # c-shifted

# Stage D init
diffs = diff(u_norm).clamp_min(phi_min_gap)
delta_u_init = inverse_softplus(diffs)      # (K-1=5,)

# Limits
observed_max_disp = u_raw.max() - u_raw.min()       # world units (d_unit is unit-norm)
disp_limit_softplus = max(observed_max_disp * 1.3, disp_limit_min=0.05)
theta_limit_softplus = theta_limit_min=0.3          # neutral
```

REVOLUTE primary (axis = `a_unit`, origin = step 7):

```
u_raw_k = signed_angle_around_axis(c_2d_k, c_2d_c, axis=a_unit)
# Same normalise / monotonize / c-shift pipeline.

observed_max_angle = u_raw.max() - u_raw.min()       # radians
theta_limit_softplus = max(observed_max_angle * 1.3, theta_limit_min)
disp_limit_softplus = disp_limit_min=0.05            # neutral
```

NO MORE LINSPACE FALLBACK. If a state's V_k is empty, interpolate `u_raw_k` from neighbors. If ALL states are empty, mark confidence ~0 and emit a near-cardinal default (caller's Stage D will catch).

### Step 7 — Origin coupling

REVOLUTE:
```
axis = a_unit (cardinal, signed)
axis_line_point = lift_2d(c2d_center, a_coord=anchor_centroid_3d · a_unit, axis_name=a)
origin = axis_line_point + ((anchor_centroid_3d - axis_line_point) · a_unit) * a_unit
# this puts origin on the axis line at the height of the anchor centroid along a
```

PRISMATIC:
```
# Origin is mathematically arbitrary on a prismatic axis line. Choose origin so
# the axis line passes through the canonical-state centroid (gives Stage D a
# numerically centered start).
c_centroid_3d = mean(voxel_to_world(V_c_clean))
origin = c_centroid_3d - (c_centroid_3d · d_unit) * d_unit
       + ((c_centroid_3d - anchor_centroid_3d) · d_unit) * d_unit
# i.e., origin lies on the line through anchor centroid, projected to the
# canonical state centroid level along d_unit.
```

### Step 8 — Anchors output

```
anchor_band_voxels = (M_motion_corridor_64 > tau_corridor=0.1)
                   & (O_base_canonical > 0)
                   & ~is_carpet_3d

# Dilate by 1 voxel (capture boundary)
anchor_band_dilated = dilate_3d(anchor_band_voxels, radius=1)
anchor_band_dilated &= ~is_carpet_3d            # re-strip carpet post-dilate

# FPS down-sample to anchor_target_count=48
anchors_object = FPS(nonzero(anchor_band_dilated), n=48, seed=0)
```

Identical to current `anchor_extract.py` — no change.

### Step 9 — Confidence aggregation

```
confidence = w_type * type_confidence            # log-ratio margin
           + w_axis * axis_confidence             # iou_perp (pris) OR arc_quality * contact_compat (rev)
           + w_state * (#valid_states / K)
           + w_base * base_count_in_expected_range
where w_type = 0.35, w_axis = 0.25, w_state = 0.20, w_base = 0.20
```

---

## 5. Mapping to failure cases (validation)

### 5.1 30857 (prismatic drawer)

Expected behavior:
- Step 3 IoU_perp along the true drawer-slide cardinal direction ≈ 0.7+ (drawer cross-section consistent across states once we project away the slide direction); along other cardinals it is lower.
- Step 3 front-edge advance is monotone and positive along the slide direction.
- Step 3 newly-appearing voxels' direction agrees with slide direction.
- Step 4 count_stability is LOW (s_0 thin slab vs s_5 thick body) — revolute count_stability drops.
- type_logit > 0.5 → prismatic.

### 5.2 7201 / 7128 (revolute door / lid)

Expected behavior:
- Step 4 count_stability is HIGH (rigid door shape).
- Step 4 arc fit on 2D centroids in perp plane has low residual.
- Step 4 contact_compat = 1 (axis passes through hinge anchor band).
- Step 3 iou_perp is LOW for all cardinals (rotating door projection varies).
- type_logit < -0.5 → revolute.
- Axis is cardinal, not noise-fitted continuous direction.

---

## 6. Code change plan (no implementation yet)

| File | Action |
|---|---|
| `mine/pipelines/stage_c/io_contract.py` | KEEP. Already supports `secondary`. |
| `mine/pipelines/stage_c/config.py` | ADD 8 new hyperparameters (see §7). |
| `mine/pipelines/stage_c/move_geometry.py` | KEEP `compute_per_state_move_geom` as a diagnostic-only helper. NEW function: `clean_per_state_move_sets` (Step 1 above). |
| `mine/pipelines/stage_c/joint_type_detect.py` | REWRITE. Replace `reverse_align_and_score`-based scoring with perpendicular-projection scoring (Step 3 + Step 4). Populate `arc_center / arc_normal / arc_radius` for revolute candidates so axis_fit can reuse them. |
| `mine/pipelines/stage_c/axis_fit.py` | SIMPLIFY. Accept cardinal axis from joint_type_detect directly; compute origin per Step 7. Drop the corridor_PCA fallback (replaced by always-available cardinal candidate). |
| `mine/pipelines/stage_c/phi_fit.py` | REWRITE per Step 6. Replace centroid-projection with front-edge (prismatic) / arc-centroid-angle (revolute). DROP linspace fallback; if degenerate, return a low-confidence valid u_norm with phi_min_gap. |
| `mine/pipelines/stage_c/anchor_extract.py` | KEEP. |
| `mine/pipelines/stage_c/confidence.py` | KEEP STRUCTURE, retune weights to (0.35, 0.25, 0.20, 0.20). |
| `mine/pipelines/stage_c/run_stage_c_init.py` | EDIT. Wire new joint_type_detect signature; populate `JointInit.secondary` with other-type best candidate. |
| `mine/pipelines/stage_c/voxel_scoring.py` | KEEP `reverse_align_and_score` as a VERIFIER (optional confidence boost), not the primary discriminator. |
| `mine/pipelines/stage_c/viz.py` | UPDATE diagnostics to surface (a) iou_perp per axis, (b) front-edge per state, (c) 2D projection HTMLs. |

---

## 7. New hyperparameters

```python
# StageCConfig additions
tau_soft_move: float = 0.15
tau_corridor: float = 0.1
min_observers: int = 2
min_voxels_per_state: int = 30
prismatic_front_quantile: float = 0.95
prismatic_advance_threshold: float = 0.008
prismatic_min_new_voxels: int = 20
extent_scale: float = 0.5
revolute_count_cv_scale: float = 0.3
revolute_extent_cv_scale: float = 0.3
revolute_arc_residual_scale: float = 0.025
contact_radius_world: float = 0.0625          # 4/64
type_margin: float = 0.5                       # was 0.15
# confidence weights retuned
conf_w_type: float = 0.35
conf_w_axis: float = 0.25
conf_w_state: float = 0.20
conf_w_base: float = 0.20
```

---

## 8. Open questions for user

1. **Strict cardinal vs near-cardinal axis output** — strict snaps to {±X, ±Y, ±Z} unconditionally; near-cardinal allows up to ~26° tilt before snapping. Stage D refines axis continuously anyway. I lean strict.
2. **`min_observers=2` vs `min_observers=2 OR in corridor`** — I lean OR-corridor (Step 1 above) to preserve drawer-back voxels that only appear in s_5.
3. **Soft P_base subtraction in Step 1** — current proposal: only hard `O_base_canonical`. Soft has known false-negatives at edges (stageB.md §4.2.3). I lean hard-only.

---

## 9. What I am NOT touching

- `record/method.md` — keep
- `record/pipeline.md` — keep
- `record/stageB.md` — keep (Stage B is locked at v3.3.6 FINAL)
- Stage B code — keep
- Bootstrap caller surface — `pipelines/bootstrap.py` already calls `run_stage_c_joint_init(StageCInputs(...))`; the function signature stays.

After your sign-off on the three open questions, I will implement the changes one file at a time, run the three failure cases (30857, 7201, 7128), and report iou_perp / revolute_score / type_logit diagnostics back.
