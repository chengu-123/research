# Stage B v3 — Symmetric Mix + SDEdit Refinement for State 0

> Date: 2026-04-18
> Status: approved design, under implementation
> Authority: supersedes `design.md` §3 and all push-related mechanisms in `redesign.md`

## 1. Motivation

Prior Stage B iterations (VGCF, BCAC, SCAR with push) all failed to deliver
state 0 alignment. Empirically:

- Pure mix (`mean_of_middles`, 8 steps) produced near-ideal states 1-5
  alignment but state 0 exhibited systematic drift, caused by TRELLIS's
  training-data bias: closed objects are rendered systematically smaller
  than open objects.
- Adding push (any formulation: α/t, α·t, OccDG-style) made results
  strictly worse, because push and mix both pull toward cross-state
  consensus — the compound pull over-homogenized states and amplified
  state 0's drift via iterative forcing.

Two constraints govern the new design:

- **State 0 is authoritative** for the closed-state geometry (drawer
  closed position, fine outline details from the real input image). It
  must not be overwritten by averaging with states 1-5.
- **States 1-5 are video-diffusion-generated** views. They are
  approximately consistent across states (video model enforces temporal
  coherence) but have shape/color drift from rotation/translation
  synthesis. Their geometric consensus (intersection) still encodes the
  correct cabinet body + interior walls.

The goal is to use states 1-5's geometric consensus to **fill** state 0's
TRELLIS-shrunk regions and missing interior, while **preserving** state
0's unique closed-drawer feature.

## 2. Architecture — Two-Pass

### Pass 1 — K-parallel sampling with symmetric mix (no push)

Run the standard `SCARSampler.sample()` with the following configuration:

- `mix_steps = 8`
- `extreme_mix_mode = symmetric`  (per-state self-retention)
- `mix_weights = (0.3, 0.4, 0.3)` — state k keeps 40% of itself, borrows
  30% from state 0 and 30% from state K-1 as anchors
- `alpha_peak = 0.0` — **push fully disabled**. Historic SCAR push is
  kept in code as dead branch (alpha = 0 ⇒ identity).

Output: `{z^(k)_final}_{k=0..K-1}` — K SS latents at `(8, 16, 16, 16)`
each, plus standard decoded `O_stack` at `(K, 64, 64, 64)`.

Pass 1's state 0 is aligned with 1-5 at coarse scale (via mix); it is
not the final output. Its purpose is to enable accurate "state 0-unique"
region identification in Pass 2's guide.

### Pass 2 — Augmented Intersection Guide + SDEdit (endpoint refinement)

Given Pass 1's `{z^(k)_final}` and `{O^(k)}` (decoded probability
fields), refine the ENDPOINT states `{0, K-1}` (per the `refine_states`
config). Each endpoint gets its own guide built from the OTHER K-1
states; middle states are preserved as identity from Pass 1.

**Step 2.1 — Per-target augmented intersection (64³ probability space)**

For each target state `t` in `refine_states`:

```
P^(k) = σ(decoder(z^(k)_final))                         # (K, 64, 64, 64)

# Reference pool = all states except target t.
ref = [k for k in 0..K-1 if k != t]

P_base   = min_{k in ref} P^(k)                         # soft intersection over refs
P_excl_t = ReLU(P^(t) - max_{k in ref} P^(k))           # target's exclusive voxels
P_guide_t = max(P_base, P_excl_t)                       # union
O_guide_t = (P_guide_t > 0.5).float()                   # binarize
```

For state 0: `ref = {1..K-1}` → `P_base` = open-state base consensus;
`P_excl_0` = closed-drawer voxels only state 0 has.

For state K-1: `ref = {0..K-2}` → `P_base` = base consensus that
*includes state 0's shrunk output*; `P_excl_{K-1}` = fully-extended
drawer voxels only state K-1 has.

**Known asymmetry risk for state K-1 (flagged)**: because state 0's
Pass-1 output is systematically shrunk by TRELLIS's closed-state bias,
its inclusion in state K-1's reference pool pulls `P_base` inward at
cabinet-outer voxels. Target state K-1's refined output may therefore
inherit a mild shrink at outer shell. Tracking via
`sdedit_report.voxel_delta_state{K-1}` — if negative (net voxel loss),
this is the symptom. Mitigation (reserved for v3.1 if symptom observed):
exclude state 0 from state K-1's reference pool, or refine state 0
first and use the refined output as reference.

Binarization before encoding matches the encoder's training
distribution (binary occupancy from Objaverse meshes). An OOD-check
fallback path is not added in this first implementation.

**Step 2.2 — Encode guide to SS latent**

```
z_guide = encoder(O_guide)                         # (1, 8, 16, 16, 16)
```

`encoder` is `pipe.models['sparse_structure_encoder']`, frozen. The
encoder's mean branch is used (default `sample_posterior=False`).

**Step 2.3 — SDEdit K=6 batch re-sampling**

Build K=6 "guide" latents per state:

```
guides[k] = z^(k)_final  for all k                # identity guide (default)
for t in refine_states:
    guides[t] = z_guide_t                         # override each target with its own
```

With the default `refine_states = [0]`:
- `guides[0]` = augmented intersection guide for state 0
- `guides[1..K-1]` = identity (their Pass 1 latents)

Earlier this defaulted to `[0, K-1]` (symmetric endpoint refinement); reverted
to `[0]` only — see §4.5 for rationale. The K-1 path is still available as a
knob for ablation.

**Noise term — reuse Pass 1's shared K-parallel noise** (NOT fresh sampling).
Pass 1 initialised all K trajectories from a single noise tensor (`eps`
broadcast to K copies via `noise = eps.repeat(K, 1, 1, 1, 1)`), so state 0's
Pass 2 must stay in the same random-seed family to preserve cross-state
consistency with states 1-5 (which are retained from Pass 1 exactly). Using
fresh noise would put state 0 on a disjoint random trajectory, defeating the
K-parallel invariant.

```
ε = noise_from_pass1                              # (K, 8, 16, 16, 16), all K equal
x_{t*} = (1 - t*) · guides + (σ_min + (1 - σ_min) · t*) · ε
```

`t* = 0.5` (balanced between guide geometry and c_k conditioning).

Denoise from `t*` to 0 using the plain `FlowEulerGuidanceIntervalSampler`
methods (inherited by `SCARSampler`) with `scar_enabled=False`
equivalent — no mix, no push. Schedule length is
`ceil(N_full × t*) = ceil(25 × 0.5) = 13` Euler steps from `t*` to 0,
with the same `rescale_t` as Pass 1.

At each step, per-state conditioning `c_k` drives the DiT forward via
standard CFG + guidance interval.

Output: `{z^(k)_refined}_{k=0..K-1}`.

**Step 2.4 — Selective replacement**

```
z_final_v3[k] = z^(k)_final        # default: identity from Pass 1
for t in refine_states:
    z_final_v3[t] = z^(t)_refined  # override each refined target
```

With the default `[0, K-1]`: middle states 1..K-2 preserve Pass 1
exactly; endpoints 0 and K-1 use Pass 2 refined latents. Middle states
never benefit from Pass 2 (by design — they are the reference pool,
not targets).

### Final Output

```
O_stack = (σ(decoder(z_final_v3)) > 0.5)          # (K, 64, 64, 64)
remove_disk(O_stack)                               # optional carpet removal
save O_stack + diagnostics
```

Downstream Stage C/D/F unchanged.

## 3. Hyperparameters

| Param | Default | Purpose |
|---|---|---|
| `mix_steps` | 8 | Pass 1 mix duration |
| `extreme_mix_mode` | `symmetric` | Pass 1 mix formula |
| `mix_weights` | `(0.3, 0.4, 0.3)` | anchor-self-anchor weights |
| `alpha_peak` | 0.0 | Push disabled |
| `t_star` | 0.5 | SDEdit noise level (balanced) |
| `sdedit_steps` | 13 | Derived from 25 × 0.5 |
| `refine_states` | `[0]` | Target = state 0 only (K-1 endpoint refinement reverted — see §4.5) |

All other params inherit Pass 1's TRELLIS defaults (`rescale_t`,
`cfg_strength`, `cfg_interval`).

## 4. Known Risks

1. **Encoder OOD risk** — `E(augmented_intersection_occupancy)` may fall
   outside the distribution the encoder was trained on (Objaverse real
   objects). User elected to skip pre-implementation encoder OOD check.
   Failure mode: `z_guide` is garbage → SDEdit produces unreasonable
   output. Mitigation on detection: fall back to `z_guide = mean(z^(k))`
   or drop Pass 2 entirely.

2. **SDEdit's "generate from scratch" stress** — Augmented intersection
   guide includes state 0's unique drawer voxels, so the guide already
   has a drawer-at-closed-position hint. SDEdit is in its typical
   "refine existing structure" regime. Risk is lower than with pure
   intersection, but if the drawer voxels in `P_excl` are too few or
   misplaced, DiT may produce an incomplete drawer.

3. **Pass 1 state 0 jitter** — symmetric mix (40% self-retention) mixes
   state 0 along with 1-5. Some jitter in state 0's Pass 1 output is
   acceptable because state 0's Pass 1 output is only used to compute
   `P_excl`, not as a final result.

4. **State 0 / state 1 drawer voxel overlap** — if state 1's drawer is
   barely open, state 0's drawer and state 1's drawer may overlap
   substantially. Then `P_excl` shrinks (state 0's drawer is no longer
   "exclusive" to state 0), and SDEdit's guide loses the drawer hint.
   Risk depends on actual joint range of the input data. Acceptable for
   v3 — to be addressed in future iterations if needed.

## 4.5. State K-1 endpoint refinement: experimented, reverted

An earlier revision of this spec defaulted `refine_states` to `[0, K-1]`:
refine both endpoints symmetrically. Implementation + synthetic test
confirmed the pollution mechanism I had flagged prospectively:

- For state 0, reference pool = states 1..K-1. `min_{k=1..K-1} P^(k)` is
  clean because the TRELLIS-shrunk state 0 is excluded.
- For state K-1, reference pool = states 0..K-2. The `min` includes
  state 0, so wherever state 0 is wrongly empty (outer cabinet shell),
  `P_base` is zero. State K-1's guide therefore LOSES its outer cabinet
  shell at exactly the voxels state 0 is wrong about — the opposite of
  what we want for state K-1.

The downstream effect depends on whether c_{K-1} can restore the lost
shell during SDEdit denoising. Cabinet-outer voxels have weak image
signal (they're at boundary of foreground/background), so c_{K-1} is
unlikely to fully restore them.

Decision 2026-04-18: revert `refine_states` to `[0]` only. State K-1
keeps its clean Pass 1 output. The `refine_states` knob remains wired
up in config + code as an ablation lever (can be re-enabled as `[0,
-1]` for paper comparison).

Two mitigations remain on the shelf if K-1 refinement is revisited:

- **Exclude state 0 from K-1's reference pool** — use `min_{k=1..K-2}`
  instead, at the cost of a smaller (K-3) reference set.
- **Sequential refinement** — refine state 0 first, use the refined
  state 0 as K-1's reference. Ordering dependency but eliminates the
  pollution.

## 5. Implementation Plan

- `configs/v1.yaml` — set `extreme_mix_mode: symmetric`, `alpha_peak:
  0.0`; add `stage_b_sdedit` block with `t_star: 0.5`.
- `pipelines/stage_b_scar.py`:
    - `run_scar` — after Pass 1 decode, invoke Pass 2 builder + sampler.
    - `_build_augmented_intersection_guide(P_stack) -> z_guide` — 64³
      ops + encoder call.
    - `_sdedit_refine_k6(sampler, flow_model, guides, cond, ...) ->
      z_refined` — K=6 SDEdit loop using parent `sample_once`.
- `run_v1.py` — no changes needed (config-driven).

## 6. Ablation (deferred per user)

For the CVPR paper, the planned ablation matrix:

- A0: Pass 1 only, no SDEdit
- A1: Pass 1 + pure intersection guide (no state 0 exclusive)
- A2: Pass 1 + augmented intersection (this design)
- A3: different `t_star` ∈ {0.3, 0.5, 0.7}
- A4: alternative guide = mean of states 1-5

Deferred; to be run after core implementation is verified.
