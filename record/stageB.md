# Stage B — Authoritative Reference (v3.3.6 FINAL)

> **Last updated**: 2026-05-24
> **Status**: production LOCKED
> **Code**: `pipelines/stage_b_scar.py`, `TRELLIS/trellis/pipelines/samplers/scar.py`,
> `TRELLIS/trellis/modules/transformer/modulated.py`
> **CLI**: `scripts/stageb.py`
> **Input mux**: `pipelines/utils/state_input.py`
> **Predecessor docs**: `record/past/stageB/stageB.md` (v4.3 historical authoritative).
>
> This document supersedes the v3.3 / v3.3.1 / v3.3.2 / v3.3.3 / v3.3.4 / v3.3.5
> sections in `record/method.md` §5.2 and `record/pipeline.md` §6.2 with the
> empirically locked v3.3.6 FINAL defaults. method.md and pipeline.md retain the
> v3.3 design narrative; this file is the production source of truth for what
> the code actually does and why.
>
> **v3.3.6 FINAL is the LOCKED Stage B design.** Four iterations
> (v3.3.4 motion-aware safety / v3.3.4 Plan C+ multi-source motion /
> v3.3.5 default-per-state composition / v3.3.6 selector) converged on:
> v3.3.4 motion-aware safety floor as default, with F1 variance dynamic-M,
> kv_floor=0.2, and all other variants retained as ablation switches.

---

## Table of Contents

1. [Role and I/O contract](#1-role-and-io-contract)
2. [Current design (v3.3.6 FINAL)](#2-current-design-v336-final)
3. [Output contract](#3-output-contract)
4. [Evolution timeline v3.3.1 -> v3.3.6](#4-evolution-timeline)
5. [Ablation switches](#5-ablation-switches)
6. [Empirical observations](#6-empirical-observations)
7. [Open issues and future work](#7-open-issues-and-future-work)

---

## 1. Role and I/O Contract

### 1.1 Position in pipeline

```
[Stage A: 21-frame Wan2.2 I2V (or 6 segmented PNG inputs)]
                |
                v
[Stage B: SCAR sampling + saturated BMCSA = K-state base extractor]
                |
                v
[Stage C: joint init / EM / MRF graph-cut on base + move evidence]
```

Stage B's job is K-state base consensus extraction. The K=6 state images encode
the same physical base + different drawer positions. By forcing K-parallel
sampling under cross-state coupling (mix + BMCSA), the model converges to a
base-consistent solution. Two outputs:

- Pass 1: per-state move evidence (K-state Pass-1 occupancy retains state-
  specific drawer positions).
- Pass 2: canonical base (K-state Pass-2 occupancy is highly consistent after
  BMCSA K/V averaging; K-vote distills it).

### 1.2 Input contract

Stage B accepts EITHER of two input sources via `pipelines/utils/state_input.py`:

- **(a) Stage A video tensor**: `wan_video_target_3FHW_uint8.pt`, a torch.Tensor
  uint8 of shape `[3, F, H, W]` (typically F=21, 480x832). We sample K=6 frames
  at indices `[0, 4, 8, 12, 16, 20]`.
- **(b) Segmented image directory**: K=6 PNGs named `{i:02d}_seg.png`
  (FreeArt3D convention).

DINOv2 patch tokens are extracted from each of the 6 images and become the
K-parallel SS-DiT conditioning (`(K, 1374, 1024)`).

### 1.3 Static assumptions

- Single object, single joint, joint type in {revolute, prismatic}.
- Locked-off camera; world frame fixed.
- TRELLIS SS-VAE, SS-DiT, SLAT-DiT, D_GS, DINOv2 all frozen.
- The 6 images encode K=6 articulation states of the same physical instance.

---

## 2. Current Design (v3.3.6 FINAL)

### 2.1 Three-sentence summary

1. **Pass 1**: K-parallel SCAR z_t mix (symmetric, no push) -> K=6 occupancies
   with sharpened middle-state edges and preserved per-state drawer geometry.
2. **Pass 2**: SDEdit from t_star=0.5 + BMCSA on all 24 DiT self-attn blocks.
   Motion-aware safety floor: `M_move = max(M_motion_static, M_motion_dynamic) *
   (1 - M_base_geom)`, clamped at `1 - kv_floor=0.8`. Dynamic-M signal source
   is variance (F1 fix; not cosine).
3. **Output**: K-vote (>= ceil(0.83*K)) on Pass-2 K-state binary gives the
   canonical base; `Pass-1 AND NOT base` gives per-state motion candidate.

### 2.2 Pass 1 — K-parallel SCAR z_t mix (25 Euler steps)

`SCARSampler.sample()` in `TRELLIS/trellis/pipelines/samplers/scar.py`.

```
For step = 0..24:
    1. Optional mix (only when step < mix_steps=8, mix_space='z_t'):
         for each k in 0..K-1:
             z_t^(k) <- w_first * z_t^(0) + w_self * z_t^(k) + w_last * z_t^(K-1)
         Default weights (0.3, 0.4, 0.3) preserve 40% self per state; the
         mixed z_t becomes both the DiT input AND the next step's z_t.

    2. K-parallel DiT forward:
         pred_v = ss_dit(z_t, t, cond_k)

    3. Push (DISABLED by default: alpha_peak=0; the push branch is inert).

    4. Euler step:
         z_{t-dt} = z_t - dt * pred_v
```

**Why z_t mix and not x_0 mix**: in z_t (noisy latent) space, the per-state
conditional denoising under c_k acts as a self-correcting force pulling each
state's mixed-and-noisy latent back to its correct clean trajectory. In x_0
(clean signal) space, mixed content propagates without this self-correction
and pollutes both Pass-1 outputs and downstream BMCSA. Empirical: x_0 mix
produced visible "s0 双桌 (two-drawer ghost)" mode duplication; z_t mix does
not. v4.3's 4/4 sample success was on z_t mix.

The Pass-1 final latent is decoded to soft_p1 (sigmoid) and binary_p1
(threshold 0.5). Optional carpet removal via `remove_disk` (FreeArt3D
convention) follows. soft_p1 / binary_p1 are the per-state move-bearing
occupancy.

### 2.3 Pass 2 — SDEdit + motion-aware BMCSA (12 Euler steps from t_star=0.5)

`_sdedit_refine_k6_bmcsa()` in `pipelines/stage_b_scar.py`.

```
Setup (once):
    P_base_shared_64 = mean_K(soft_p1)                # (64, 64, 64)
    M_base_geom_token = sigmoid((P_base_shared - 0.5) / tau_M=0.05),
                       avg_pool 64->16 = (1, 4096, 1)
    M_attn_16 = sigmoid((cross_state_cosine(z_final, 8-ch) - 0.7) / 0.05)
              = (1, 4096, 1)   # ONLY applied at guide stage (NOT in BMCSA)
    M_motion_corridor_64 = footprint * (1 - shared)   # Pass-1 derived
    Per-state z_guide built from P_base_shared + per-state P_excl,
    encoded by SparseStructureEncoder.

Pass 2 loop (12 fixed Euler steps, t* = 0.5):
    For step = 0..11:
        # ---- Compute dynamic-M (F1: variance-based, not cosine) ----
        prev_x_0_pred (K, 8, 16, 16, 16): previous step's Tweedie estimate
        log_var = log(cross_state_variance(prev_x_0_pred).sum(channel) + 1e-6)
        M_dyn_base_like = sigmoid((quantile(log_var, 0.65) - log_var) / 0.5)
        M_motion_dynamic = 1 - M_dyn_base_like
        M_motion_combined = max(M_motion_static, M_motion_dynamic)

        # ---- v3.3.6 FINAL composition (= v3.3.4 motion-aware safety) ----
        M_move = M_motion_combined * (1 - M_base_geom)
        M_move = clamp(M_move, 0, 1 - kv_floor=0.8)

        # ---- BMCSA blend per DiT self-attn block (all 24 blocks) ----
        y_self   = self_attn(h)                                 # per-state Q/K/V
        y_shared = self_attn(h, share_kv_across_batch=True)     # K-mean K/V
        h        = M_move * y_self + (1 - M_move) * y_shared

        Euler step on K-parallel batch.
        prev_x_0_pred <- this step's pred_x_0  (for next step's M_dyn)
```

**v3.3.6 composition behavior per voxel category** (with default kv_floor=0.2,
so M_move is capped at 0.8):

| Voxel type | M_base | M_motion | M_move | Attention |
|---|---|---|---|---|
| Confident base far from motion | ~1.0 | ~0 | **0** | 100% K-share (base preserved) |
| Base back face (M_base partial) | ~0.7 | ~0 | 0 | ~100% K-share (no M_attn attack) |
| Confident base with motion overlap | ~1.0 | ~0.5 | 0 | 100% K-share (drawer fuses to floor — known limitation) |
| Motion corridor far from base | ~0 | ~0.8 | 0.8 | 80% per-state (drawer geometry preserved) |
| Drawer interior consistent | ~0.2 | ~0.3 | 0.24 | mostly K-share, slight per-state |
| Air | ~0 | ~0 | 0 | K-share (harmless, no object) |

The K-redundancy floor (`kv_floor=0.2`) keeps at least 20% K-averaging at every
voxel, providing safety against single-state TRELLIS instability (the s3-loss
mitigation that prevents v3.3.2-style isolated state corruption).

After Pass 2, K-state occupancies are highly consistent (the "very good base"
the user empirically confirmed). Decoded to soft_p2 / binary_p2.

### 2.4 Output extraction

```
PRIMARY (hard binary):
    O_base_canonical = (votes_p2 >= ceil(0.83 * K)).astype(uint8)
                       # votes_p2 = sum_K(binary_p2)
    O_move_per_state = binary_p1 AND NOT O_base_canonical.astype(uint8)

SECONDARY (soft, for Stage C MRF unary):
    P_base_canonical = mean_K(soft_p2)             # (64, 64, 64) float
                       # NO var_penalty (was multiplicative attenuator, hurt edges)
    P_move_evidence_per_state =
        max(soft_p1 - 0.8 * P_base_canonical, 0)   # (K, 64, 64, 64) float
    base_confidence, move_confidence = aliases of soft fields
```

K-vote threshold (>= 83% of K) is intentional strictness: a hallucinated common
envelope rarely survives 5/6-vote. This strictness is what makes the base
clean.

---

## 3. Output Contract

### 3.1 Directory layout

```
outputs/<name>/
|-- PRIMARY hard binary (what Stage C should consume):
|   |-- O_base_canonical.npy        (64, 64, 64) uint8
|   `-- O_move_per_state.npy        (K, 64, 64, 64) uint8
|-- SECONDARY soft (for Stage C MRF unary):
|   |-- P_base_canonical.npy        (64, 64, 64) float
|   |-- base_confidence.npy         alias of P_base_canonical
|   |-- P_move_evidence_per_state.npy (K, 64, 64, 64) float
|   `-- move_confidence.npy         alias of P_move_evidence_per_state
|-- K-state raw (legacy back-compat):
|   |-- O_stack.npy / O_stack_soft.npy        Pass-2 K-state binary/soft
|   |-- O_stack_pass1.npy / _soft.npy         Pass-1 K-state binary/soft
|   `-- z_final.pt                            (K, 8, 16, 16, 16) Pass-2 latents
|-- diagnostics:
|   |-- dit_hidden.pt                         {block_idx: (K, 4096, 1024)} fp16
|   |-- dynamic_M_log.pt                      (S=12, B=24, L=4096) fp16 M_eff stack
|   |-- scar_diagnostics.json                 per-step mix/push/base_frac
|   |-- sdedit_report.json                    mode/t_star/voxel_deltas/M_compute_mode
|   |-- icp_report.json                       (ICP disabled by default)
|   |-- meta.json                             input source + cfg snapshot
|   `-- config.yaml                           OmegaConf dump of resolved cfg
`-- viz/:
    |-- O_base_canonical.html                 PRIMARY base (the proven good output)
    |-- O_move_per_state.html                 PRIMARY per-state move candidate
    |-- P_base_canonical.html                 SECONDARY soft base
    |-- P_move_evidence_per_state.html        SECONDARY soft move
    |-- P_base_cross_state_std.html           Pass-2 K-state std (diagnostic)
    |-- O_stack.html, O_k_*.html              raw K-state viz
    |-- base_move_preview.html                SAJO-style joint-free split preview
    |-- bmcsa/:
    |   |-- M_base_64.html, P_base_shared_64.html
    |   |-- M_attn_16.html, M_attn_64.html, attn_agreement_16.html
    |   |-- M_motion_corridor_64.html         only when use_motion_corridor=true
    |   |-- M_dynamic_block_{00..23}_mean_64.html
    |   |-- M_dynamic_step_{00..11}_mean_64.html
    |   |-- M_dynamic_scalar_step_block.npy
    |   |-- dynamic_M_diagnostics.html        (S,B) heatmap + B-B cos + evolution
    |   `-- multi_gate_decomposition.html     4 gate value histograms
    |-- guide/                                per-state P_base/P_excl/P_guide HTMLs
    |-- per_step/                             Pass-1 25 step Tweedie occupancy
    |-- pass2_per_step/                       Pass-2 12 step Tweedie occupancy
    `-- scar_base_mask/                       only when mix_space='x_0' (ablation)
```

### 3.2 Stage C downstream consumption

| Stage C input | Source |
|---|---|
| Canonical base anchor | `O_base_canonical.npy` (hard, primary) |
| Per-state motion candidate | `O_move_per_state.npy` (hard); Stage C still applies component filter / swept corridor / temporal counterpart validation |
| MRF unary (base) | `P_base_canonical.npy` (soft) |
| MRF unary (move) | `P_move_evidence_per_state.npy` (soft) |
| Joint init geometric prior | `viz/bmcsa/M_attn_16.npy` (and `M_attn_64.npy` if needed) |
| K-state evidence | `O_stack.npy` (Pass-2 base consensus) and `O_stack_pass1.npy` (Pass-1 move-bearing) |
| DiT semantic features | `dit_hidden.pt` (block 14/16/18, K x 4096 x 1024 fp16) |

Do NOT use mean of `P_move_evidence_per_state` as the move canonical. Per-state
residuals must be aligned via fitted joint motion before any per-state
geometry is averaged.

---

## 4. Evolution Timeline

This section documents the three iterations v3.3.1 -> v3.3.2 -> v3.3.3 of
Stage B in full: design intent, exact code changes, problems found by
empirical run or first-principles critique, status (active / reverted /
kept-as-ablation), and lesson distilled. Each version's lesson informs the
next iteration.

### 4.0 Pre-history: v4.3 (`record/past/stageB/stageB.md`) baseline

The empirically validated 4/4 production version. Key mechanisms used as the
reference point throughout this document:

- Pass 1: K-parallel SCAR sampler, mix on z_t (noisy latent) for steps 0-7
  with symmetric weights (0.3, 0.4, 0.3); push disabled (`alpha_peak=0`).
- Pass 2: SDEdit from t_star=0.5 (12 steps) + BMCSA on all 24 DiT self-attn
  layers. M_base is STATIC, computed once from Pass-1 `P_base_shared =
  mean_K(soft_p1)`, pooled to token space, used unchanged at every step
  and block.
- M_attn (v4.2/v4.3): semantic agreement gate on Pass-1 z_final (8-channel
  cross-state cosine). Applied at the GUIDE stage to filter P_base_shared
  at source (the v4.3 "Plan 3" / Problem 13 fix).
- Output: `O_stack.npy` (K, 64^3) uint8 only; downstream Stage C/D partitions
  base/move from O_stack.

Empirical result: 4/4 success on samples 30857, 7201, 7128, 26525.

### 4.1 v3.3.1 (Stage B perspective: paper-design only, code unchanged)

#### 4.1.1 Design intent (from method.md v3.3 + post-critique fixes)

`record/method.md` and `record/pipeline.md` v3.3 introduced two paper-level
changes to Stage B vs v4.3:

- **(D-v3.9)** SCAR mix: z_t mix -> x_0 mix. Stated motivation: x_0 mix
  preserves per-state noise variance (ODE consistency); z_t mix reduces
  noise variance by `||w||^2 = 0.34` for default weights, so the model sees
  slightly OOD inputs.
- **(D-v3.10)** BMCSA M: static -> dynamic per-block. Stated motivation:
  Pass-2 hidden evolves during SDEdit denoising; static M_base from Pass-1
  end goes "stale" as the latent moves.

v3.3.1 post-critique fixes (C1/S1/S3/S4/M3) addressed bugs and ergonomic
issues mostly in Stage D/F, NOT in Stage B's sampling logic. The Stage B
items in v3.3.1 critique were:

- **(M3) Bootstrap step removal**: the SLAT sampler call on U_seed (method.md
  B6 / pipeline.md B7) was wasted -- its output `z_slat0_seed` was never read
  downstream (joint init uses z_final, the next step always re-samples SLAT
  on U_object). Save ~30s + intermediate latent.
- **(S1.b) U_seed dilate radius 1 -> 2**: more conservative initial coverage
  ahead of the periodic silhouette check.

#### 4.1.2 Actual code changes for Stage B in v3.3.1

In `pipelines/stage_b_scar.py`:
- Output schema enriched with `dit_hidden.pt` capture (block 14/16/18,
  t_star=0.3) for downstream Stage C SegMatch v8 use.
- Multi-gate / multi-output / soft fields all DEFERRED to v3.3.2.

In `pipelines/utils/state_input.py` (NEW file, but functionally identical
to legacy `load_seg_images` until Stage A is wired):
- Mux to accept either 6 PNGs OR a Stage A `.pt` video.

In `scripts/stageb.py` (NEW file):
- Independent CLI entry. Mirrors run_v1.py's Stage B portion plus mux.

In `TRELLIS/trellis/pipelines/samplers/scar.py`:
- No change to the sampler's actual mix or push logic in v3.3.1; remains
  v4.3 behavior (z_t mix, static M_base via existing kwargs from
  stage_b_scar.py).

In `TRELLIS/trellis/modules/transformer/modulated.py`:
- No change; v4.3 BMCSA branch unchanged.

#### 4.1.3 Problems found in v3.3.1

None at empirical level (since Stage B code was effectively unchanged from
v4.3). The paper-level v3.3 design (x_0 mix + dynamic-M) was on paper only,
not yet implemented. Production behavior identical to v4.3.

#### 4.1.4 Lesson

v3.3.1's Stage B critique work was largely deferred; the actual Stage B
behaviour modifications happened in v3.3.2. v3.3.1 is the right *baseline*
to understand because all subsequent changes are tracked relative to it.

#### 4.1.5 Status

Active as code baseline. The output schema added in v3.3.1 (`dit_hidden.pt`,
mux, scripts/stageb.py) is retained in v3.3.3.

---

### 4.2 v3.3.2 (full Stage B redesign: x_0 mix + multi-gate BMCSA + dual-channel soft output)

This was the largest single Stage B revision. Three sub-stages of change,
each addressing a different concern.

#### 4.2.1 Design intent

Three independent goals motivated v3.3.2:

1. **Implement method.md v3.3 designs (D-v3.9 and D-v3.10) in code**: switch
   SCAR mix from z_t to x_0 and BMCSA M from static to dynamic per-block.
2. **Reframe Stage B as base extractor with dual-channel output**: explicitly
   surface base + move as separate fields, so Stage C does not have to
   re-partition from raw O_stack.
3. **Solve the "s0 two-drawer ghost"**: empirical Pass-1 s0 had visible
   mode-duplication artifacts after the v3.3 paper x_0 mix change.

#### 4.2.2 Code changes

**(a) SCAR-x_0 base-masked mix** in `TRELLIS/trellis/pipelines/samplers/scar.py`:

```python
def _compute_base_mask_from_x_0(pred_x_0, base_quantile=0.5):
    # K-state per-voxel divergence (8-ch) -> median threshold -> base mask.
    x_0_mean = pred_x_0.mean(dim=0, keepdim=True)
    div = (pred_x_0 - x_0_mean).pow(2).mean(dim=(0, 1))   # (D, H, W)
    threshold = torch.quantile(div.flatten(), base_quantile)
    return (div <= threshold).float()

def _mix_x_0(pred_x_0, step_idx, base_quantile=0.5):
    # Symmetric mix applied ONLY to base voxels; move voxels keep self.
    full_mix = w_first * pred_x_0[0:1] + w_self * pred_x_0 + w_last * pred_x_0[-1:]
    base_mask = _compute_base_mask_from_x_0(pred_x_0, base_quantile)
    mixed = base_mask * full_mix + (1 - base_mask) * pred_x_0
    return mixed, True, base_frac, base_mask
```

In `sample()`, when `mix_space='x_0'`:
```python
pred_x_0, pred_eps, pred_v = self._get_model_prediction(...)
x_0_mixed, ... = self._mix_x_0(pred_x_0, step_idx)
# Reconstruct z_{t_next} from MIXED x_0 + ORIGINAL eps (ODE consistent):
pred_x_prev = (1 - t_prev) * x_0_mixed + (sigma_min + (1-sigma_min) * t_prev) * pred_eps
```

**(b) Multi-gate BMCSA** in `TRELLIS/trellis/modules/transformer/modulated.py`:

```python
# Replaced v4.3 single static M_base with multi-gate product:
M_eff = M_base_geom * M_attn * (1 - M_motion_corridor) * M_dynamic
# where:
#   M_base_geom    = kwargs["M_base"]              static (Pass-1 derived)
#   M_attn         = kwargs["M_attn"]              static (z_final agreement)
#   M_motion_corridor = kwargs["M_motion_corridor"]   static (NEW gate)
#   M_dynamic       = sigmoid(K-cosine on prev_x_0_pred, 8-ch)  dynamic per step
#                     (signal swapped from 1024-d hidden to 8-ch x_0_pred)
```

Dynamic-M role: redefined from "classifier" to "attenuator" -- can only
reduce sharing, never single-handedly classify a voxel as base.

**(c) Motion corridor mask** in `pipelines/stage_b_scar.py`:

```python
def _compute_motion_corridor_tokenspace(soft_p1, token_resolution=16):
    footprint = soft_p1.max(dim=0).values
    shared    = soft_p1.mean(dim=0)
    P_motion  = footprint * (1 - shared)
    M_tok = avg_pool3d(P_motion, kernel=4)
    return M_tok.view(1, L, 1)
```

`(1 - M_motion_corridor)` factor in BMCSA gates OUT voxels in the swept
volume of the moving part.

**(d) Sampler-DiT interface for prev_x_0_pred** in `_sdedit_refine_k6_bmcsa`:

```python
prev_x_0_pred = guides.detach()
for step_idx, (t, t_prev) in enumerate(t_pairs):
    bmcsa_kwargs["prev_x_0_pred"] = prev_x_0_pred
    bmcsa_kwargs["M_motion_corridor"] = M_motion_corridor
    out = sampler.sample_once(..., **bmcsa_kwargs)
    prev_x_0_pred = out.pred_x_0.detach()        # update for next step
```

**(e) Dual-channel output (initial hard-binary version)**:

```python
# Pass-2 K-state binary -> K-vote primary.
votes_p2 = binary_p2.sum(axis=0)
O_base_canonical = (votes_p2 >= ceil(0.83 * K)).astype(uint8)
O_move_per_state = binary_p1 AND NOT O_base_canonical
```

**(f) yaml defaults flipped**:
```yaml
scar.mix_space:                  z_t -> x_0
stage_b_sdedit.M_compute_mode:   static -> dynamic
stage_b_sdedit.attn_m_apply_at:  guide -> both
stage_b_sdedit.use_motion_corridor: (new) -> true
```

**(g) Post-GPT-review tweak (mid-v3.3.2 sub-iteration)**: dual-channel
output upgraded from hard binary primary to soft + confidence primary. GPT
critique: "O_base = mean > 0.5 is too hard; should use soft + var_penalty +
connected components." Implementation:

```python
sigma_var = 0.15
var_penalty = 1.0 / (1.0 + (std_p2 / sigma_var) ** 2)
P_base_canonical = mean_p2 * var_penalty        # PRIMARY soft
P_move_evidence_per_state = max(soft_p1 - 0.8 * P_base, 0)   # PRIMARY soft
# Hard binary kept as alias at threshold 0.4.
```

Per-step base mask capture for viz validation
(`viz/scar_base_mask/base_mask_step_*.html`).

#### 4.2.3 Problems found in v3.3.2 (empirical run on sample 30857)

User reported three concrete failures:

**Problem 1: Pass-1 s0 "two-drawer overlap"** (mode duplication despite
base-masked mix).
- Symptom: s0 output occupancy contains TWO drawer-like structures
  simultaneously, one closed and one offset.
- Root cause: SCAR-x_0 mix at the endpoints. For k=0, the symmetric mix
  becomes `x_0_mixed[0] = 0.7 * x_0[0] + 0.3 * x_0[K-1]`. In 8-channel SS
  latent space, `0.7 * z_closed_drawer + 0.3 * z_open_drawer` lands at a
  midpoint between two modes; SS-VAE decoder produces the union of both
  modes (mode duplication). Base-masked mix does not save this because the
  drawer area at s0 has K-state divergence > median (move voxel), but the
  mix still gets applied because... actually it does NOT save it because
  the base mask is computed on `pred_x_0` AT THIS STEP, which has its own
  K-state divergence pattern that does not yet reflect the s0 drawer's
  closed-position uniqueness vs s5 fully-open uniqueness. Result: mix
  applied to drawer voxels at endpoints -> mode duplication propagated to
  output.
- Critical insight: z_t mix does NOT have this problem because the model's
  conditional denoising under c_k actively pulls the mixed-and-noisy
  latent back to each state's correct mode. x_0 mix in clean signal space
  has no such self-correction; the mid-point in clean space stays as the
  decoder's two-mode output.

**Problem 2: M_dynamic saturated to ~1 everywhere**.
- Symptom: `viz/bmcsa/M_dynamic_block_23_mean_64.html` and
  `M_dynamic_step_11_mean_64.html` showed all-1 maps.
- Verification: `M_dynamic_scalar_step_block.npy` of shape (S=12, B=24)
  -- EVERY entry was 0.998. Saturation is uniform across all (step, block).
- Root cause: K-state cosine on the same physical object during shared-
  noise sampling is fundamentally high. The original 1024-d hidden cosine
  was hypersphere-saturated (LayerNorm projects every state to a unit
  sphere where the same-object same-token similarity stays >0.95). The
  v3.3.2 fix moved the signal source to 8-channel `prev_x_0_pred`, which
  does NOT have the LayerNorm hypersphere issue, but K=6 states of the same
  physical instance under shared noise still produce cosine > 0.7 on most
  voxels. With sigmoid `(agree - 0.7) / 0.05`, agree=0.98 -> sigmoid(5.6)
  = 0.996. Saturation again.
- Critical insight: cross-state agreement during shared-noise sampling is
  NOT a useful base/move classifier at any latent layer of SS-DiT. The
  discriminative signal lives in decoded occupancy (`P_base_shared` after
  SS-VAE decoder) or in M_attn computed on z_final (the Pass-1 SS latent
  AFTER full sampling). DURING sampling, K-states are necessarily correlated.

**Problem 3: Base lost voxels (worse than v4.3 and worse than the previous
v3.3.2 hard-binary first version)**.
- Symptom: `O_base_canonical` body had visible holes / missing surface
  voxels compared to the "very good base" of the earlier sub-iteration.
- Root cause analysis:
  - Multi-gate product is conjunctive: `M_eff = M_base * M_attn *
    (1 - M_motion) * M_dyn`. For a base voxel at the cabinet edge:
    - K-state std on soft_p2 is slightly > 0, so `M_attn` (sigmoid
      threshold 0.7) drops slightly below 1.
    - `shared` (= mean_K of soft_p1) at the edge is slightly < 1, so
      `M_motion = footprint * (1 - shared) > 0`, so `(1 - M_motion) < 1`.
    - Any imperfect M_base value at the soft edge.
    - All four multiplied: M_eff drops well below 1 at edges -> K/V
      averaging weakened -> base extraction incomplete -> edge voxels
      missing.
  - The user-praised "very good base" had been the saturated-M_dynamic
    version (single M_eff ~ 1 everywhere). The multi-gate disentanglement,
    intended to "protect move while keeping base", actually attenuated base
    edges more aggressively than it protected move (which already had
    Pass-1 evidence).
- Soft `P_base = mean * var_penalty` had the same conjunctive failure: at
  edges, `std > 0` makes `var_penalty < 1`, and multiplied with `mean`
  pulls the output below the threshold that would have classified them
  base in a hard K-vote. GPT's anti-hallucination motivation was real but
  the empirical false-negative on real edges was larger than the
  hypothetical false-positive on hallucinated common envelopes.

#### 4.2.4 Lessons distilled from v3.3.2

1. **Conjunctive (multiplicative) gates catastrophically attenuate edges**.
   Anywhere a sigmoid mask transitions softly through 0.5-1.0, multiplication
   compounds the attenuation. For an "OR-of-evidence" semantic (any
   evidence -> probably base), use additive blending or maximum aggregation;
   for "AND-of-evidence" semantic (all gates agree -> definitely share),
   accept that edges will be aggressively excluded.
2. **K-state cross-attention agreement during shared-noise sampling is not
   a base/move discriminator at any internal latent layer**. Empirically
   verified across both 1024-d DiT hidden and 8-channel x_0_pred. The
   discriminative signal must come from a post-sampling fixed reference
   (Pass-1 P_base_shared in occupancy space) or a layer-time-independent
   computation (M_attn on z_final cross-state cosine).
3. **z_t mix is self-correcting via conditional denoising; x_0 mix is
   self-reinforcing in clean space**. Mathematically the mean is the same
   between z_t mix and x_0 mix; but the model's denoising trajectory under
   c_k naturally pulls mixed-and-noisy latents back to the correct mode in
   z_t space, while it propagates contamination in x_0 space.
4. **Base extractor (Pass 2 K/V averaging) and move preserver (Pass 1
   per-state sampling) should be SEPARATE PASSES, not one mask in one
   pass**. Trying to make BMCSA both protect move AND extract base via a
   single gate forces the unsolvable trade-off where any base-protecting
   relaxation of M_eff at edges leaks into move averaging, and any
   move-protecting reduction of M_eff in candidate-move regions hurts base
   completeness.

#### 4.2.5 Status

v3.3.2 yaml defaults reverted in v3.3.3. All v3.3.2 code is RETAINED in
the tree and accessible via ablation switches; only the production yaml is
changed.

---

### 4.3 v3.3.3 (current production: empirical revert + reframe explicit)

#### 4.3.1 Design intent

Three goals:

1. Restore the empirically validated saturated-K-state-consensus behaviour
   that produced the user-praised "very good base" output.
2. Make explicit that Stage B's job is base extraction in Pass 2 and move
   evidence in Pass 1; do NOT try to do both in one mask.
3. Preserve v3.3.2's code investments as ablation switches so future work
   (and AAAI ablation tables) can reproduce / contrast.

#### 4.3.2 Code changes (relative to v3.3.2)

**(a) yaml defaults flipped back** (the only changes to actual runtime
behaviour):

```yaml
scar.mix_space:                  x_0 -> z_t          # model self-correction
stage_b_sdedit.use_motion_corridor: true -> false   # don't attenuate base
stage_b_sdedit.attn_m_apply_at:  both -> guide       # don't attenuate base
stage_b_sdedit.M_compute_mode:   dynamic              # UNCHANGED; saturates ~1
```

In effect: `M_eff = M_base_geom * M_dyn`, with M_dyn ~ 1 by saturation, so
`M_eff = M_base_geom` -- numerically equivalent to v4.3 static behaviour
(with negligible noise from imperfect saturation).

**(b) Dual-channel output PRIMARY/SECONDARY inverted** in
`pipelines/stage_b_scar.py`:

```python
# v3.3.2: soft was PRIMARY, hard was alias.
# v3.3.3: hard is PRIMARY (K-vote), soft is SECONDARY (no var_penalty).

# PRIMARY (hard binary, restores "very good base"):
min_votes = ceil(0.83 * K_v)
votes_p2 = binary_p2.sum(axis=0)
O_base_canonical = (votes_p2 >= min_votes).astype(uint8)
O_move_per_state = binary_p1 AND NOT O_base_canonical

# SECONDARY (soft, for Stage C MRF unary):
P_base_canonical = mean_K(soft_p2)            # NO var_penalty
P_move_evidence_per_state = max(soft_p1 - 0.8 * P_base, 0)
```

**(c) All v3.3.2 code retained as ablation switches**:
- `mix_space: x_0` still works (base-masked SCAR-x_0 mix path active)
- `use_motion_corridor: true` still works (motion gate enters BMCSA)
- `attn_m_apply_at: both` still works (M_attn at BMCSA blend)
- `M_compute_mode: static` still works (no dynamic computation; pure v4.3)
- All viz / log outputs unchanged

#### 4.3.3 Problems found / verified in v3.3.3

NOT YET RE-RUN ON SAMPLE 30857 AT TIME OF DOCUMENT (pending user
verification). Theoretical expectation: matches the earlier saturated-
dynamic-M output (the "very good base") plus the new dual-channel output
for downstream convenience.

Risk: if dynamic-M does not actually saturate uniformly to ~1 in some
samples (e.g. if K=6 image-conditioning includes a state with corrupted
DINOv2 features), M_eff = M_base_geom * M_dyn may be slightly less than
M_base_geom and produce v4.3-plus-noise behaviour rather than exact v4.3.
Sanity check is the (S, B) heatmap in `viz/bmcsa/dynamic_M_diagnostics.html`
-- expect all entries > 0.95.

If observed degradation persists on production samples, escalate to
`M_compute_mode: static` for exact v4.3 reproduction.

#### 4.3.4 Lesson

1. **Empirical observation > theoretical critique**. GPT's critique of hard
   K-vote ("too hard, hallucinated common envelope passes") was valid in
   principle but the empirical balance favored hard K-vote. The strictness
   of K-vote (>= 5/6 votes) is exactly what filters hallucinations; soft
   alternatives sacrifice strictness for hypothetical robustness.
2. **Code investment is not wasted by reversion**. v3.3.2 code (multi-gate
   BMCSA, motion corridor, soft fields, x_0 mix path, per-step base mask
   viz) is now Ablation Switch Library, valuable for AAAI ablation Table
   ("we tried disentangled gates and they hurt base; here's the proof").
3. **Refactoring a working algorithm has empirical risk even when the
   refactor seems clean on paper**. The v3.3 paper-level design changes
   (x_0 mix, dynamic-M) and v3.3.2 expansions all looked principled but
   produced worse runs than the v4.3 baseline. The minimum-viable
   production reference should always be the most recently empirically
   validated version, not the most recently theoretically appealing version.

#### 4.3.5 Status

SUPERSEDED by v3.3.4 / v3.3.5 / v3.3.6. See sections 4.4 / 4.5 / 4.6.

---

### 4.4 v3.3.4 (motion-aware safety + F1 variance dynamic-M)

#### 4.4.1 Design intent

Two parallel goals:

1. **Fix the dynamic-M cosine saturation root-cause** (F1). v3.3.2 / v3.3.3
   dynamic-M used cosine on L2-normalised 8-channel `prev_x_0_pred`, which
   discards the latent MAGNITUDE that carries the strongest base/move
   occupancy discriminator. Switch to **un-normalised cross-state variance
   + in-batch log-quantile sigmoid** (same math as SCAR Pass-1
   variance mask). Result: dynamic-M no longer saturates to ~1; signal
   becomes per-step adaptive (0.30-0.45 typical range).

2. **Restructure BMCSA composition to motion-aware semantics**. v3.3.2/3
   composition `M_eff = M_base * M_attn * (1 - M_motion) * M_dyn` had
   the multiplicative chain attenuate base BACK FACE (where M_attn was
   moderate due to semantic ambiguity). v3.3.4 changes the gate semantics
   from "K-share strength" to "per-state preservation strength":

   ```
   M_motion_combined = max(M_motion_static, M_motion_dynamic)
   M_move = M_motion_combined * (1 - M_base_geom)
   M_move = clamp(M_move, 0, 1 - kv_floor=0.8)
   h      = M_move * y_self + (1 - M_move) * y_shared
   ```

   M_attn moves OUT of BMCSA composition (kept only at guide stage).
   kv_floor=0.2 K-redundancy floor prevents v3.3.2's s3-loss
   (per-state corruption with no K backup).

#### 4.4.2 Code changes

- `TRELLIS/trellis/modules/transformer/modulated.py`: F1 variance dynamic-M
  + new motion-aware composition + kv_floor floor.
- `pipelines/stage_b_scar.py`: thread `M_dynamic_signal`, `var_percentile`,
  `var_eta`, `kv_floor` kwargs through `_sdedit_refine_k6_bmcsa`.
- `configs/v1.yaml`: defaults `M_dynamic_signal: variance`,
  `var_percentile: 0.65`, `var_eta: 0.5`, `kv_floor: 0.2`,
  `use_motion_corridor: true`.

#### 4.4.3 Empirical (30857)

- F1 dynamic-M: confirmed unsaturated (per-step 0.30-0.45, std=0 within
  step because prev_x_0_pred is shared across all 24 blocks per step).
- Base completeness: improved over v3.3.3 (more voxels survive at base
  back face because M_attn no longer attacks).
- Base/move separation: clearer; user reported "更进一步的保持了base，并且明显的区分出了base和move".
- Drawer-above-cabinet physical effect: NOT visible (cabinet floor M_base
  high + motion high -> M_move=0 -> full K-share -> drawer fuses to floor).

#### 4.4.4 Status

CORE OF v3.3.6 FINAL. v3.3.6 keeps v3.3.4's composition as default.

---

### 4.5 v3.3.4 Plan C+ multi-source motion corridor (rejected ablation)

#### 4.5.1 Design intent

User insight: "motion 信号应该来自跨状态 latent 关系，不光在 dinov2 patch 中".
v3.3.4's single-source motion corridor `M_motion = footprint * (1 - shared)`
was empirically too sparse on 30857 (1349 voxels >0.1, max=0.61) because
SCAR Pass-1 z_t mix had heavily aligned K-state occupancies, collapsing
(1-shared) to ~0 at most trajectory voxels.

Plan C+: OR-compose 3 cross-state motion signals to overcome SCAR squashing:

- **Source A**: occupancy `footprint * (1 - shared)` (existing, weak)
- **Source B**: `z_final` 8-d cross-state variance (16^3 -> 64^3 trilinear)
- **Source C**: `dit_hidden` 1024-d cross-state variance at block 14/16/18
  (16^3 -> 64^3); requires `scar.capture_dit_hidden=true`

Each source normalised via log + in-batch quantile sigmoid, OR-composed
via voxel-wise max. dit_hidden capture moved from end-of-Pass-2 to
end-of-Pass-1 so it's available for motion source computation.

#### 4.5.2 Empirical (30857)

Plan C+ raised motion signal max=1.00, mean=0.645, >0.3 voxel count
260,176 (vs v1 single-source 276 voxels >0.3). 943x signal strength gain.

**BUT no visible improvement** in output: at OBJECT voxels, M_motion_enhanced
mean = 0.9998 (essentially 1 everywhere on object). Combined with v3.3.4's
`M_move = motion * (1 - M_base)` formula, M_move is still ~0 at cabinet
floor (because M_base ~ 1 there) -> drawer still fuses to floor.

**Diagnosis**: Plan C+ identifies more motion correctly, but at the
base-motion BOUNDARY (where physical detail emerges), v3.3.4's
multiplicative safety floor `(1 - M_base)` kills the per-state signal
regardless of motion strength. Plan C+ direction was wrong: stronger
motion alone cannot override v3.3.4 base safety floor at the boundary.

#### 4.5.3 Status

REJECTED as default. Retained as ablation switch
`motion_corridor_source: enhanced` (default `occupancy_only`).

---

### 4.6 v3.3.5 default-per-state composition (rejected ablation)

#### 4.6.1 Design intent

Reverse-engineered from v3.3.2 BMCSA gate analysis on
`outputs/30857/stageb_3.3.2`. Key finding:

> v3.3.2 dynamic-M cosine was SATURATED to 0.9976 (a bug, not a feature).
> This made `M_eff = M_base * M_attn * (1 - M_motion) * 0.9976 ~= 0` at
> 99% of voxels (because M_base mean was 0.012 sparse). 99% of tokens
> got per-state attention by accident — and that's what produced the
> user-observed physical effect (drawer z=37 vs cabinet z=36).

v3.3.5 attempts to reproduce v3.3.2's per-state dominance WITHOUT the
cosine saturation bug, via inverted composition:

```
P_definite_base = M_base * (1 - M_motion_combined)
M_move = 1 - alpha * P_definite_base    # default alpha = 0.9
```

Now M_move defaults to 1 (per-state) everywhere, reduced only at
"confident base AND no motion" voxels. M_attn deliberately NOT included
(avoid v3.3.2 back face attenuation).

#### 4.6.2 Empirical (30857)

Synthetic composition test on 7 voxel categories: PASS — v3.3.5 matches
v3.3.2's per-state strength at cabinet floor + improves on back face
(no M_attn attack). Reproduces v3.3.2's M_move=0.8 at the critical
cabinet-floor-with-motion voxels.

**BUT actual production run on 30857**: user reported "看起来和3.3.4没什么
区别". Visual base/move separation similar to v1 baseline; drawer-above-
cabinet physical effect did NOT emerge.

**Diagnosis**: Even with 80% per-state attention at cabinet floor, the
per-state c_k (DINOv2 of each state's image) does NOT differentiate
drawer Y position enough to push DiT outside its trained prior of
"drawer on floor". The mechanism that produced v3.3.2's physical effect
was specific to that run's cosine saturation interacting with a particular
random seed; not reproducible architecturally.

#### 4.6.3 Status

REJECTED as default. Retained as ablation switch
`bmcsa_composition: v335` (default `v334`).

---

### 4.7 v3.3.6 FINAL (selector + lockdown)

#### 4.7.1 Design intent

After 4 iterations exploring physical-effect reproduction (v3.3.4 +
Plan C+ + v3.3.5), the empirical conclusion is:

> The drawer-above-cabinet physical effect cannot be reproduced
> deterministically at the BMCSA composition level in this framework.
> v3.3.2's appearance was a side effect of dynamic-M cosine saturation,
> not a designable mechanism.

v3.3.6 locks in the empirically validated default (v3.3.4 motion-aware
safety floor) and exposes all explored variants as ablation switches
for the AAAI paper ablation table.

#### 4.7.2 Code changes

- `TRELLIS/trellis/modules/transformer/modulated.py`: add
  `bmcsa_composition: 'v334' | 'v335'` selector inside BMCSA block,
  routing to the two composition paths. Default `v334`.
- `pipelines/stage_b_scar.py`: thread `bmcsa_composition` from cfg.
- `configs/v1.yaml`: lock `bmcsa_composition: v334` as the FINAL
  Stage B production default; document all 11 yaml knobs.

#### 4.7.3 Production behavior summary

```
Pass-1 SCAR z_t mix (25 steps, weights 0.3/0.4/0.3, push disabled)
   -> per-state move evidence preserved

Pass-2 SDEdit + motion-aware BMCSA (12 steps from t*=0.5)
   M_dyn = F1 variance signal (no saturation)
   M_motion_combined = max(M_motion_static, 1 - M_dyn)
   M_move = M_motion_combined * (1 - M_base_geom)
   M_move = clamp(M_move, 0, 0.8)        # kv_floor=0.2 s3-loss safety
   h = M_move * y_self + (1 - M_move) * y_shared
   -> base voxels K-averaged; motion-corridor-non-base voxels per-state

Output:
   O_base_canonical = K-vote(Pass-2 binary, >= 5/6) hard primary
   O_move_per_state = Pass-1 binary AND NOT O_base hard primary
   P_base_canonical = mean_K(Pass-2 soft) soft secondary for Stage C MRF
   P_move_evidence_per_state = max(Pass-1 soft - 0.8*P_base, 0) soft
```

#### 4.7.4 Status

**PRODUCTION LOCKED**. Future Stage B changes should be ablation
switches, not new defaults. Drawer-Y physical correctness is moved
to Stage C/D post-process scope or paper limitations.

---

## 5. Ablation Switches

All v3.3.2 / 3 / 4 / 5 code retained; switch via yaml flag to study.

| Flag | v3.3.6 Default | Effect when changed |
|---|---|---|
| `scar.mix_space` | `z_t` | `x_0`: SCAR mixes in clean signal space; expected s0 ghost duplication (v3.3.2 fail) |
| `scar.alpha_peak` | `0` | `>0`: re-enables Tweedie variance push (historic; not recommended) |
| `stage_b_sdedit.use_motion_corridor` | `true` | `false`: motion corridor gate disabled (v3.3.3 behavior; expected to keep base but lose any motion-aware preservation) |
| `stage_b_sdedit.motion_corridor_source` | `occupancy_only` | `enhanced`: Plan C+ multi-source motion (OR of occupancy + z_final variance + dit_hidden variance). Over-saturates on object voxels; not recommended |
| `stage_b_sdedit.attn_m_apply_at` | `guide` | `bmcsa`/`both`: M_attn enters BMCSA blend as a gate; v3.3.2 had this and base back face suffered |
| `stage_b_sdedit.M_compute_mode` | `dynamic` | `static`: no per-step dynamic-M; M_motion_dynamic source removed; equivalent to v4.3 exact |
| `stage_b_sdedit.M_dynamic_signal` | `variance` | `cosine`: legacy v3.3.x path; saturates to ~1 (the bug F1 fixed) |
| `stage_b_sdedit.var_percentile` | `0.65` | Tighter (e.g. 0.8) shrinks motion-dynamic firing region |
| `stage_b_sdedit.var_eta` | `0.5` | Smaller (e.g. 0.25) sharpens motion-dynamic sigmoid |
| `stage_b_sdedit.kv_floor` | `0.2` | `0.0`: removes K-redundancy safety; expected s3-loss risk on edge samples |
| `stage_b_sdedit.bmcsa_composition` | **`v334`** | **`v335`**: default-per-state composition (v3.3.5). Empirically equivalent on 30857; does not produce physical drawer-Y effect |
| `stage_b_sdedit.base_kshare_alpha` | `0.9` | Only effective when `bmcsa_composition=v335`; controls K-share strength at confident base |

CLI shortcut overrides (in `scripts/stageb.py`):

```bash
# v3.3.6 FINAL production (default):
python scripts/stageb.py --input ... --output_dir ...

# v3.3.5 default-per-state ablation:
python scripts/stageb.py --input ... --output_dir ... --bmcsa_composition v335

# Plan C+ multi-source motion ablation (over-saturates):
# edit yaml: motion_corridor_source: enhanced

# Pure v4.3 reproduction:
python scripts/stageb.py --input ... --output_dir ... \
    --mix_space z_t --M_compute_mode static

# v3.3.2 x_0 mix legacy (expected: s0 two-drawer ghost):
python scripts/stageb.py --input ... --output_dir ... --mix_space x_0
```

---

## 6. Empirical Observations

### 6.1 Sample 30857 (prismatic drawer)

- **v3.3.2** (multi-gate + x_0 mix + soft primary): base lost voxels at
  cabinet edges; s0 had "two-drawer overlap" mode duplication;
  `M_dynamic_scalar_step_block` was all-0.9976 (cosine saturation bug).
  Side effect of saturation: M_eff = M_base * M_attn * (1-M_motion) * ~1
  ~= 0 at 99% of voxels (since M_base mean=0.012) -> per-state attention
  dominant -> drawer-Y position preserved per state -> user-observed
  "drawer z=37 vs cabinet z=36" physical effect.
- **v3.3.3** (saturated K-state consensus + z_t mix + hard K-vote):
  matched the user-reported "very good base" run; physical effect lost
  (because v3.3.3 default removed motion corridor; M_eff stayed at
  v4.3-style static behavior).
- **v3.3.4 v1** (motion-aware safety + F1 variance dynamic-M, default
  occupancy_only source): clearest base + base/move separation observed
  to date. User confirmed "更进一步保持了 base 并且明显区分出 base 和 move".
  Physical drawer-Y effect NOT visible.
- **v3.3.4 v2 (Plan C+ enhanced motion)**: motion signal strength
  rose 943x (max=1, mean=0.645, >0.3 count 260,176 vs v1 276); however
  user reported visual output indistinguishable from v1. Diagnosis:
  M_motion_enhanced saturates to ~1 at object voxels; v3.3.4's
  multiplicative safety floor (1 - M_base) still kills M_move at
  cabinet-floor boundary.
- **v3.3.5 default-per-state**: BMCSA composition inverted to default
  per-state. Reproduces v3.3.2's M_move ≈ 0.8 at cabinet-floor-with-
  motion voxels (per synthetic 7-voxel test). Production run on 30857
  reported "看起来和 3.3.4 没什么区别"; physical effect not reproduced.
  Diagnosis: per-state attention requires per-state c_k DiT outputs to
  geometrically differ at the drawer Y position; in this framework
  DINOv2 cond does not push DiT outside its trained prior of
  "drawer-on-floor" reliably.
- **v3.3.6 FINAL**: locks v3.3.4 motion-aware composition as default;
  Plan C+ and v3.3.5 retained as ablation switches. Drawer-Y physical
  correctness moved out of Stage B scope.

### 6.2 Why M_dynamic saturates regardless of representation dim

- 1024-d hidden cosine: saturation explained by hypersphere normalization
  under LayerNorm + same-object-different-state semantic similarity.
- 8-channel `prev_x_0_pred` cosine: lower-dim, but K=6 states of the same
  physical object on the same denoising trajectory under shared noise are
  highly correlated; cosine still > 0.7 for most voxels (sigmoid sigma=0.05
  amplifies anything > 0.7 to near 1).

The lesson: cross-state agreement on a single object during shared-noise
sampling is *not* a useful base/move classifier signal at any layer of the
SS-DiT internal representations. The discriminative signal is in the
*decoded* occupancy (v4.3 P_base_shared mean) or the SS latent post-Pass-1
(v4.3 M_attn z_final cosine), which are sparse / lower-dim / not on a unit
sphere. dynamic-M as a classifier was a wrong design.

dynamic-M as an *attenuator* (v3.3.2 GPT review) was correct in principle
but unnecessary in practice -- the saturated M ~ 1 makes it act like
identity multiplication, and the static M_base + M_attn (when at guide
stage) already provide all the geometric and semantic gating Pass 2 needs.

### 6.3 Why the "saturated" version is actually correct

The user's empirical observation -- "saturated dynamic-M extracts a very
good base" -- is the right way to look at it. The mechanism:

- BMCSA's `(1 - M_eff) * y_self + M_eff * y_shared` with M_eff = 1
  everywhere becomes pure cross-batch K/V averaging at every self-attn
  layer.
- K/V averaging across K=6 states forces the K parallel trajectories
  toward the cross-state agreement direction.
- For base voxels (K agree in content): averaging reinforces the consensus
  -> clean base extraction.
- For move voxels (K disagree): averaging smooths them out -> they are
  absorbed into the "common" representation -> Pass 2 cannot represent them.

This is acceptable because Pass 1 *already* preserved the per-state move
geometry (no BMCSA in Pass 1). Stage B's job is then: Pass 2 -> base
extractor; Pass 1 -> move evidence source. Do not try to make a single
pass produce both.

---

## 7. Open Issues and Future Work

### 7.1 M_attn semantic gate's hinge-axis blind spot

Inherited from v4.3 / stageB.md section 9.1. M_attn cannot distinguish
"consistently occupied by drawer" from "consistently occupied by cabinet".
Voxels on a revolute hinge axis (drawer material, same world position in
all K states) get high M_attn and may be classified as base. Affects
revolute objects; not present on prismatic. Mitigation deferred to Stage C
component-level processing.

### 7.2 P_excl ghost components

Inherited Pass-2 SDEdit guide construction: `P_excl_k = P_target -
max_{j!=target} P_other`. For TRELLIS hallucinated single-state voxels, this
keeps them as part of the guide. Per Stage C v8.1 footprint formulation we
already do `min_observers >= 2` voting downstream; Stage B itself does not
filter ghosts. GPT review proposed component-level temporal counterpart /
swept corridor / image-cond validation; not implemented (deferred to
Stage C).

### 7.3 K = 6 hard-coded

Stage A produces 21 frames; Stage B samples 6 (indices 0, 4, 8, 12, 16, 20).
Different K is not currently a switch; downstream Stage C K-vote threshold
(>= ceil(0.83 * K)) does scale, but other defaults (mix_weights, etc.)
assume K = 6.

### 7.4 Soft secondary fields' role in Stage C is not yet wired

`P_base_canonical` and `P_move_evidence_per_state` are saved for Stage C MRF
unary use, but the current `stage_c_sajo` driver reads only `O_stack`.
Stage C v9 (planned) should switch to consuming soft fields directly for
better calibration in graph-cut.

### 7.5 dynamic-M log

`dynamic_M_log.pt` (S=12, B=24, L=4096 fp16 stack) is captured each run.
With v3.3.6 F1 variance dynamic-M, the log shows per-step variation
(0.30-0.45 typical range) instead of the v3.3.3 cosine-saturated 0.998.
All 24 blocks within a step give identical M values because
`prev_x_0_pred` is shared across blocks per step (variance is computed
from a per-step input). Per-block variation would require computing
variance from current block hidden state (tested in v3.3.2; saturated).
Acceptable as is.

### 7.6 Resolution dependency

Stage A produces 480x832 video (v3.3.1 NEW.2 fix to align with Wan2.2
official SUPPORTED_SIZES). Stage B downscales to TRELLIS DINOv2 input
(518x518). This works but the cropping is hard-coded in `pipe.preprocess_image`.
If Stage A switches to a different official size (832x480 / 720x1280),
Stage B's image preprocessing remains correct but the K=6 frame sampling
indices are unchanged.

### 7.7 Drawer-Y physical correctness (BMCSA-level UNSOLVABLE)

**Problem**: TRELLIS training distribution puts drawer voxel = cabinet
floor voxel + 1 layer (physically correct: drawer sits ON cabinet floor,
not merged WITH it). But TRELLIS inference under K-state K/V averaging
collapses drawer to merged-with-floor (z_drawer = z_cabinet). User
observed v3.3.2 accidentally produced the correct z_drawer = z_cabinet + 1
on sample 30857 due to the cosine saturation bug.

**Attempted fixes (all rejected on 30857)**:

1. v3.3.4 motion-aware safety: M_move=0 at cabinet floor (M_base*motion
   both high) -> K-share dominant -> drawer fuses to floor.
2. Plan C+ enhanced motion sources: signal strength up 943x but
   v3.3.4's `(1 - M_base)` floor still kills M_move at boundary.
3. v3.3.5 default-per-state: M_move=0.8 at cabinet floor (per-state
   dominant); but per-state c_k DINOv2 condition does not differentiate
   drawer Y enough to push DiT outside trained prior.

**Conclusion**: At BMCSA composition level, drawer-Y physical correctness
is NOT reliably reproducible in this framework. v3.3.2's appearance
was a non-deterministic side effect, not a designable mechanism.
Drawer-Y physical correctness moved out of Stage B scope:

- **Option 1 (recommended)**: Stage C mesh post-process — after
  base/move segmentation, detect drawer-cabinet vertical adjacency
  and shift drawer mesh up by 1 voxel.
- **Option 2**: Stage D non-collision loss — per-instance optimization
  loss that penalizes drawer-cabinet voxel overlap in canonical pose.
- **Option 3**: paper limitation — explicit acknowledgment that BMCSA
  cannot enforce TRELLIS-distribution-external physical constraints.

---

_End of stageB.md (v3.3.6 FINAL authoritative)_
