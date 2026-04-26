# Stage B v4 — SDEdit + Base-Masked Cross-State Attention (BMCSA)

> Date: 2026-04-18
> Status: approved design, pending implementation plan
> Supersedes: `2026-04-18-stageb-v3-sdedit-design.md` (v3 remains as legacy / ablation baseline)
> Authority: overrides `design.md §3` and all prior push-related mechanisms

## 1. Executive Summary

Stage B v3 (SDEdit + augmented-intersection guide) fixed state 0's static geometry
at the guide construction moment but had no mechanism to maintain base consistency
across K=6 states during the 12-step Pass-2 denoising — each state's DiT forward
remained independent, so small per-state drift during denoising could still break
the consistency the guide established at t\*.

Stage B v4 keeps v3's SDEdit starting point (which is good) and adds
**Base-Masked Cross-State Attention (BMCSA)** during Pass-2 denoising: at every
DiT self-attention layer, each state's query attends BOTH to its own per-state
K/V AND to cross-state mean K/V, blending the two attention outputs by a spatial
mask `M_base` derived from Pass 1's `mean_{k=1..K-1} P^(k)`. At base voxels,
attention output comes from the cross-state consensus (feature-level base
sharing); at move voxels, from per-state self-attention (per-state drawer
preservation).

The mechanism is training-free, follows MorphAny3D's source-edit + kwargs pattern
(arXiv:2601.00204), and is a symmetric generalisation: all K states get refined
simultaneously in the same Pass-2 batch, each with its own c_k image
conditioning, starting from a per-state augmented-intersection guide, evolving
under cross-state attention sharing gated by M_base.

## 2. Problem Diagnosis Recap

1. **TRELLIS closed-state shrink bias** — state 0's TRELLIS output is systematically
   smaller than states 1-5 for the same object because training data biases
   closed-state renderings to more compact occupancy.
2. **Per-state drift during K-parallel sampling** — even with Pass-1 symmetric
   mix (0.3, 0.4, 0.3), accumulated per-step DiT differences make states 1-5
   slightly disagree at base voxels in final output.
3. **Push is unstable** — the Tweedie-gradient force field sometimes helps,
   sometimes hurts, depending on data; its direction (toward cross-state mean)
   is not reliably correct.
4. **SDEdit alone is static** — v3's guide only biases the Pass-2 STARTING point;
   during denoising, K states evolve independently and drift apart again.
5. **Encoder OOD risk** — v3 builds guides in 64^3 occupancy then encodes via
   SparseStructureEncoder; the encoder was trained on real Objaverse occupancy,
   not synthetic `max(intersection, exclusive)` unions. Clear risk of unstable
   guide latents.

**BMCSA addresses (2), (3), (4), and partially (5)**:
- (2): per-step cross-state attention sharing prevents drift.
- (3): replaces push with a principled attention-level mechanism.
- (4): makes the cross-state coupling ACTIVE throughout denoising, not just at
  t\*.
- (5): BMCSA operates in DiT's native feature space; even if the SDEdit guide
  latent is slightly OOD, BMCSA pulls features toward DiT's natural
  cross-state consensus at base voxels.

**(1)** is addressed by the SDEdit starting point (inherited from v3): each
state's guide at `t*=0.5` already has the correct scale from M_base.

## 3. Architecture Overview

```
                  Pass 1                              Pass 2
                                          ┌──────────────────────────────┐
  (K images c_0..c_5)                     │                              │
         │                                │    K=6 parallel sampling     │
         ▼                                │    from per-state x_{t*}     │
  SCAR sampler                            │    down to t=0 in 12 steps   │
  (symmetric mix,                         │                              │
   push OFF)                              │    Every self-attn layer:    │
         │                                │    BMCSA blend               │
         ▼                                │    y = (1-M)·y_self          │
  {z^(k)_final_p1}                        │      + M·y_shared            │
  {P^(k) = σ(D(z^(k)))}                   │                              │
         │                                │    cross-attn: per-state c_k │
         │                                │                              │
         ▼                                └──────────────────────────────┘
  Build M_base (s1-5 mean)                              ▲
  Build per-state z_guide_k                             │
         │                                              │
         └──► z_guide_k, M_base_tokenspace ────────────┘
         │                                              │
  Noise K=6 batch: x_{t*}^(k) ─────────────────────────┘
  = (1-t*)·z_guide_k + σ_{t*}·ε_shared

  After Pass 2:
  O_stack = σ(D(z_pass2_final)) > 0.5 → remove_disk → save
```

## 4. Components (each specified as an isolated unit)

### 4.1 Pass 1 — unchanged (SCAR symmetric mix, push off)

- Input: K image conditionings, shared noise `ε ~ N(0, I)` broadcast K times
- Sampler: `SCARSampler` with
  - `extreme_mix_mode = symmetric`
  - `mix_weights = (0.3, 0.4, 0.3)`
  - `mix_steps = 8`
  - `alpha_peak = 0.0` (push off)
- Output: `z^(k)_final ∈ R^{K×8×16×16×16}`, `P^(k) = σ(decoder(z^(k))) ∈ R^{K×64×64×64}`

No code change from current v3.

### 4.2 M_base construction

**Input**: `P^(k)` from Pass 1 decoded occupancy (K, 64, 64, 64).

**Computation**:

```
# v4.1 (2026-04-18 second revision): P_base_shared is the SAME tensor for all
# K targets; computed once from mean over ALL K states (including state 0).
# Including state 0: even though TRELLIS gives it a closed-state shrink bias,
# at K=6 the bias is diluted 1/K ≈ 17 % of the mean -- at cabinet outer-shell
# voxels (state 0 ≈ 0.2, s1-5 ≈ 0.9) the mean is 0.78 > 0.5, so the shell
# remains inside P_base_shared. Including state 0 also retains its real-image
# contribution at voxels where it is correct.
P_base_shared = mean_{k=0..K-1} P^(k)                  # (64, 64, 64), ALL K states
M_base_64 = sigmoid((P_base_shared - 0.5) / τ_M)       # soft mask ∈ [0, 1]
# token_resolution = flow_model.resolution / flow_model.patch_size.
# TRELLIS-image-large: resolution=16, patch_size=1 → token_resolution=16, L=4096.
# pool_kernel = 64 / token_resolution = 64 / 16 = 4.
M_base_tok = avg_pool3d(M_base_64[None, None], kernel=4, stride=4)   # (1, 1, 16, 16, 16)
M_base_flat = M_base_tok.view(1, 4096, 1)              # (1, L=4096, 1) broadcasts with (K, L, D)
```

**Ablation knob**: `exclude_state_0=True` (config) reverts to `mean over s1-5`
and drops state 0's contribution. Kept as a sweep point for the paper, NOT
the default.

**Shape invariant**: token-space M_base is `(1, L, 1)` where L = 4096 = 16^3 (TRELLIS-image-large: resolution=16, patch_size=1).
Broadcasts against attention outputs of shape `(K, L, D)` via the L axis, leaving
batch and channel dims free.

**Hyperparameters**:
- `τ_M = 0.05` (soft mask sharpness; smaller = sharper)
- Mean (not min) over states 1..K-1: TRELLIS bias on state 0 would drag min down;
  mean uses all 5 open-state observations uniformly.

**Rationale for excluding state 0**: state 0 is the known outlier (closed-state
shrink bias). Including it in the min/mean would contaminate the base mask with
its shrunk outer shell. Using only s1-5 gives a TRELLIS-unbiased base estimate,
precisely what we want state 0 to be pulled toward.

**Token space alignment**: TRELLIS SS DiT uses `resolution=16, patch_size=2`,
yielding `(16/2)^3 = 512` spatial tokens. M_base at 64^3 is spatial-aligned with
the occupancy output; downsampling by factor 8 reaches the token resolution 8^3,
with token flatten order matching patchify's. This is a 1-to-1 spatial
correspondence.

### 4.3 Per-state guide construction

**Input**: `P^(k)` (K, 64, 64, 64), `P_base_shared` (64, 64, 64) from §4.2,
SparseStructureEncoder `E` (from `pipe.models['sparse_structure_encoder']`).

**Computation** (v4.1: P_base_shared is THE SAME tensor for every target k):

```
# P_base_shared: computed ONCE above in §4.2 (mean over all K states).
# Reused here uniformly across all targets — every state's guide has the
# same base component. This is what makes all K guides start Pass 2 with
# an aligned base.
for k in 0..K-1:
    non_target = [j for j in 0..K-1 if j != k]
    P_max_non_target = max_{j in non_target} P^(j)           # (64, 64, 64)
    P_excl^(k) = ReLU(P^(k) - P_max_non_target)              # state k's unique voxels

    # CANONICAL v4.1: base is the shared mean, not per-target min.
    # (v4-draft used min_{j != k} P^(j) per target, which gave each state a
    # DIFFERENT P_base; that was a bug and has been fixed.)
    P_guide^(k) = max(P_base_shared, P_excl^(k))             # (64, 64, 64)
    O_guide^(k) = (P_guide^(k) > 0.5).float()                # binarise for encoder
    O_guide^(k) = O_guide^(k).unsqueeze(0).unsqueeze(0)       # (1, 1, 64, 64, 64)
    z_guide^(k) = E(O_guide^(k))                             # (1, 8, 16, 16, 16)

z_guide = concat(z_guide^(k), dim=0)                         # (K, 8, 16, 16, 16)
```

Because P_base_shared is identical across k, every `P_guide^(k)` has the
same "base" subset; the difference across k lives entirely in `P_excl^(k)`.
This is the mathematical statement of "shared base, per-state exclusive".

**Properties per voxel type**:

| Voxel type | P^(0) | P^(1..5) | P_base | In P_guide^(0)? | In P_guide^(K-1)? |
|---|---|---|---|---|---|
| Cabinet outer (s0 shrunk, 1-5 occupied) | 0.2 | 0.9 | 0.9 | ✓ (via P_base) | ✓ (via P_base) |
| Cabinet interior wall (s0 occluded, 1-5 see) | 0.1 | 0.9 | 0.9 | ✓ | ✓ |
| s0 closed drawer | 0.9 | 0.0 | 0.0 | ✓ (via P_excl_0) | ✗ |
| sK-1 extended drawer | 0.0 | 0.9 for K-1 only | 0.18 (mean 0.9/5) | ✗ | ✓ (via P_excl_{K-1}) |
| Middle s_k drawer position | 0.0 | 0.9 for k only | 0.18 | ✗ | via P_excl_k if k unique |
| Empty air | 0.0 | 0.0 | 0.0 | ✗ | ✗ |

### 4.4 SDEdit initialisation

**Input**: `z_guide` (K, 8, 16, 16, 16), `ε_shared` = Pass 1's init noise
(expanded to K identical copies via `.repeat(K, 1, 1, 1, 1)`).

**Computation**:

```
t_star = 0.5
sigma_min = sampler.sigma_min (from SS flow model config)

x_{t*}^(k) = (1 - t*) · z_guide^(k) + (sigma_min + (1-sigma_min) · t*) · ε_shared
```

With `t*=0.5` and typical `sigma_min ≈ 1e-5`:
- Signal weight: 0.5
- Noise weight: ≈ 0.5

**Shared noise**: `ε_shared` is THE noise tensor from Pass 1's initialization. All
K states use the same noise realisation. This keeps Pass 2 within Pass 1's
random-seed family, preserving the K-parallel invariant.

### 4.5 Pass 2 denoising — BMCSA over 12 fixed steps

**Input**: `x_{t*}` (K, 8, 16, 16, 16), per-state cond `c_k`, M_base_flat
(1, 1, 512, 1), sampler + flow_model.

**Schedule**: `np.linspace(t*, 0, 13)` → 12 Euler step pairs. This replaces the
v3 schedule of `ceil(total_steps × t*)` steps, making Pass 2 step count
independent of t\*.

**Per-step forward** (K=6 batch, all at same t):

```
for t, t_prev in schedule_pairs:
    # DiT forward with BMCSA flags in kwargs.
    v_cfg = pipe.sparse_structure_flow_model(
        x_t=x_t, t=t, cond=c_0..K-1, neg_cond=c_neg,
        bmcsa_flag=True,
        bmcsa_blocks=config.bmcsa_blocks,       # default: all blocks
        bmcsa_strength=config.bmcsa_strength,   # default: 1.0
        M_base=M_base_flat,
    )
    x_{t-dt} = x_t - (t - t_prev) · v_cfg
```

**BMCSA application** (inside each selected DiT self-attention layer, when
`bmcsa_flag=True`):

Given block input `h ∈ R^{K × 512 × D}`:

```
# Standard self-attention (each state attends to its own tokens):
y_self = self.self_attn(h)                              # (K, 512, D)

# Shared self-attention (each state's Q against cross-batch-mean K/V):
y_shared = self.self_attn(h, share_kv_across_batch=True)  # (K, 512, D)

# Spatial blend by M_base:
M_b = M_base.to(y_self.dtype)                           # (1, L, 1), broadcasts with (K, L, D)
effective_M = clamp(bmcsa_strength × M_b, 0.0, 1.0)
h = (1 - effective_M) · y_self + effective_M · y_shared
```

Cross-attention (to per-state image conditioning) is unchanged.

### 4.6 Output

```
z_final_v4 = x_0 (at end of Pass 2)                      # (K, 8, 16, 16, 16)
O_stack = σ(decoder(z_final_v4)) > 0.5                   # (K, 64, 64, 64)
remove_disk(O_stack)
save O_stack, z_final_v4, diagnostics
```

All K states are replaced with Pass 2 output (not just state 0).

## 5. Source Code Modifications

Four files in `mine/TRELLIS/` need edits. All follow MorphAny3D's
source-edit + kwargs pattern; **no monkey-patch, no forward hook**.

### 5.1 `mine/TRELLIS/trellis/modules/attention/modules.py`

**Class**: `MultiHeadAttention`

**Change**: `forward` method gains a `share_kv_across_batch: bool = False` kwarg.
When True and self-attention mode, K and V are averaged across the batch
dimension and expanded back, before RMS norm + SDPA.

**Diff-level description**:

```python
# Before (line 112):
def forward(self, x, context=None, indices=None) -> torch.Tensor:
    B, L, C = x.shape
    if self._type == "self":
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
        if self.use_rope:
            q, k, v = qkv.unbind(dim=2)
            q, k = self.rope(q, k, indices)
            qkv = torch.stack([q, k, v], dim=2)
        if self.attn_mode == "full":
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q); k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(qkv)
        ...

# After:
def forward(self, x, context=None, indices=None, share_kv_across_batch=False) -> torch.Tensor:
    B, L, C = x.shape
    if self._type == "self":
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
        if self.use_rope:
            q, k, v = qkv.unbind(dim=2)
            q, k = self.rope(q, k, indices)
            # If sharing K/V, mean across batch after RoPE (linearity allows either order).
            if share_kv_across_batch and B > 1:
                k = k.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).contiguous()
                v = v.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).contiguous()
            qkv = torch.stack([q, k, v], dim=2)
        elif share_kv_across_batch and B > 1:
            # No RoPE path: split, mean, restack.
            q, k, v = qkv.unbind(dim=2)
            k = k.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).contiguous()
            v = v.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).contiguous()
            qkv = torch.stack([q, k, v], dim=2)

        if self.attn_mode == "full":
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q); k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(qkv)
        ...
    # Cross-attention branch unchanged (no share_kv semantics for cross-attn).
    # ...rest unchanged...
```

**Note**: Averaging happens BEFORE `qk_rms_norm` (which normalises per-vector).
This is a deliberate choice: we want to normalise the averaged feature, not
average normalised features (the two are different and the former is more
stable for attention scales).

### 5.2 `mine/TRELLIS/trellis/modules/transformer/modulated.py`

**Class**: `ModulatedTransformerCrossBlock`

**Change**: `_forward` method receives `**kwargs` (it already did via `**kwargs`
inheritance), and branches on `kwargs.get("bmcsa_flag")` around the self-attn
call. The cross-attn branch is unchanged.

**Diff-level description** (analogous to MorphAny3D's `ss_tfsa_flag` pattern):

```python
# Current (paraphrased):
def _forward(self, x, mod, context, **kwargs):
    # ...modulation setup...
    h = self.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    h = self.self_attn(h)                              # ← only branch
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    # ...
    h = self.norm2(x)
    h = self.cross_attn(h, context=context)
    # ...

# After:
def _forward(self, x, mod, context, **kwargs):
    # ...modulation setup...
    h = self.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

    # --- BMCSA self-attention ---
    bmcsa_flag = kwargs.get("bmcsa_flag", False)
    block_idx  = kwargs.get("block_idx", None)
    bmcsa_blocks = kwargs.get("bmcsa_blocks", None)
    if bmcsa_flag and (bmcsa_blocks is None or block_idx in bmcsa_blocks):
        y_self   = self.self_attn(h)                                   # (K, L, D)
        y_shared = self.self_attn(h, share_kv_across_batch=True)       # (K, L, D)
        M = kwargs["M_base"]                                           # (1, L, 1) pre-shaped
        strength = kwargs.get("bmcsa_strength", 1.0)
        eff_M = torch.clamp(strength * M, 0.0, 1.0).to(y_self.dtype)
        h = (1.0 - eff_M) * y_self + eff_M * y_shared
    else:
        h = self.self_attn(h)
    # -----------------------------

    h = h * gate_msa.unsqueeze(1)
    x = x + h

    # Cross-attention unchanged
    h = self.norm2(x)
    h = self.cross_attn(h, context=context)
    h = h * gate_mlp_or_whatever...
    # ...rest unchanged...
```

### 5.3 `mine/TRELLIS/trellis/models/sparse_structure_flow.py`

**Class**: `SparseStructureFlowModel`

**Change**: `forward` accepts `**kwargs` and threads them through each block
along with a `block_idx` identifier.

**Diff-level description**:

```python
# Before:
def forward(self, x, t, cond):
    # ...patchify + pos_emb...
    for block in self.blocks:
        h = block(h, mod, cond)        # plain
    # ...unpatchify...

# After:
def forward(self, x, t, cond, **kwargs):
    # ...patchify + pos_emb...
    for i, block in enumerate(self.blocks):
        h = block(h, mod, cond, block_idx=i, **kwargs)
    # ...unpatchify...
```

`block_idx` lets `_forward` selectively apply BMCSA to a subset of blocks when
`bmcsa_blocks` kwarg is set.

### 5.4 `mine/pipelines/stage_b_scar.py`

**New module constants / functions**:

- `_compute_M_base_tokenspace(soft_p1, tau_M)`: constructs M_base_flat from
  `soft_p1[1:]` (s1-5 mean), downsampled to token resolution.

- `_sdedit_refine_k6_bmcsa(...)`: new Pass-2 function that
  1. Builds `guides` = per-state `z_guide_k`
  2. Computes `x_{t*} = (1-t*)·guides + σ_{t*}·ε_shared`
  3. Runs 12 Euler steps calling `sampler.sample_once(flow_model, sample, t,
     t_prev, cond, neg_cond, cfg_strength, cfg_interval, bmcsa_flag=True,
     M_base=M_base_flat, bmcsa_blocks=cfg_blocks, bmcsa_strength=cfg_strength)`.
     The sample_once and its inherited `_get_model_prediction` will forward
     kwargs into `model(x_t, t, cond, **kwargs)`.
  4. Returns `z_refined` (K, 8, 16, 16, 16).

- `run_scar` integration:
  - Remove the `_sdedit_refine_k6` + v3 augmented-intersection Pass 2 call
  - Add the v4 path gated by `cfg_sdedit.get("mode") == "bmcsa"` (or similar)
  - Replace ALL K states with `z_refined` (not just the listed `refine_states`)

- Config: `configs/v1.yaml` gains
  ```yaml
  stage_b_sdedit:
    mode: bmcsa                 # v3 | bmcsa. default bmcsa
    t_star: 0.5
    pass2_steps: 12             # fixed, not derived from t_star × total_steps
    bmcsa_strength: 1.0
    bmcsa_blocks: all           # "all" | list[int]
    tau_M: 0.05                 # M_base sigmoid sharpness
  ```

**Inheritance note on `sample_once`**: `SCARSampler` inherits `sample_once`
from `FlowEulerGuidanceIntervalSampler` (which inherits
`GuidanceIntervalSamplerMixin`). The mixin overrides `_inference_model` to
handle CFG + interval; it passes `**kwargs` through to `super()._inference_model`
which ultimately calls `model(x_t, t, cond, **kwargs)`. No changes needed to
the sampler chain; the kwargs mechanism is already transparent.

## 6. Default Hyperparameters

| Parameter | Default | Rationale |
|---|---|---|
| `mix_steps` | 8 | Pass 1 mix, unchanged |
| `extreme_mix_mode` | `symmetric` | Pass 1 mix, unchanged |
| `mix_weights` | `(0.3, 0.4, 0.3)` | Pass 1 mix, unchanged |
| `alpha_peak` | 0.0 | Push off |
| `t_star` | 0.5 | Balance guide signal vs c_k refinement |
| `pass2_steps` | 12 | Fixed; independent of t\* |
| `tau_M` | 0.05 | M_base sigmoid sharpness (tight around 0.5 threshold) |
| `bmcsa_strength` | 1.0 | Full blend at base voxels |
| `bmcsa_blocks` | `all` | Apply to every DiT block's self-attn |
| `guide_mode` | `augmented_intersection` | Preserve per-state drawer hints |

All other sampler params (`cfg_strength`, `cfg_interval`, `rescale_t`) inherit
Pass 1 defaults.

## 7. Failure Modes and Mitigations

### 7.1 `M_base` misplacement

**Failure**: If Pass 1's s1-5 agreement is off (e.g. one of states 1-5 is a
video-diffusion outlier), M_base marks wrong regions as "base". BMCSA shares
attention at those wrong regions, propagating error to all K states.

**Detection**: visualise `M_base_64` pre-downsample. Expect it to look like
cabinet + interior walls, not extending into drawer regions.

**Mitigation**: switch M_base to `min` instead of `mean` over s1-5 (stricter),
or fallback to `guide_mode = pure_intersection` (drops per-state excl).

### 7.2 `share_kv_across_batch` OOD

**Failure**: DiT self-attn was trained with per-sample K/V; feeding mean-across-
batch K/V is distribution-shift. y_shared may produce garbage features.

**Detection**: Pass 2 output with `bmcsa_strength=0` (disable) should match
v3 SDEdit (sanity check). With `strength=1`, compare per-layer feature
statistics of y_self vs y_shared; large divergence in norm/variance suggests
OOD.

**Mitigation**: reduce `bmcsa_strength` (e.g., 0.5 = half blend); restrict
`bmcsa_blocks` to middle layers only (blocks 4-8 of 12); use a RMS-normalised
mean (normalise each K/V before averaging).

### 7.3 Pass-1 middle-state drawer overlap

**Failure**: If adjacent states have overlapping drawer voxels (small joint
range), `P_excl_k` shrinks for middles, guide has partial drawers, SDEdit
must regenerate from c_k. If c_k image signal on drawer is weak, drawer
becomes incomplete in Pass-2 output.

**Detection**: compare per-state drawer voxel counts Pass 1 vs Pass 2. Loss >
20% indicates regeneration failure.

**Mitigation**: swap `max_{j != k}` to `second_max_{j != k}` in P_excl (more
permissive); or use a learned exclusivity proxy (deferred to v5 if needed).

### 7.4 Encoder OOD on synthetic occupancy

**Failure**: `E(augmented_intersection)` may produce latent outside training
distribution, leading to abnormal x_{t\*} initialisation.

**Detection**: compute `||z_guide^(k)||` and compare to `||z^(k)_final_p1||`
magnitudes (should be same order). Also check `D(z_guide^(k))` looks like
the input `O_guide^(k)`.

**Mitigation**: for serious OOD, fall back to `z_guide^(k) = z^(k)_final_p1`
(identity guide). SDEdit + BMCSA still works without the augmented guide,
just without the explicit base/excl separation.

### 7.5 BMCSA over-homogenisation at move boundary

**Failure**: `M_base` is continuous (sigmoid); voxels at the base/move boundary
have `M ≈ 0.5`, so they receive 50% shared attention — potentially
homogenising move-boundary features that should be per-state.

**Detection**: visualise final state-k drawer relative to state-(k+1) drawer.
If boundaries look "averaged", this is happening.

**Mitigation**: sharper `tau_M` (e.g., 0.02 → steeper transition). Or threshold
M to {0, 1} hard gate.

## 8. Ablation Plan (for CVPR / AAAI paper)

| Ablation ID | Configuration | Purpose |
|---|---|---|
| A0 | Plain K-parallel (no mix, no push, no BMCSA) | Lower bound |
| A1 | Pass 1 only (mix + push) | Legacy SCAR baseline |
| A2 | Pass 1 (mix, no push) + no Pass 2 | Mix-only baseline |
| A3 | v3 SDEdit (Pass 1 mix + guide + SDEdit, no BMCSA) | SDEdit-only baseline |
| A4 | v4 BMCSA (Pass 1 mix + guide + SDEdit + BMCSA) | **Full method** |
| A5 | v4 with `bmcsa_blocks = middle half only` | Layer selectivity |
| A6 | v4 with `bmcsa_strength = 0.5` | Blend strength |
| A7 | v4 with `M_base = min_{1..K-1}` instead of mean | Base aggregation op |
| A8 | v4 with `M_base` from ALL K states (mean over 0..K-1) | Ablate state-0 exclusion |
| A9 | v4 with different t\* ∈ {0.3, 0.5, 0.7} | SDEdit noise level sweep |
| A10 | v4 without `P_excl` (guide = P_base only for all k) | Ablate per-state hint |
| A11 | v4 with inference-time MCA cache (for temporal consistency across states) | Future extension (MorphAny3D-style ordering) |

Expected story: A0 < A2 ≈ A3 < A4; A5, A6, A7, A8 close to A4 but each slightly
weaker; A9 shows t\* choice matters; A10 tests whether per-state hints are
needed or base guide alone is enough.

## 9. Implementation Tasks (handoff to writing-plans)

- **T1** — Edit `modules/attention/modules.py`: add `share_kv_across_batch` to
  `MultiHeadAttention.forward`. Unit test: forward with B=6, verify that with
  `share_kv=True`, output at batch index 0 is independent of variations in the
  other 5 inputs' K/V projections (they should all average in).
- **T2** — Edit `modules/transformer/modulated.py`: add BMCSA branch to
  `ModulatedTransformerCrossBlock._forward`. Unit test: with `bmcsa_flag=False`,
  output matches current master. With `bmcsa_flag=True` and `M_base=0`
  everywhere, output matches `bmcsa_flag=False`. With `M_base=1` everywhere,
  output equals `self.self_attn(h, share_kv_across_batch=True)` mod gating.
- **T3** — Edit `models/sparse_structure_flow.py`: thread `**kwargs` through
  `forward` with `block_idx=i`. Unit test: forward without bmcsa kwargs unchanged;
  with bmcsa kwargs, block_idx visible in each block's call.
- **T4** — Implement `_compute_M_base_tokenspace` in `stage_b_scar.py`. Unit test:
  given synthetic `soft_p1` where s1-5 share a cuboid, output M_base at token
  resolution has 1s inside, 0s outside.
- **T5** — Implement `_sdedit_refine_k6_bmcsa` in `stage_b_scar.py`. Unit test:
  runs end-to-end on a mock sampler + model; output shape `(K, 8, 16, 16, 16)`,
  does not NaN.
- **T6** — Rewire `run_scar` Pass 2 dispatch to BMCSA path when `mode == "bmcsa"`.
  Keep v3 path available via `mode == "v3"`.
- **T7** — Update config (`configs/v1.yaml`) with new BMCSA block; update
  `run_v1.py` to pass cfg through.
- **T8** — Smoke test: run Pass 1 + Pass 2 with BMCSA on real pipeline, verify
  no crash, save diagnostics.
- **T9** — Add visualisation: `viz/bmcsa/M_base_64.html`, `viz/bmcsa/M_base_8.html`,
  `viz/bmcsa/pass2_per_step/` (K state occupancy per step).
- **T10** — Self-review + optional: performance profiling (BMCSA overhead
  estimate). Expected: ~1.5–2x per self-attn layer cost, ~1.3–1.5x total
  Pass-2 cost. Pass 2 has 12 steps vs Pass 1 25 steps, so overall <2x of
  v3 total.

## 10. What this spec does NOT include (out of scope)

- **Not implemented in v4**: TFSA-style temporal caching across frames. Our
  sampling is K-parallel not sequential; there's no "prev frame" concept to
  cache. If we later serialise state sampling, TFSA-style caching across state
  ordering could be added (deferred to v5 if needed).
- **Not implemented**: cross-attention BMCSA. Cross-attn conditions on per-state
  DINOv2 `c_k`; sharing across states would muddy per-state conditioning.
  Self-attn is the right layer to share.
- **Not implemented**: DDIM inversion for exact starting point (MorphAny3D uses
  3D inversion via VoxHammer). We stick with SDEdit's forward-noising which is
  simpler and empirically works for our case.
- **Not implemented**: dynamic M_base that updates during Pass 2. M_base is
  computed once from Pass 1 and held static throughout the 12 steps.
- **Removed vs v3**: `refine_states` config is deprecated. BMCSA by design
  refines all K states simultaneously; there's no meaningful "subset"
  interpretation.

---

End of v4 spec.
