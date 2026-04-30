# Pipeline: Frozen TRELLIS + In-Loop RASA Adapter + Wan2.2 I2V-A14B W-RFSDS for Single-Image Articulated 3D/URDF Generation

## 0. One-line thesis

From a **single closed-state input image**, generate a physically usable articulated 3D asset by inserting a **test-time optimizable cross-state articulated adapter** inside frozen TRELLIS sparse-structure generation, bridging the TRELLIS sparse-structure/SLAT discrete support break with a **fixed-support differentiable gate**, and optimizing part assignment, single-DoF trajectory, and cross-state visible texture using **Wan2.2 I2V-A14B W-RFSDS** under a locked-camera video prior.

The method is not StageB post-processing and not StageB+StageC mechanical projection. The part and joint variables participate in the forward generation/rendering loop and receive gradients from Wan2.2 RFSDS.

---

## 1. Target problem

### Input

- One RGB image `I0` of an articulated object in a closed or resting state.
- A text prompt or short template, for example:
  - `locked-off single-camera video; the drawer slides open; camera static; object body stationary`
  - `locked-off single-camera video; the cabinet door rotates open; camera static; material consistent`
- Optional manual joint-type prior for the MVP: `prismatic` or `revolute`.

### Output

- `canonical_base_mesh.glb`
- `canonical_move_mesh.glb`
- `object.urdf`
- `joint.json` with joint type, axis, origin/pivot, range, and sampled state values.
- `texture_provenance.json` indicating which surfaces are input-visible, Wan/RFSDS-exposed, or hallucinated.
- `optimized_video.mp4` showing the predicted locked-camera articulation.
- Diagnostics: part probabilities, support gates, residual maps, RFSDS schedules, ablations.

### Scope for the first paper

- Single input image.
- Single movable part.
- Single-DoF joint: prismatic or revolute.
- Frozen TRELLIS and frozen Wan2.2 backbones.
- Test-time optimization only for inserted adapters and compact object-specific variables.

Multi-part and kinematic-tree extensions are reserved for follow-up or appendix.

---

## 2. Design principles

1. **Freeze all original backbones.**
   - TRELLIS SS-DiT, SS decoder, SLAT-DiT, SLAT decoders: frozen.
   - Wan2.2 I2V-A14B DiT and VAE weights: frozen.
   - DINO/image encoder: frozen.

2. **Do not modify original TRELLIS operators.**
   We do not edit TRELLIS attention, decoder, argwhere behavior, or sampler logic as a primary claim. We wrap the SS-DiT with a zero-initialized residual adapter and use a fixed-support relaxation around the known argwhere discontinuity.

3. **The inserted layer must be in-loop.**
   Cached-hidden PAM is not sufficient as a main method. RASA must be executed during the SS denoising/sampling forward pass so RFSDS gradients can reach adapter parameters and part logits through the generation path.

4. **Base consistency must be architectural.**
   Base equality is guaranteed by a shared canonical base branch. It is not left to attention similarity or a post-hoc projection.

5. **Wan2.2 is not a segmentation labeler.**
   Wan2.2 supplies a rectified-flow video prior. Incorrect segmentation, joint parameters, or texture should induce an RFSDS residual on the rendered video, and that residual is routed to our inserted variables through differentiable rendering.

6. **Geometry first, texture second.**
   Early iterations should restrict shortcut paths so RFSDS pressure reaches part/joint variables. Texture residuals are opened later, especially under low-noise Wan timesteps.

7. **Be honest about topology.**
   TRELLIS `argwhere(decoder(z_s) > 0)` is not differentiable. We avoid differentiating through coordinate extraction by optimizing continuous gates on a conservative sparse support canvas.

---

## 3. Public facts used by the pipeline

- TRELLIS uses a unified SLAT representation that can decode to meshes, radiance fields, and 3D Gaussians, and employs rectified-flow transformers for 3D generation [TRELLIS].
- Wan2.2 I2V-A14B is an image-to-video model with a two-expert MoE design, roughly 27B total parameters and 14B active parameters per denoising step; high-noise expert handles coarse layout/motion and low-noise expert refines detail [Wan2.2].
- CHORD introduces W-RFSDS/RFSDS to distill rectified-flow video priors into 4D representations by rendering videos, passing them through a video model, and using the velocity residual as an optimization gradient [CHORD].
- FlexiCubes was designed for gradient-based mesh optimization over an isosurface representation, but topology events remain something to handle cautiously in an SDS/RFSDS setting [FlexiCubes].

---

## 4. Pipeline overview

```text
I0 + locked-camera motion prompt
    ↓
[Phase 0] Initialization from frozen TRELLIS
    - Encode image condition
    - Run zero-adapter TRELLIS SS once or a small number of seeds
    - Build conservative fixed support S
    - Initialize SLAT / Gaussian / mesh proxies
    ↓
[Phase 1] In-loop RASA SS generation
    - K state-coded denoising trajectories share the input condition
    - RASA adapters inserted between selected SS-DiT blocks
    - RASA outputs part-aware token features, canonical part logits, and joint variables
    ↓
[Phase 2] Fixed-support differentiable gate
    - Do not differentiate through argwhere
    - Use constant support coords S
    - Each iter computes current SS decoder logits on S
    - support_gate = sigmoid(logit_S / temperature)
    - gate controls SLAT features, Gaussian opacity, SDF margin, and render contribution
    ↓
[Phase 3] Articulated state synthesis
    - canonical base B and canonical move M
    - single-DoF transform T(q_k)
    - state k = B ∪ T(q_k)M
    - base is identical by construction
    ↓
[Phase 4] Differentiable rendering
    - Use Gaussian/soft-volume renderer in early geometry phase
    - Use mesh/FlexiCubes/nvdiffrast later for final shape and export
    - Render locked-camera video Vθ
    ↓
[Phase 5] Wan2.2 I2V-A14B W-RFSDS
    - first-frame clamp/anchor
    - fixed-camera prompt and negative prompt
    - Wan VAE encode keeps gradient wrt rendered video
    - Wan DiT velocity predictor is frozen and no-grad
    - RFSDS surrogate backpropagates to RASA, part logits, joint, gate, and texture variables
    ↓
[Phase 6] Cross-state visible texture refinement
    - Keep geometry stable
    - Optimize SLAT/Gaussian color residual and texture provenance weights
    - Fuse input-visible, opened-state visible, and never-visible texture priors
    ↓
[Phase 7] Export and verification
    - threshold canonical base/move
    - decode/bake mesh and texture
    - export URDF
    - simulate joint motion
```

---

## 5. Phase details

## Phase 0 — Initialization

### Goals

- Obtain a conservative candidate spatial support for SLAT/mesh/3DGS.
- Initialize the object identity close to the input image.
- Ensure RASA starts as a no-op so the initial forward equals original TRELLIS.

### Procedure

1. Encode input image with TRELLIS image encoder/DINO path.
2. Run frozen TRELLIS SS generation with RASA residual scale set to zero.
3. Decode the initial SS latent into 64³ occupancy logits.
4. Build `SUPPORT_COORDS` without gradient:

```text
S = original occupied coords
  ∪ dilation(original occupied coords, radius=1 or 2)
  ∪ top-K uncertain/high-logit boundary coords
  ∪ optional sweep candidates from rough motion prompt
```

Do not use a raw global `logit > -3` threshold as the only policy. It may explode support size or miss thin structures.

5. Initialize SLAT/3DGS/mesh proxy on `SUPPORT_COORDS`.

### Output

- `support_coords.pt`
- `initial_occ_logits.npy`
- `initial_slat.pt`
- `initial_mesh_or_gaussian/`

---

## Phase 1 — RASA in-loop sparse-structure generation

### Goal

Insert a learnable cross-state part adapter inside the SS generation path while freezing original TRELLIS weights.

### Core adapter

RASA = **RFSDS-Optimized Articulated State Adapter**.

Recommended main version: `RASA-Core`.

- Same-voxel cross-state attention across K state-coded trajectories.
- Part-slot bottleneck with base/move/uncertain slots.
- Optional DINO slot re-attention, applied to part slots rather than all voxels.
- Zero-initialized residual injection between selected SS-DiT blocks.

Avoid making full 4096×4096 voxel self-attention the main method; keep that as an ablation.

### State construction

- K state codes, for example K=6 or K=8.
- Shared image condition for all states.
- Shared or correlated noise to prevent base drift.
- State code `s_k = k/(K-1)` tells the adapter the relative opening progress.

### Output variables

- `p_occ_64`: canonical occupancy probability.
- `p_move_64`: movable-part probability.
- `p_base_64 = p_occ_64 * (1 - p_move_64)`.
- `p_move_can = p_occ_64 * p_move_64`.
- Joint parameters for prismatic or revolute candidate.

---

## Phase 2 — Fixed-support differentiable gate

### Problem

TRELLIS converts SS logits to active SLAT coordinates via a hard threshold and integer coordinate extraction. This is not differentiable.

### Solution

Use fixed coordinates but not fixed occupancy.

Every optimization iteration:

```python
z_s = RASA_wrapped_SS_sampler(...)
occ_logits = SS_decoder(z_s)                         # differentiable wrt RASA
support_logits = gather(occ_logits, SUPPORT_COORDS)  # coords fixed, logits differentiable
support_gate = sigmoid(support_logits / T)
```

`SUPPORT_COORDS` is a constant sparse canvas. `support_gate` is the continuous existence variable.

### Gate usage

The gate should affect all renderable representations:

```text
SLAT feature_i      ← gate_i * feature_i
Gaussian opacity_i  ← gate_i * opacity_i
Gaussian scale_i    ← (0.25 + 0.75 gate_i) * scale_i
SDF_i               ← SDF_i + beta * (1 - gate_i)
```

Thus RFSDS can push a floater toward disappearance by decreasing gate or increasing SDF margin. Do not rely on Wan2.2 alone to notice small floaters.

---

## Phase 3 — Articulated state synthesis

### Canonical representation

Let:

- `B(x)` be canonical base probability or surface representation.
- `M(x)` be canonical movable-part probability or surface representation.
- `T(q_k)` be the single-DoF transform for state k.

State occupancy:

\[
O_k(x) = 1 - (1 - B(x))\left(1 - T(q_k)M(x)\right).
\]

Base is exactly identical for all states because the same `B` is used.

### Joint candidates

Run either:

- prismatic candidate only, if prompt says sliding drawer;
- revolute candidate only, if prompt says rotating door/lid;
- both candidates for a short selection window, then keep the lower-energy branch.

Do not rely on an invalid continuous limit between revolute and prismatic.

---

## Phase 4 — Differentiable rendering

### Early geometry phase

Prefer Gaussian or soft-volume rendering for smoother gradients. CHORD uses 3DGS-style representations for smooth gradient computation; the same consideration applies here.

### Late geometry/export phase

Use mesh/FlexiCubes/nvdiffrast after geometry is stable. FlexiCubes is suitable for mesh optimization and final mesh export, but early RFSDS geometry optimization should avoid relying solely on mesh topology changes.

### Camera

Use locked camera, matching the input image. Optional small camera jitter can be added only after the method is stable; it is not part of the MVP.

---

## Phase 5 — Wan2.2 I2V-A14B W-RFSDS

### Prompt template

Positive prompt:

```text
A locked-off single-camera video. The first frame is exactly the provided image.
The camera is static, no pan, no zoom, no orbit. The object body remains stationary.
Only the movable part opens with a physically plausible single-DoF motion.
Material, color, lighting, and texture remain consistent across frames.
Newly revealed interior surfaces are coherent with the object material.
```

Negative prompt:

```text
camera movement, zoom, pan, orbit, object morphing, changing identity,
new object, changing background, texture flicker, non-rigid deformation,
floating pieces, body moving
```

### Gradient rule

Wan VAE encoder must keep gradient with respect to the rendered video input.
Wan DiT teacher velocity prediction is frozen and evaluated without gradient.

```python
z = wan_vae.encode(V_rendered)          # grad ON wrt rendered video, weights frozen
eps = torch.randn_like(z)
tau = sample_tau(iter)                  # high → low annealing
z_tau = (1 - tau) * z.detach() + tau * eps

with torch.no_grad():
    v_hat = wan_i2v_velocity(z_tau, tau, image_cond=I0, text_cond=prompt, cfg=cfg)

grad = w(tau) * (v_hat - eps + z.detach())
loss_rfsds = (z * grad.detach()).sum()
loss_rfsds.backward()
```

This sends gradients to RASA, part logits, joint variables, support gates, and later texture variables.

---

## Phase 6 — Cross-state visible texture refinement

### Motivation

Single closed-state texture is insufficient for surfaces revealed only after opening, such as drawer side walls, cabinet interior, or door backs.

### Texture categories

1. **State0-visible exterior texture**
   - Anchored strongly to input image.

2. **Opened-state exposed texture**
   - Inferred through Wan2.2 fixed-camera video prior and RFSDS.
   - Optimized through SLAT/Gaussian texture residual.

3. **Never-visible texture**
   - Filled by TRELLIS SLAT prior and low-noise Wan detail prior.
   - Must not be claimed as true reconstruction.

### Multi-State Visible Texture Fusion (MS-VTF)

For each canonical surface/voxel/Gaussian `i`, define donor states:

\[
\mathcal{D}(i)=\{k \mid i \text{ is visible in state } k\}.
\]

Fuse canonical texture:

\[
C_i = \sum_{k\in\mathcal{D}(i)} w_{k,i}\,\mathrm{InvWarp}_{q_k}(C_{k,i}).
\]

Weights depend on visibility confidence, view angle, part consistency, first-frame priority, and RFSDS residual confidence.

---

## Phase 7 — Export and verification

### Export

- Threshold canonical base/move probabilities.
- Decode or extract base and move meshes.
- Bake or transfer texture.
- Write URDF with joint type, axis, origin, limits.

### Verification

- Render closed state and compare to input.
- Simulate articulation in PyBullet/MuJoCo.
- Check base stays static.
- Check move part is rigid under the predicted joint.
- Check no large floaters or disconnected artifacts.

---

## 6. Optimization schedule

Recommended MVP schedule:

```text
0–100 iters:
    adapter residual scale warmup
    strong first-frame anchor
    limited geometry changes

100–1200 iters:
    optimize RASA, part logits, support gates, joint variables
    high/mid tau RFSDS for motion and geometry
    texture residual mostly frozen

1200–1600 iters:
    geometry hardening
    optional no-grad support refresh
    low LR for joint, boundary refinement for part/gate

1600–2000 iters:
    texture refinement
    low tau RFSDS
    optimize SLAT/Gaussian color residual and MS-VTF weights
```

For A14B, use fewer frames/resolution during debugging, then scale to final resolution once gradient direction is validated.

---

## 7. Losses

Total:

\[
L = L_{\mathrm{W\text{-}RFSDS}} +
\lambda_0 L_{\mathrm{first}} +
\lambda_1 L_{\mathrm{base\text{-}static}} +
\lambda_2 L_{\mathrm{single\text{-}DoF}} +
\lambda_3 L_{\mathrm{part\text{-}reg}} +
\lambda_4 L_{\mathrm{support}} +
\lambda_5 L_{\mathrm{texture}} +
\lambda_6 L_{\mathrm{adapter}}.
\]

Key terms:

- `L_first`: pixel/mask/DINO/LPIPS anchor for frame 0.
- `L_base-static`: prevents state-dependent base changes.
- `L_single-DoF`: rigidity, monotonicity, axis/range priors.
- `L_part-reg`: volume ratio, TV/Potts, entropy annealing, connectedness.
- `L_support`: sparsity and gate margin over fixed support.
- `L_texture`: state0 photometric anchor, cross-state texture consistency, low-noise detail prior.
- `L_adapter`: residual magnitude and LoRA/RASA scale regularization.

---

## 8. Milestones

### M0 — RFSDS gradient-direction sanity

Before full implementation, verify on synthetic or PartNet-style assets:

- wrong mask should receive gradient toward including the true movable region;
- wrong axis should receive gradient toward better motion;
- static video should receive motion-inducing residual under the motion prompt;
- camera-moving shortcut should be suppressed by locked-camera prompt;
- texture flicker should be penalized in low-noise RFSDS.

Scalar loss separation is insufficient. Gradient direction must be tested.

### M1 — Single drawer or door, one joint

- RASA-Core only.
- Fixed support + differentiable gate.
- Geometry phase only.
- Output base/move/joint.

### M2 — Texture and export

- Add MS-VTF.
- Add SLAT/Gaussian texture residual.
- Export URDF and simulate.

### M3 — Full benchmark and ablation

- PartNet-Mobility / GAPartNet / selected real images.
- Compare with retrieval, supervised, diffusion, and video-prior baselines.

---

## 9. Ablations

Required:

1. No RASA.
2. Cached-hidden PAM instead of in-loop RASA.
3. RASA-Core.
4. RASA-Core + DINO slot re-attention.
5. RASA-Full.
6. Fixed support only, no differentiable gate.
7. Gate with no support refresh vs optional no-grad refresh.
8. Gaussian/soft-volume early render vs mesh-only render.
9. No Wan RFSDS.
10. RFSDS without locked-camera prompt.
11. No first-frame clamp.
12. No texture fusion.
13. Prismatic-only vs revolute-only vs branch selection.

---

## 10. Metrics

- Base consistency IoU.
- Movable part IoU.
- Joint type accuracy.
- Joint axis angular error.
- Joint origin/pivot error.
- Range error.
- Closed-state image similarity.
- Articulated-state Chamfer / F-score.
- Floater count / disconnected artifacts.
- URDF simulation validity.
- Texture LPIPS/CLIP/DINO consistency.
- Runtime and GPU memory.

---

## 11. Paper claims to make and not make

### Safe claims

- The method uses a frozen TRELLIS 3D prior and a frozen Wan2.2 I2V video prior.
- It inserts a test-time optimizable cross-state articulated adapter inside TRELLIS SS generation.
- It avoids differentiating through coordinate extraction by optimizing continuous gates over a fixed sparse support canvas.
- RFSDS gradients are routed to inserted part/joint/texture variables through differentiable rendering.
- Base consistency is architectural, not post-hoc.

### Unsafe claims

Do not claim:

- argwhere is made differentiable;
- hidden geometry is recovered as ground truth;
- Wan2.2 directly provides segmentation labels;
- all hidden texture is true reconstruction;
- arbitrary multi-joint kinematic trees are solved in the MVP.

---

## 12. References

- [TRELLIS] Structured 3D Latents for Scalable and Versatile 3D Generation, Microsoft Research / CVPR 2025: https://www.microsoft.com/en-us/research/publication/structured-3d-latents-for-scalable-and-versatile-3d-generation/
- [Wan2.2] Wan-AI/Wan2.2-I2V-A14B model card: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- [CHORD] Choreographing a World of Dynamic Objects: https://huggingface.co/papers/2601.04194
- [FlexiCubes] Flexible Isosurface Extraction for Gradient-Based Mesh Optimization: https://research.nvidia.com/labs/toronto-ai/publication/2023_siggraph_flexicubes/
