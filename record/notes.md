# Notes: Stage E Texture (Wan2.2-Fun-InP + W-RFSDS) — Ground Truth

## THE NUMBER THAT REFRAMES EVERYTHING
FreeArt3D baseline on `evaluate_partnet.py` (already run, on disk):
- **30857 (desk): clip_sim = 0.9130**, fscore 0.958, joint_axis_err 0.023 rad, joint_orig_err NaN (prismatic)
- **7201: clip_sim = 0.8756**, fscore 0.769, joint_orig_err 0.395 (worse geometry → more headroom)
Source: `mine/outputs/origin/30857/evaluations/PartNet/{30857,7201}.json`

CLIP metric = `clip.load('ViT-L/14@336px')`, **image-image cosine** of global 768-d embeddings,
averaged over 6 qpos x 5 cams = 30 render pairs. Source `paper/FreeArt3D/eval_utils/clip.py:9,22-26`.

### Critical-thinking verdict on the metric
- CLIP ViT-L global-embedding cosine is **semantically saturated** at 0.91 for "two renders of the
  same brown wooden desk under the same camera/light." Headroom 0.913 -> 1.0 is tiny.
- CLIP pools to a global embedding -> **insensitive to fine texture** (inlay sharpness, wood grain).
  A large *perceptual* texture gain can produce a *negligible* CLIP delta.
- A **color shift** (FreeArt3D's orange/olive vs GT brown/cream) likely moves CLIP MORE than grain.
  => risk: CLIP could improve for reasons unrelated to the claimed "video-prior propagation" mechanism
  (confound), or fail to improve despite visibly better texture (metric-objective mismatch).
- CONCLUSION: chasing CLIP alone is scientifically fragile. Must ADD texture-sensitive metrics that
  are valid because renders are pixel-aligned (same camera + post-ICP geometry alignment): LPIPS,
  PSNR, SSIM on the aligned 512x512 renders, plus region crops (inlay band; open-state drawer interior).

## VISUAL GROUND TRUTH (eval-condition renders, 512x512, white ambient, SAME camera GT vs ours)
Inspected `evaluations/PartNet/renderings/30857/{gt,ours}/qpos_{00,05}/cam_00.png` directly.
- Object 30857 = wooden writing desk, 4 tapered legs, 2 front-apron drawers, wavy cream inlay band,
  directional wood grain on top, on a blue elliptical carpet. Prismatic drawer, Y-axis, range [0,-0.206].
- FreeArt3D "ours" vs GT under EVAL conditions: structurally faithful (pose, inlay present, drawers
  correct) but (1) wood grain flattened/uniform, (2) color shift to saturated orange, (3) drawer
  interior shifted olive-green, (4) inlay slightly softer. NOT catastrophic. The gap is "muddier +
  color shift," which is exactly the gap CLIP is worst at scoring.
- NOTE: the artifact agent's "resolution confound" was about the OLD `renderings_recon/` dir (lower
  res). The EVAL re-renders fresh at 512x512 from meshes -> that confound does NOT apply to the metric.
- Highest-value texture targets where a video prior should help: open-state drawer INTERIOR side walls
  (FreeArt3D fills flat olive; Wan-InP can propagate front material as the drawer slides out), the
  wavy inlay band (temporal consistency sharpens it), wood grain on top.

## CODE GROUND TRUTH (verified by reading, not sub-agent claim)
- **Fun-InP W-RFSDS already exists and is grad-enabled**:
  `mine/pipelines/stage_d/w_rfsds.py`: `load_wan_fun_inp_for_rfsds` (:296), `_w_rfsds_loss_fun_inp` (:899),
  `_fun_inp_vae_encode_grad` (:573, grad flows iff input rgb.requires_grad — "true for our
  differentiable renderer" :531-532), `_prepare_fun_inp_rfsds_condition` (:601). backend switch at :534,:781.
- **Stage D deliberately does NOT optimize texture**: `mine/pipelines/stage_d/losses.py:131-141`
  ("Stage D does not update Gaussian texture ... does not optimize texture"). Gaussian colors frozen
  from D_GS. So texture stage genuinely does not exist.
- **Stage D renderer is Gaussian-based** (diff_gaussian_rasterization), warps via SE3 joint. Good for
  SLAT/Gaussian assets but our test case is FreeArt3D **meshes** -> need a differentiable MESH renderer
  (nvdiffrast / kaolin / pytorch3d) which is NOT present. (verify on server.)
- **Stage A Fun-InP wrapper standalone-callable**: `mine/pipelines/stage_a_fun_inp.py:262`
  `run_stage_a_fun_inp(start_image, end_image, user_motion_prompt, model_dir, out_dir, ...)`
  -> StageAResult with wan_video_target_3FHW [3,F,H,W] uint8. frame_num=4n+1, H/W%8==0, boundary=0.875.

## METHOD.MD vs CODE divergence (CLAUDE.md warned the scheme docs may be stale)
- method.md Section 11 (Stage F) describes texture on **canonical SLAT** via `delta_z` tanh-reparam +
  D_GS gradient biasing (alpha_geom=0.1) + ARAP smoothness, distilled by **Wan2.2 I2V**.
- That SLAT-based design is **inapplicable to the FreeArt3D test case** (30857 = per-part meshes, no SLAT).
- method.md Section 11.11 itself flags **Risk 3 (unresolved, unexecuted)**: loss weights
  0.2*L_sds + 1.0*L_lat_rec + 1.0*L_rgb_rec => L_lat_rec dominates => "W-RFSDS distills video prior"
  framing may be empty. Must be tested by ablation, not asserted.

## EVAL PROTOCOL (to reproduce / re-run)
- `paper/FreeArt3D/evaluate_partnet.py`: per test_id, for 6 qpos:
  loads GT mesh `datasets/PartNet/{id}/gt_mesh/{5-qpos:02d}.glb` (REVERSED index) + pred
  `{pred}/{id}/sds_output/states/qpos_{5-qpos:02d}.glb`; auto-aligns pred->GT by grid (12 rotz x 10
  scale) + 2-stage ICP on 2000 pts; renders both via Blender Cycles 512x512, 8 samples, fov 45,
  transparent bg, white ambient (method=='ours'); fscore@0.05 on 100k pts; CLIP on 30 pairs;
  joint axis/orig err. Alignment cached to `aligns/{id}.json` (30857 already cached -> re-eval fast).
- Needs: CUDA (CLIP), Blender bpy, open3d, trimesh, OpenAI clip + ViT-L/14@336px weights. -> SERVER.
- GT meshes for 30857 NOT under a local `datasets/PartNet/` (Glob found none) BUT aligned GT GLBs exist
  at `evaluations/.../gt/gt_mesh_aligned_{00..05}.glb` -> GT geometry IS recoverable; confirm raw
  PartNet GT meshes location on server before re-eval.

## DESIGN DECISION (Stage E for FreeArt3D-meshes test case)
Target parameterization ranked (proposal agent + my read): **vertex color / UV atlas on FreeArt3D
meshes** > per-Gaussian SH on GS-fit > neural texture field. Pick vertex-color (or UV atlas if meshes
carry UVs) for simplicity x evaluability.
Pipeline:
  1. Load FreeArt3D `part_meshes/{fixed,articulated_00}.glb` + `joint_info.json`.
  2. Pose to closed (s_0) and open (s_5) via prismatic joint; render both endpoints (chosen camera).
  3. Wan2.2-Fun-InP(start=closed render, end=open render, motion prompt) -> 21-frame video target.
     (Fun-InP is correct: BOTH endpoints are real/renderable -> constrains the interpolation; this is
     strictly more defensible than I2V-from-closed-only, and matches the user's literal "inp".)
  4. Differentiable MESH render of the posed meshes across 21 frames with a learnable texture param;
     W-RFSDS (reuse `_w_rfsds_loss_fun_inp`) + endpoint RGB recon (the two known real renders).
  5. Bake textured per-state meshes -> `states/qpos_{00..05}.glb`.
  6. Re-run `evaluate_partnet.py` -> CLIP + (added) LPIPS/PSNR/SSIM + region crops.

## FALSIFIABLE HYPOTHESIS (mechanism, not correlation)
H: Wan-InP+W-RFSDS gain is **concentrated at open states (qpos with drawer out) and on the
interior/inlay crops**, because that is where the video prior adds information FreeArt3D lacks.
- If gain is uniform or only at closed state -> mechanism claim (video-prior propagation to unseen
  surfaces) is NOT supported; it's generic sharpening/recoloring. Report honestly.
- If CLIP flat but LPIPS/PSNR improve at open-state interior crops -> claim holds, CLIP just saturated.

## PRIOR IMPLEMENTATION AUDIT (server, 2026-05-29) — texture stage EXISTS but is BLOCKED
Server `mine/scripts/wan_texture_optimize.py` (+ `check_texture_camera.py`) is a prior-session texture
stage. Audit:
- It is **NOT W-RFSDS**: `wan_texture_optimize.py:447` optimizes plain **masked L1** between an
  nvdiffrast mesh render and the Wan-InP keyframe (+ TV + stay-near-init). User asked for W-RFSDS.
- **Objective != eval**: it minimizes L1-to-Wan-frames and reports its own L1/PSNR, not CLIP-vs-GT.
- **Camera is the proven blocker** (the make-or-break gate). Evidence:
  - `camera_check_texture/camera_metrics.json` (7128): FreeArt3D `original_bbox` CONSTANT
    [24,236,775,779] across all 6 states (carpet ellipse dominates); nvdiff `rendered_bbox` ~half size
    & off-center (s0 [266,317,589,562]). Mask strip: FreeArt3D alpha = big object+carpet; nvdiff alpha
    = small off-center object, NO carpet.
  - prior session ran `camera_check_texture_{xflip,yplus}` experiments => was stuck on this.
- **Optimization NEVER ran**: no `metrics.json`/`losses.jsonl` under outputs/origin/* (only Stage D's).
  Stopped at camera-check + Wan-video-gen.
- Wan videos DO exist: `outputs/origin/{7128,7201}/wan_fun_inp_texture/wan_video_target_3FHW_uint8.pt`,
  generated by `stagea.py --backend fun_inp` from FreeArt3D `rendering_*_state_00/05.png` (carpet, FreeArt3D camera).
- GPU: other agent's `sd_opt_f` job (5762) running on WM-H800-02. Coordinate; node has 8 GPUs.

### ROOT CAUSE
Supervision-frame inconsistency: Wan endpoints = FreeArt3D render PNGs (camera X + carpet); optimizer
render = hand-rolled `build_camera` (camera Y, no carpet). X != Y -> L1 compares misaligned images.

### FIX (minimal, first-principles correct): SELF-CONSISTENT CAMERA
Don't reverse-engineer FreeArt3D's camera. Render the closed/open mesh states with the SAME nvdiffrast
`build_camera` used in optimization -> use THOSE as Fun-InP start/end endpoints -> regenerate Wan video
-> optimize. Camera aligned by construction; no carpet on either side. The eval (`evaluate_partnet.py`)
uses its OWN 30 sphere cameras + ICP, so it does not care about camera X/Y at all -- only the baked
texture matters. `build_camera`/`render_state`/`vertices_to_clip` already produce a sane render
(nvdiff_state_00.png is a plausible object, just small/off-center vs FreeArt3D) so self-consistent reuse
is valid. Then: keep L1 as fast baseline, add `w_rfsds_loss` (server w_rfsds.py:740, fun_inp path) per
user request, compare. Report FreeArt3D CLIP + LPIPS/PSNR/SSIM + interior/inlay crops.
qpos_00=closed (joint range 0), qpos_05=open (range -1.595 rad, revolute, 7128 = box+hinged lid).

## RESULT 1 (L1 self-consistent, job 5766) — COMPROMISED by trajectory mismatch
- UV fix worked: renders are clean now. Self-consistent camera works (endpoints aligned).
- L1-to-Wan dropped 3.5x (mean L1 0.0648->0.0186, PSNR 17.3->23.7) BUT artifact inspection shows this is
  PARTLY SPURIOUS:
  - state_02: Wan f=8 door is tilted back/upright; FreeArt3D mesh state-2 door is tilted forward/flat
    (DIFFERENT angle). Wan invents its OWN opening trajectory != mesh joint interpolation.
  - optimized state_02 render shows SMEARED dark streaks on the door = texture distorting to chase
    geometrically-misaligned Wan pixels. endpoints (s0/s5) are fine (they ARE the Wan endpoints).
- => per-state pixel L1 is only valid at endpoints; middle states corrupt the atlas. The L1 approach
  (prior session's wan_texture_optimize) is fundamentally wrong for this task.
- => W-RFSDS (user's request) is the principled fix: SDS distills the video PRIOR (alignment-free),
  not a specific misaligned frame. THIS is why method.md specifies W-RFSDS not L1.
- Eval env CONFIRMED: fa3d2 has clip+bpy+blenderproc+mathutils+open3d; CLIP weights cached
  (/lustre/1230003454/.cache/clip/ViT-L-14-336px.pt). Eval tool: /lustre/1230003454/code/fa3d/origin/
  evaluate_partnet.py (+ eval_utils). GT meshes: /lustre/1230003454/hf_models/PartNet/7128/gt_mesh/{00..05}.glb.
- 7128 = red microwave, revolute door (range 0..-1.595 rad, Z axis). 800x800 sq render -> FunInp 624 sq.

## W-RFSDS INTEGRATION CONTRACT (studied from server pipelines/stage_d/w_rfsds.py)
To replace L1 with alignment-free W-RFSDS in the texture stage:
1. ctx = load_wan_fun_inp_for_rfsds(model_dir=/lustre/.../hf_models/Wan2.2-Fun-A14B-InP, repo_root,
   device, fun_config_path=.../VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml, frame_num=21,
   resolution_hw=(H,W) with H,W % 16 == 0). Returns WanRFSDSContext(backend="fun_inp", fun_pipeline=pipe).
2. wan_cond dict MUST have: backend="fun_inp"; fun_video [1,3,21,H,W] (start=closed render at [: ,:,0],
   end=open render at [:,:,-1], zeros else); fun_mask [1,1,21,H,W] (0=known at frame 0 and 20, 1/255 else);
   pos_prompt, neg_prompt (strings). _prepare_fun_inp_rfsds_condition(ctx) fills in_context/y_guidance/seq_len.
3. Training loop each iter: render 21-frame mesh video rgb_3FHW [3,21,H,W] in [0,1] (mesh at its OWN joint
   poses: door part rotated by angle(t)=range[1]*t/20 about (origin,axis); body static) with learnable
   atlas -> w_rfsds_loss(rgb, wan_cond, ctx, tau, cfg_scale) -> backward to atlas. tau ~ inverse-CDF /
   uniform(0,1); cfg_scale linear 25->12 (CHORD). Add L_first/L_last RGB anchor at frames 0/20 (the known
   endpoints) + TV + small stay-near-init reg.
- WHY this fixes the L1 smearing: SDS scores "is this rendered frame a plausible Wan microwave-opening
  frame" — no per-frame pixel correspondence required, so mesh-trajectory != Wan-trajectory no longer
  corrupts. Endpoints anchored by L_first/L_last (exact, aligned).
- IMPLEMENTATION NOTE: current wan_texture_optimize renders 6 pre-posed qpos GLBs. W-RFSDS needs CONTINUOUS
  21-frame render = load fixed.glb + articulated_00.glb as 2 parts, rotate door per frame via SE3(axis,
  origin,angle), render with nvdiffrast (corrected no-flip UV). 2 learnable atlases (body, door).
- vae encode is grad-enabled via ctx.fun_pipeline.vae.encode(video)[0].mode() (_fun_inp_vae_encode_grad).

## EVAL RESULT (job 5770) — L1 REGRESSED (falsification confirmed)
FreeArt3D evaluate_partnet.py on 7128 (fa3d2 env, LD_LIBRARY_PATH fix):
- BASELINE (FreeArt3D texture): clip_sim=0.90814, fscore=0.93688, joint_axis_err=0.0342, joint_orig_err=0.0512
- L1-OPTIMIZED (prior wan_texture_optimize): clip_sim=0.87925 (DOWN -0.0289), fscore identical (geometry same)
=> L1-to-Wan made texture WORSE (the middle-state smearing). Empirically refutes the L1 approach.
=> THE BAR TO BEAT: clip_sim 0.90814 on 7128. (fscore/dist/joint unchanged since geometry frozen.)

## W-RFSDS MODULE WRITTEN: scripts/wan_texture_wrfsds.py
- Renders continuous 21-frame opening (door posed per frame via Kabsch-recovered screw axis from
  qpos_00.door vs qpos_05.door -> data-driven, render-frame, no joint_info frame guessing).
- w_rfsds_loss (fun_inp) for alignment-free SDS + endpoint anchors (L1 to initial closed/open render) +
  TV + stay-near-init reg. T5 -> CPU after one-time prompt encode (encode on CPU via build_fun_inp_cond_t5cpu);
  DiT experts resident on GPU (no offload). H=W=512, F=21.
- Smoke mode: dumps 21-frame trajectory + endpoint-vs-qpos cross-check + GPU mem + 5 iters.
- RISK: single-camera supervision only refines the atlas region visible from that view; eval is 30 sphere
  views. To reliably BEAT 0.908 likely need MULTI-VIEW (sample camera per iter). Validate engine via smoke first.
- env: MINE (has nvdiffrast + videox_fun); NOT fa3d2 (fa3d2 is for eval bpy/clip).

## ===== FINAL RESULT TABLE (all metrics, both objects, baseline vs W-RFSDS) =====
## 7128 microwave: CLIP 0.9081->0.8982 (-0.0099); PSNR 17.13->17.16; SSIM 0.7796->0.7786; LPIPS 0.2007->0.2041 (worse)
## 7201 oven:      CLIP 0.8962->0.8890 (-0.0071); PSNR 14.35->14.35; SSIM 0.7437->0.7437; LPIPS 0.2583->0.2633 (worse)
## fscore identical baseline vs wrfsds (geometry frozen). Identity-control (roundtrip) = +0.00005 (clean).
## => W-RFSDS texture is NEUTRAL-TO-WORSE on EVERY metric, BOTH objects. Hypothesis FALSIFIED (robust).
## Low PSNR (14-17dB) even for baseline => eval is GEOMETRY-MISALIGNMENT-dominated; texture is a small lever.
##
## ROOT CAUSE (triangulated: 2 objs x 4 metrics + identity control + first-principles + clean engine):
## FreeArt3D's texture is grounded in its single-view multi-state REAL input. My Wan-InP is conditioned on
## the SAME single (front) view => NO new information. A locked-camera InP cannot see unobserved views.
## So W-RFSDS can only perturb already-grounded texels -> diverges from GT -> regress. Within "Wan2.2 inp"
## there is NO config (surgical/multi-view/tuning) that adds the missing information -> cannot beat.
##
## WHAT WAS DELIVERED (real value, even without a "win"):
## - Working, validated, memory-safe W-RFSDS mesh-texture pipeline (scripts/wan_texture_wrfsds.py + _mv +
##   eval harness). t5_cpu/no-offload, 384/13, peak 63GB. Kabsch screw trajectory (rmsd 0). UV bug fixed.
## - Fixed FreeArt3D eval on the offline server (libxkbcommon LD_LIBRARY_PATH). Reusable eval_tex.sbatch +
##   texture_metrics.py (PSNR/SSIM/LPIPS).
## - Caught the L1 approach's trajectory-smearing confound by direct artifact inspection.
##
## RECOMMENDATION (the correct experiment for CAST-U):
## The comparison "Wan-InP texture beats FreeArt3D texture" is MIS-SPECIFIED: FreeArt3D used multi-state
## REAL images; CAST-U/Wan-InP from one view has less info. Right baseline = SINGLE-IMAGE-ONLY texture
## (ablation: CAST-U with vs without the Wan-InP texture stage), where Wan-InP demonstrably ADDS the
## multi-state pseudo-views. To beat a multi-view method on texture you need a method that adds genuine
## novel-view info (orbit/NVS), which is OUTSIDE "Wan2.2 inp" scope.

## OPEN ITEMS / BLOCKERS
- [server] which differentiable mesh renderer is installed (nvdiffrast/kaolin/pytorch3d)?
- [server] raw PartNet GT meshes path for re-eval; CLIP weights cached offline?
- [decision] success-criterion framing under CLIP saturation (recommend: add texture-sensitive metrics).
