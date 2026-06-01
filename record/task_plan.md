# Task Plan: Stage E Texture (Wan2.2 Inpainting + W-RFSDS) over FreeArt3D outputs

## Goal
Build a working post-StageD **texture optimization stage** using Wan2.2 **inpainting** + W-RFSDS that takes FreeArt3D's existing geometry / part segmentation / motion trajectory as input and produces materially better texture (under FreeArt3D's own evaluation tool) on `outputs/origin/30857`.

Stage A through Stage D (geometry + part split + joint) are owned by ANOTHER agent on this repo. I do **not** modify Stage A–D; I consume their outputs (or FreeArt3D's outputs in the test case).

## Phases
- [x] Phase 0 — Ground-truth audit (proposal / code / artifacts / eval) — DONE
- [x] Phase 1 — Parallel sub-agent decomposition (4 Agent-tool sub-agents) — DONE
- [x] Phase 2 — Critical proposal audit under CCF-A standard — DONE (see notes.md)
- [ ] Phase 3 — Stage E pipeline design (mesh-texture + Fun-InP + W-RFSDS) — design drafted in notes.md; awaiting metric-strategy confirm
- [ ] Phase 4 — Implement Stage E (no shims/try-except/chinese; emit viz)
- [ ] Phase 5 — Run on FreeArt3D's outputs/origin/30857 (their meshes + joint) — REMOTE (H800)
- [ ] Phase 6 — Re-run FreeArt3D eval + added texture-sensitive metrics; compare vs baseline
- [ ] Phase 7 — Iterate; mechanism-targeted (open-state interior / inlay) until measurably better

## HARD TRUTH (Phase 2 finding)
FreeArt3D baseline **clip_sim=0.913** on 30857 (0.876 on 7201). CLIP ViT-L/14@336px global-embedding
image-image cosine is SATURATED and texture-insensitive. "Significantly beat CLIP" is fragile; must add
LPIPS/PSNR/SSIM on the pixel-aligned eval renders + region crops as the real evidence. See notes.md.

## Key Questions
1. Does `mine/pipelines/stage_a_fun_inp.py` already implement a Wan inpainting path? What does "inp" actually mean in this repo?
2. What exact artifacts does FreeArt3D produce in `outputs/origin/30857/sds_output/` (per-part meshes? URDF? per-state poses?)
3. Where is FreeArt3D's evaluation tool? What metrics does it report?
4. What does the CURRENT `record/method.md` say about Stage F texture (post-StageD)? Is "Wan2.2 inp + W-RFSDS" already specified, or is this a new direction?
5. Is the planned texture target the canonical SLAT, mesh UV atlas, or per-Gaussian SH? (P2 in method.md says SLAT but mentions atlas export.)
6. Camera convention: FreeArt3D renderings vs TRELLIS canonical world up vs Wan2.2 native — what is the exact match?

## Decisions Made
- (will fill after Phase 1)

## Errors Encountered
- (will fill as discovered)

## USER DECISIONS (2026-05-29)
- Metric: KEEP FreeArt3D CLIP for comparability, but PRIMARY evidence = LPIPS/PSNR/SSIM on the
  pixel-aligned eval renders + inlay/interior region crops. Mechanism-targeted (gain at open-state interior).
- Target objects: **7128 first, then 7201** (NOT 30857). 30857 = local audit case only.
- These objects are NOT local (only 30857 is) -> 7128/7201 FreeArt3D outputs live on H800 server.

## Status
**Phase 3/5 — DISCOVERED prior texture stage on H800, audited, found + root-caused the blocker.**
`scripts/wan_texture_optimize.py` (L1-to-Wan, nvdiffrast) is BLOCKED on camera misalignment (proven by
camera_metrics.json + mask strip); optimization never ran. Root cause: supervised against FreeArt3D
carpet PNGs with a non-matching hand-rolled camera. FIX = self-consistent camera (render endpoints with
the optimizer's own camera -> regen Wan video -> optimize). See notes.md "PRIOR IMPLEMENTATION AUDIT".
NEXT: implement self-consistent-camera fix; run camera re-check on 7128; then optimize + eval.
GPU note: other agent's sd_opt_f (5762) live on WM-H800-02 (8 GPUs; coordinate).

## PROGRESS 2026-05-29 (execution)
- ROOT-CAUSED + FIXED the texture-scramble bug: `wan_texture_optimize.py` merge_parts had a WRONG
  `uv[:,1] = 1.0 - uv[:,1]` V-flip. GPU debug job 5765 rendered 3 variants; **noflip = clean microwave**
  (matches FreeArt3D), uvflip/texflip = scrambled. nvdiffrast here uses glTF top-left UV origin -> no
  flip. Fix = delete the flip line (1 line). Pushed to server scripts/wan_texture_optimize.py.
- Confirmed atlas legit (1024x1024, recognizable unwrap) + material->image identity map.
- LAUNCHED self-consistent pipeline job 5766 (tex_pipeline_7128.sbatch): STEP1 clean endpoint renders ->
  STEP2 Fun-InP from OUR renders (camera-aligned by construction, no carpet) -> STEP3 wan_texture_optimize
  (corrected UV). Out: outputs/origin/7128/{wan_fun_inp_selfcam, texture_opt_selfcam}.
- AFTER: inspect compare_keyframes.png + metrics.json; then run FreeArt3D evaluate_partnet.py on baseline
  vs optimized (CLIP + add LPIPS/PSNR/SSIM). Need PartNet GT meshes location (not under mine/datasets;
  gt_render sbatch references /lustre/1230003454/hf_models/PartNet via blenderproc render_partnet_mesh.py).
- STILL TODO per user: swap L1 -> w_rfsds_loss (fun_inp path) and compare; multi-view supervision.

## PROGRESS 2026-05-29 (cont.) — eval harness
- FreeArt3D eval needs bpy; compute node lacked libxkbcommon.so.0. FIX: LD_LIBRARY_PATH=
  /lustre/1230003454/env/fa3d2/lib (conda-bundled). Eval job 5770 now RUNNING past bpy import
  ("Evaluating 7128 with ours"), iterating 6 qpos (ICP+Cycles render+CLIP), ~30-40min.
  Eval tool: /lustre/1230003454/code/fa3d/origin/evaluate_partnet.py; env fa3d2; res ->
  .codex_run/eval_res_base/7128.json (baseline) + eval_res_opt/7128.json (L1 optimized).
- AWAITING: 7128 baseline clip vs L1-optimized clip (+ fscore/joint). Expect L1 middle-state smearing
  may NOT beat baseline -> motivates W-RFSDS. Then implement W-RFSDS texture optimizer (contract above).
- W-RFSDS texture optimizer not yet written. Next coding task once eval numbers are in.

## PROGRESS 2026-05-29 (cont.2) — W-RFSDS engine + autonomous loop
- EVAL DONE (5770): baseline 7128 clip_sim=0.90814; L1 result 0.87925 (REGRESSED -0.029). Bar = 0.90814.
- Wrote scripts/wan_texture_wrfsds.py (W-RFSDS, t5_cpu, no offload, Kabsch screw trajectory, 21-frame,
  anchors+TV+reg). Smoke job 5773 launched (mine env). Background wait-agent a5c76771 polling 5773 ->
  will pull cross-check imgs to .server_pull\smoke_* + report mem/loss/traceback.
- BUG fixed (T5-to-CPU ordering: must move T5 to cpu BEFORE encode_prompt(device=cpu)). Re-pushed module,
  but 5773 already started with old code -> expect crash at build_fun_inp_cond_t5cpu; resubmit fixed after.
- AUTONOMOUS LOOP (user away): smoke -> fix -> full run -> eval(via fa3d2 sbatch like eval_7128) -> if
  single-view doesn't beat 0.908, add MULTI-VIEW (sample camera per iter) -> rerun. MUST beat 0.90814.
- Wait-agent pattern: plink -batch -hostkey 'SHA256:urXhG4PK9+7bKeHlMQD9A4Y815+rbQ0Le47jxEZ50u8' -pw
  '2004003023@gch' 1230003454@10.100.16.13 ; server-side `for i in $(seq..); do squeue -j JID|grep -q JID
  && sleep 10 || break; done` then cat results. eval pred dir trick: symlink {pred}/7128/sds_output -> out/sds_output.

## SMOKE ITERATIONS (W-RFSDS engine validation)
- 5773: crashed T5 device mismatch (encode_prompt with T5 on GPU). FIXED: move text_encoder.to(cpu) at
  start of build_fun_inp_cond_t5cpu, encode with device=cpu. Trajectory verified PERFECT (Kabsch rmsd 0,
  91.4deg revolute; frame00==qpos_closed byte-identical; opening sequence visually correct).
- 5775 (512/21): T5 fix worked (load peak 53.6GB, ~25GB free), reached iter 1, OOMed in loss.backward()
  through VAE encode by ONLY 288MiB on 79GB H800. => VAE-encode-backward is the wall; t5_cpu necessary
  but not sufficient at 512/21. (No-offload keeps 2 DiT experts=28GB resident, fixed.)
- 5776 (384/13): submitted. Cuts VAE-backward mem ~0.35x -> expect peak ~60-63GB (margin ~16GB). Atlas
  stays 1024^2. wait-agent a26c4d47 polling. wrfsds_full.sbatch already staged at 384/13 (iters 300),
  submit when 5776 confirms no-OOM + finite decreasing loss.
- NEXT after full run: eval via fa3d2 sbatch (symlink pred), compare clip_sim vs 0.90814. If single-view
  insufficient -> MULTI-VIEW (sample camera per iter). MUST beat 0.90814.

## STATUS: smoke 5776 PASSED (peak 63.2GB, losses finite+down, SDS-dominated). FULL single-view 5777
## RUNNING (300 iters, wait-agent a290ac03 polling + pulling opt traj/atlas). eval_wrfsds.sbatch STAGED.

## DECISION TREE after single-view W-RFSDS eval (bar clip_sim=0.90814):
## 0. FIRST inspect pulled opt renders (wrf_*) — texture changed sensibly, NOT smeared/corrupted like L1?
## 1. clip_sim > 0.90814 -> WIN. Confirm fscore unchanged. Add LPIPS/PSNR/SSIM (user primary metric) + run 7201.
## 2. clip_sim ~ 0.908 (texture barely moved) -> loosen (lr^, anchor/reg v, iters^) AND/OR MULTI-VIEW
##    (refine whole atlas; single view can't move a 30-view metric). Most likely path.
## 3. clip_sim < 0.908 (regressed) -> Wan generic prior diverges from THIS object's GT on VISIBLE surfaces
##    (FreeArt3D already GT-grounded from real multi-view images). Pivot SURGICAL: strong anchor on observed
##    surfaces, refine ONLY never-observed interior texels (visibility/supervision_provenance mask). Multi-view
##    alone won't fix a wrong direction.
## STANDING RISK: FreeArt3D texture is grounded in REAL images of the object; Wan is a generic prior. The
## genuine win region is the NEWLY-EXPOSED interior at open states that FreeArt3D hallucinated as flat fill.

## ===== KEY RESULT (eval 5778): single-view W-RFSDS clip_sim=0.89824, REGRESSED -0.00990 vs 0.90814. =====
## fscore unchanged (0.93688). Renders CLEAN (not corrupted like L1) but texture barely moved (reg 0.003)
## yet STILL regressed -> CONFIRMS Wan generic prior diverges from THIS object's GT. FreeArt3D already
## GT-grounded from real images. => BROAD refinement (incl. broad multi-view) will regress more. ABANDON
## broad-MV. PIVOT: SURGICAL interior-only.
##
## SURGICAL PLAN (wan_texture_wrfsds_mv.py + --interior_only): compute per-atlas OBSERVED mask = texels
## visible when CLOSED from a dense camera set (these are FreeArt3D-observed exteriors); FREEZE them
## (mask param.grad to 0 -> stay EXACTLY at FreeArt3D init -> exterior views can't regress); optimize ONLY
## interior texels (never seen closed = cavity walls + door inner face = FreeArt3D flat hallucination) via
## multi-view W-RFSDS over the opening. Save masks for inspection. This isolates the change to the only
## region with headroom; if it STILL regresses, Wan interior < flat fill for GT -> honest negative on 7128.
## Then ALSO run 7201 (baseline 0.876, weaker FreeArt3D texture -> more headroom, W-RFSDS likelier to help).
## Report LPIPS/PSNR/SSIM (interior-sensitive, user's primary metric) alongside CLIP.
##
## HONEST META: beating CLIP 0.908 on 7128 may be infeasible (GT-grounded baseline + saturated CLIP);
## surgical interior-only is the best principled shot. 7201 is the more winnable case.

## ===== FIRST-PRINCIPLES FINDING (input structure verified) =====
## FreeArt3D input for 7128 = 6 imgs, SINGLE viewpoint, multi-state (rendering_joint_00_state_00..05;
## PartNet 00_seg..05_seg). NOT multi-view. Eval renders 30 SPHERE views (all around).
## => FRONT(+interior-from-front): FreeArt3D OBSERVED -> grounded -> hard to beat.
## => BACK/SIDES: FreeArt3D never observed (single view) -> TRELLIS-hallucinated. My Wan-InP is conditioned
##    on the SAME single front view (locked camera) -> NO back info -> W-RFSDS only supervises the front.
## => single-view W-RFSDS perturbs the grounded front (regress) + never reaches the ungrounded back.
## => WITH SINGLE-VIEW INPUT THERE IS NO NEW INFORMATION TO IMPROVE ANY REGION. Beating FreeArt3D texture
##    on 7128 via single-view Wan-InP is ~infeasible (fundamental, not tuning). This is the falsification.
##
## REMAINING PRINCIPLED SHOTS (autonomous, in order):
## 1. Identity control 5779 DONE: identity clip_sim=0.90819 vs baseline 0.90814 = delta +0.00005 (ZERO,
##    within render noise). ROUNDTRIP IS CLEAN. => single-view W-RFSDS -0.0099 is a REAL divergence, not an
##    artifact. 7128 single-view infeasibility CONFIRMED by control + first-principles + eval (triangulated).
## 2. 7201: baseline 0.876 (FreeArt3D texture WEAKER there) -> front-sharpening via Wan-InP has REAL
##    headroom -> W-RFSDS may genuinely beat. Run full W-RFSDS on 7201 + eval. THE winnable case.
##    STATUS: 7201 = stainless-steel OVEN (revolute X, 94deg, door drops down). W-RFSDS opt 5780 DONE
##    clean (peak 63.2GB, finite loss, motion rmsd 0). Renders CLEAN (metallic oven, door opens, interior
##    rack). Door atlas changed MORE than 7128 (570->1005KB PNG) = more texture movement (metallic had room).
##    Eval 5781 RUNNING (baseline+wrfsds), wait-agent aa4c6690. Decisive 7201 number pending (~1hr).
##    texture_metrics.py STAGED (scripts/): PSNR/SSIM/LPIPS on eval render pairs (run after eval, mine env:
##    python scripts/texture_metrics.py .codex_run/eval_res_base/renderings/7201 ours ; and eval_res_wrf7201).
## 3. If still no win: report HONEST finding + reframe the correct comparison for CAST-U: Wan-InP texture
##    vs SINGLE-IMAGE-ONLY texture (the user's actual setting), NOT vs FreeArt3D's view-grounded recon.
##    Also report LPIPS/PSNR/SSIM (user's primary metric) which may show front-detail gains CLIP misses.
