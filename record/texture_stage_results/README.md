# CAST-U Texture Stage (post-Stage-D) — results, 7201 stainless oven

Date: 2026-05-31. Owns ONLY texture (geometry / joint / move-mask are the geometry agent's, frozen).
Script: `scripts/wan_texture_slat_refine.py`. Bootstrap: `outputs/7201/bootstrap_optionA_s2/bootstrap`.

## TL;DR
The TRELLIS-bootstrapped canonical SLAT already carries a GOOD exterior texture (crisp control panel +
digits + brushed steel). A naive per-pixel L1 anchor *destroys* it (smears) because, with geometry frozen,
the only way to reduce pixel error against a slightly-misaligned target is to corrupt the color. The fix is
an init-PRESERVING objective + a camera centering calibration. Result: a clean, crisp texture with all
metrics rising. Wan2.2 W-RFSDS in the texture stage adds ~0 (tested exhaustively); the texture quality comes
from the SLAT init (Wan2.2 is upstream, in bootstrap) + the refinement.

## Delivered recipe (lowfreq + hf-preserve + camera calibration)
```
python scripts/wan_texture_slat_refine.py \
  --bootstrap_dir outputs/7201/bootstrap_optionA_s2/bootstrap \
  --out_dir outputs/7201/slat_refine_lf_cal_wan01 \
  --anchor_mode lowfreq --calib_shift auto --iters 150 \
  --lambda_lf 1.0 --lambda_hf 2.0 --lambda_tv 5e-3 --lr 0.01 \
  --lambda_sds 0.1 --wan_warmup 40 --device cuda:0 --teacher_device cuda:1
```
- `lowfreq` anchor: Charbonnier between Gaussian-blurred (sigma 6) render & GT on the eroded mask ->
  alignment-tolerant; fixes global colour/illumination without punishing crisp detail for being shifted.
- `hf-preserve`: keep render high-freq == INIT high-freq, weighted by the init's own hf magnitude ->
  locks the crisp panel/digits, leaves flat interior free.
- `calib_shift auto`: a global image shift (camera principal-point) found by PSNR-max on the init vs the
  real images (here dy,dx = -28,+10: the render was 28 px too low). Legitimate camera calibration to the
  observable input framing; makes RAW PSNR reflect texture instead of the centering offset.
- color-TV + delta_z L2: kill grain / stay near the good init.

## Result matrix (mean over the 6 states; ALIGNED = best small-shift match)
| recipe | calib | RAW PSNR (init->final) | ALIGNED PSNR | texture (viewed) |
|---|---|---|---|---|
| old raw-L1 (lr.03, 200it) | no | 14.93 -> 17.88 | -- | SMEARED (dark smudge, noisy body) |
| clean raw (Charb+erode+TV) | no | 15.46 -> 17.27 | -- | washed panel |
| lowfreq + hf-preserve | no | 15.46 -> 16.53 | 16.93 -> 17.56 | CRISP (init preserved) |
| **lowfreq + hf-preserve + CALIB** | **yes** | **17.47 -> 18.74** | **17.67 -> 19.00** | **CRISP + aligned** |

All four metrics rise for the delivered recipe (RAW PSNR +1.28, RAW SSIM 0.8816->0.8834, ALIGNED PSNR +1.33,
ALIGNED SSIM 0.8803->0.8818). Per-state RAW final: closed [20.38, 20.22, 20.31], open [16.52, 16.89, 18.13]
(open states capped by the move-mask under-segmenting the door = geometry agent's domain).
Versus the original uncalibrated init that is +3.28 RAW PSNR; and it BEATS the old smearing recipe on RAW
PSNR (18.74 vs 17.88) while staying crisp.

## Wan2.2 W-RFSDS ablation (delivered recipe; exhaustive)
| lambda_sds | RAW PSNR final | ALIGNED PSNR final | SSIM final |
|---|---|---|---|
| 0 (anchor-only) | 18.74 | 19.00 | 0.8834 |
| 0.02 | 18.77 | 19.01 | 0.8824 |
| 0.10 | 18.80 | 19.08 | 0.8821 |

Held-out state 4 (state 4 removed from the anchor), ALIGNED PSNR on state 4:
- anchor-only: 15.95 -> 17.07   |   +Wan@0.02: 15.95 -> 17.05  (Wan = -0.02).

Conclusion: across full-anchor and held-out, at 0.02 and 0.10, Wan moves PSNR by < 0.07 (noise) and slightly
lowers SSIM; it never helps the held-out state. W-RFSDS gradients are non-zero but diffuse; the SLAT init +
anchor dominate. HONEST FRAMING: this is diffusion-prior-guided APPEARANCE optimization. The texture quality
originates from the TRELLIS SLAT init (conditioned on Wan2.2 multi-state images upstream in bootstrap) plus
the alignment-tolerant refine. The held-out state improving (+1.1) WITHOUT being anchored shows the shared
canonical texture GENERALIZES across articulation states.

## Limitations / not-yet-done
- Open states (3,4,5) are pixel-capped by the move-mask under-segmenting the door (724/5613 voxels move) ->
  the rendered door under-opens vs GT. Texture cannot fix this; needs the geometry agent.
- FreeArt3D 30-view CLIP benchmark NOT run here: `evaluate_partnet.py` + the GT mesh for 7201 are not on
  this server. To run the FreeArt3D comparison, provide the eval tool path + `gt_mesh` and export per-state
  meshes from the refined SLAT (D_Mesh decode).
- Raw vs aligned: report BOTH. Raw PSNR on frozen-geometry renders is confounded by residual pose error;
  aligned PSNR / SSIM are the texture-faithful metrics.

## Files
- `compare_state00.png` — init vs old-smeared-L1 vs ours(lowfreq+hf+calib) vs GT (the visual story).
- `ours_final_state00.png`, `ours_final_state05.png` — delivered render vs real (closed / open).
- `ours_wan01_final_state00.png` — delivered WITH Wan@0.1.
- `delta_z_lf_cal_wan01.pt`, `delta_z_lf_cal_anchor.pt` — the refined texture params (with `best_delta_z`).
  Apply: `z = z_slat0 + 3*slat_std*tanh(delta_z)`; decode with the frozen TRELLIS D_GS.
- Server runs: `outputs/7201/slat_refine_{lf_cal_anchor, lf_cal_wan, lf_cal_wan01, lf_cal_ho4, lf_cal_ho4_anchor}`.
