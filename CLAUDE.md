# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Two pipelines coexist in this repo:

1. **FreeArt3D (upstream baseline, SIGGRAPH Asia 2025)** — `run_two_parts.py` / `run_multi_parts.py`. Multi-view segmented images of an articulated object at several joint states → coarse mesh via TRELLIS → joint init via GIM-DKM + RANSAC → SDS refinement → URDF.
2. **v1 (AAAI submission, active development)** — `run_v1.py` + `configs/v1.yaml`. The research direction has shifted to: **single closed-state image (state-0) → canonical part-decomposed 3D + single-DoF joint + URDF**, with Wan2.2 I2V acting as a pseudo-multi-state oracle. Stages A→F (see Architecture below). Method/pipeline drafts live under `record/method.md`, `record/pipeline.md`, `record/target.md`.

User-facing guidance in `record/readme.md` is the source of truth for code-style / scheme-style rules — read it before editing.

## Environment

```bash
conda activate mine
```

`setup.sh` installs torch/kaolin and builds local deps. For non-default CUDA/torch, install a matching [kaolin](https://github.com/NVIDIAGameWorks/kaolin). GIM-DKM checkpoint must be placed at `gim/weights/` (FreeArt3D path); Wan2.2 weights are loaded from a local checkpoint dir (Stage A sets `HF_HUB_OFFLINE=1` before any HF import — never remove this guard).

## Architecture — FreeArt3D (legacy entrypoints)

`run_two_parts.py` / `run_multi_parts.py` load `configs/default.yaml` and sequence:

1. `pipelines/recon.run_recon` — TRELLIS-based base-mesh recon from `{i:02d}_seg.png` (or PartNet naming `rendering_joint_{j:02d}_state_{i:02d}.png`).
2. `pipelines/estimate.run_estimate` — GIM-DKM + RANSAC joint init (type from CLI or `configs/partnet.json`).
3. `pipelines/sds.run_sds` — Core SDS optimization against TRELLIS prior + voxel/intersect losses. Writes `sds_output/{part_meshes,states,joint_info.json}`.
4. `pipelines/urdf.write_urdf` — Pack into `outputs/{name}/output.urdf`.

`--stage {all,recon,estimate,sds}` re-runs a single stage against intermediates in `outputs/{name}/`. PartNet runs may need `cfg.ransac_threshold` tuned (0.03 / 0.1); `WashingMachine` overrides `cfg.min_azi/max_azi` to avoid blind spots. Custom `--input_dir` without `05_seg.png` is treated as PartNet-style (`run_two_parts.py:31`).

## Architecture — v1 (active, AAAI)

`run_v1.py` is the single entrypoint. `run_stage_bc.sh` is a wrapper that stops after Stage C. Layout under `outputs/{name}/`: `inputs/`, `stage_b/`, `stage_c_sajo/`, `stage_d_placeholder/`, `stage_f/`. Stages:

- **Stage A — `pipelines/stage_a_wan.py`**: Wan2.2 I2V (vendored at `Wan2.2/`). Single-seed (seed=42, F=21, 288×512), fixed `guide_scale=5.0`. Prompt is composed in `pipelines/wan_helpers/prompts.py` (`build_articulated_prompts` — user describes part motion, code appends the universal camera-lock addon + neg list that explicitly omits "static/motionless"). Optical-flow background-static sanity check in `pipelines/utils/optical_flow.py`. Stage A is not yet wired into `run_v1.py` (Stage B currently consumes pre-existing segmented images via `cfg.io.image_pattern`).
- **Stage B — `pipelines/stage_b_scar.py`** (default; legacy `stage_b_vgcf.py` retained for ablation): Pass-1 K-parallel SCAR sampler (`trellis/pipelines/samplers/SCARSampler`) with symmetric mix on `x_t` for steps 0..`mix_steps-1`; Pass-2 SDEdit from `t_star` with BMCSA (Base-Masked Cross-State Attention) inside DiT self-attn (`cfg.stage_b_sdedit.mode = bmcsa`). Outputs `O_stack.npy [K,64,64,64]`, `z_final.pt`, `dit_hidden.pt` (mid-late blocks for downstream).
- **Stage C — `pipelines/stage_c_sajo.py`** (driver) with internals in `pipelines/sajo/{anchors,bic,em,init,screw,warp}.py`. Joint-free soft base/move split → contact anchors → dual revolute/prismatic EM → BIC selection. Outputs `M_base`, `M_move`, `T_k`, `joint_info.json`, viz HTMLs. An alternative, newer Stage C lives under `pipelines/stage_c_segmatch/` (segmentation+matching axis refinement, material classifier, ICP warm-start, swept-volume fit).
- **Stage D — `pipelines/stage_d_placeholder.py`**: placeholder that runs a single standard TRELLIS Stage-2 sampler on `O_0` (no dual Stage-2 yet). Real Stage D / E (ACTF fusion) are deferred per `configs/v1.yaml` header.
- **Stage F — `pipelines/stage_f_assemble.py`**: vertex-level mesh split by SAJO masks (`pipelines/mesh_split.py`) → URDF assembly → pybullet validation (penetration depth check at sampled qpos).

Stage selector: `--stage {all,b,c,d,f}` resumes from existing artifacts. Sampler selector: `cfg.stage_b.sampler ∈ {scar, bcac, vgcf}`. Joint type can be forced via `--joint_type {revolute,prismatic}` to skip BIC.

## Vendored code — do not treat as ours

- `TRELLIS/` — the original Microsoft TRELLIS, including FlexiCubes mesh rep. We add custom samplers (`SCARSampler`, `BCACSampler`, `VGCFSampler` in `trellis/pipelines/samplers/`) — those are ours; the rest of TRELLIS is not.
- `Wan2.2/` — vendored Wan2.2 I2V package (`import wan` after `sys.path.insert`). Stage A is the only consumer.
- `gim/` — GIM-DKM image matching (FreeArt3D pipeline only).

`run_v1.py` and `pipelines/stage_a_wan.py` insert their dependency dirs into `sys.path` before any submodule import — preserve that ordering.

## Tests

Tests under `tests/` are **standalone scripts**, not a pytest suite. Run individually:

```bash
python tests/test_stage_a_smoke.py
python tests/test_scar_sampler.py
python tests/test_stage_b_e2e_30857.py            # heavy, needs CUDA + TRELLIS
python tests/test_stage_c_segmatch_v6.py
```

`test_stage_a_smoke.py` does NOT load Wan2.2 weights — it only checks the prompt builder, optical-flow check, and viz helpers. Anything named `*_e2e_*` requires GPU + full checkpoints.

## Input conventions

- `cfg.train_num_state` (default 6) segmented images per joint. Example inputs use `00_seg.png … 05_seg.png` + a `05_pure.png` (unsegmented final state); PartNet uses `rendering_joint_{j:02d}_state_{i:02d}.png` + `rendering_pure_joint_{j:02d}_state_05.png`. v1 default pattern is `rendering_joint_00_state_{i:02d}.png` (configurable via `cfg.io.image_pattern`).
- A "carpet"/grounding disk under the object is required (FreeArt3D plane-fit prior). `add_disk.py` is the GUI helper; for v1 the carpet is expected to be in the user-supplied `s_0_with_carpet` (Stage A passes the carpet through Wan I2V end-to-end; Stage G export strips it via `is_carpet_mask`).

## Code conventions (from `record/readme.md` — these override defaults)

- All code lives under `mine/`. **No UTF-8 BOM, no Chinese characters** in source files.
- **No `try/except` around imports**. Imports must succeed; if a vendored dep is missing, raise loudly (see `stage_a_wan.py:50`).
- Avoid scattered `try/except` blocks generally.
- **No "compatibility" / patch / fallback code**, no graceful degradation, no "we'll fix this later" shims. Take the shortest-path correct implementation.
- Every major experimental stage **must produce visualizations** (HTML/PNG under the stage's `viz/` dir) for debugging.
- When porting features from a reference repo, preserve the original functionality as much as needed — don't silently simplify.
- All shell scripts use LF line endings (not CRLF).

When proposing scheme changes, do not silently rewrite `record/method.md` / `record/pipeline.md` — confirm direction with the user first. The drafts there are working documents and may be wrong; verify against the actual code and current SOTA before acting on them.

## Stage / tuning notes

- TRELLIS `tiny-cuda-nn` is non-deterministic; results vary slightly even with seeded runs.
- v1 Stage B Pass-2 BMCSA expects `token_resolution=16` to match `flow_model.resolution / flow_model.patch_size` for TRELLIS-image-large.
- v1 Stage D placeholder always pulls `O_0 = stage_b_res.O_stack[0]` (the `use_vgcf_O_0=false` branch is unimplemented and warns).
- `outputs/{name}/config.yaml` is the dumped OmegaConf — inspect it to recover the exact run config.
