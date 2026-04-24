# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FreeArt3D (SIGGRAPH Asia 2025): training-free articulated 3D object generation by optimizing articulation parameters against a pretrained 3D diffusion prior (TRELLIS). Given multi-view segmented images of an object at several joint states, it reconstructs a base mesh, estimates the joint, and refines everything via SDS.

## Environment

```bash
conda activate mine

```

`setup.sh` installs torch/kaolin and builds local dependencies. If using a different CUDA/torch, install a matching [kaolin](https://github.com/NVIDIAGameWorks/kaolin). GIM-DKM checkpoint must be placed at `gim/weights/` (see README).

## Architecture

The entrypoints `run_two_parts.py` / `run_multi_parts.py` are thin drivers that load `configs/default.yaml` (OmegaConf) and sequence three pipeline stages from `pipelines/`:

1. **`pipelines/recon.run_recon`** — Uses TRELLIS (vendored under `TRELLIS/`) to reconstruct a coarse base mesh from the input segmented images (`{i:02d}_seg.png` for examples, or `rendering_joint_{j:02d}_state_{i:02d}.png` for PartNet). Output goes to `outputs/{name}/recon`.
2. **`pipelines/estimate.run_estimate`** — Estimates initial joint parameters (type = `prismatic` or `revolute`, axis, pivot) using GIM-DKM feature matching (`gim/`) and RANSAC (thresholds in `default.yaml`). Joint type is a CLI flag, or auto-selected for `--partnet` runs via `configs/partnet.json`.
3. **`pipelines/sds.run_sds`** — The core optimization. Jointly refines the base mesh, part mesh, and joint parameters (axis, pivot, scale, qpos) by score distillation sampling against the TRELLIS 3D diffusion prior, plus voxel and intersection losses. Config knobs: `sds_weight`, `voxel_weight`, `intersect_weight`, `cfg_strength`, `noise_start/end`, `*_lr`, `total_iters`. Writes `sds_output/part_meshes/{fixed,articulated_NN}.glb`, `sds_output/states/`, and `sds_output/joint_info.json`.
4. **`pipelines/urdf.write_urdf`** — Packs the fixed base mesh + articulated parts + joint info into `outputs/{name}/output.urdf`.

Key supporting code: `pipelines/utils/` (shared helpers), `eval_utils/` (metrics used by `evaluate_partnet.py`), `artpipe/` (articulation optimization internals), `gim/` (image matching, needs weights), `TRELLIS/trellis/` (vendored TRELLIS, including FlexiCubes mesh representation — do not treat as our code).

## Input conventions

- `cfg.train_num_state` (default 6) segmented images are expected per joint. Example inputs use filenames `00_seg.png … 05_seg.png` plus a `05_pure.png` (unsegmented final state). PartNet inputs use `rendering_joint_{j:02d}_state_{i:02d}.png` and `rendering_pure_joint_{j:02d}_state_05.png`.
- Inputs must include a "carpet"/disk under the object (used as a stable grounding prior). `add_disk.py` is provided to add one to arbitrary SAM-segmented user images.
- Custom `--input_dir` that doesn't have `05_seg.png` is treated as PartNet-style naming — keep this branch in mind when adding new data formats (`run_two_parts.py:31`).

## Stage control & tuning notes (from code comments)

- `--stage {all,recon,estimate,sds}` lets you re-run a single stage against existing intermediates in `outputs/{name}/`.
- For PartNet runs, `cfg.ransac_threshold` may need tuning (try 0.03 / 0.1) when estimation fails; there is a commented example in `run_two_parts.py`.
- For the `WashingMachine` category the reconstruction view range is overridden (`cfg.min_azi`, `cfg.max_azi`) to avoid blind spots — a useful hint when adding new categories that reconstruct poorly.
- Some CUDA kernels (notably tiny-cuda-nn used via TRELLIS) are non-deterministic, so results may differ slightly across runs even with seeds fixed.
