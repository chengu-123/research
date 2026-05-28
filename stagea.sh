#!/usr/bin/env bash
#SBATCH --job-name=stagea
#SBATCH --partition=H800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --time=4:00:00
#SBATCH --output=stagea_%j.out
#SBATCH --error=stagea_%j.err
#
# Stage A: Wan2.2 21-frame articulated-object video.
# All hyperparameters are fixed by pipeline v3.3.1 inside scripts/stagea.py
# (seed=42, frame=21, Wan 832*480 area profile with input aspect preserved,
# guide=3.5, unipc, 50 steps, lang=en, no model offload on H800, ...).
# This wrapper only takes the three things that actually vary per run:
#   $1  OBJECT_ID    PartNet object id under ~/hf_models/PartNet/
#   $2  OUTPUT_DIR   destination dir for the Stage A artifacts
#   $3  PROMPT       user motion description (quoted)
#
# Usage:
#   cd /path/to/mine        # sbatch must be invoked from repo root
#   sbatch stagea.sh 30857 outputs/30857/stage_a "A brown wooden desk. The right drawer slides open, smoothly and completely pulling outward. "

set -euo pipefail

# Local -u disable: mine conda env ships intel-mkl whose activate hook
# references unset $MKL_INTERFACE_LAYER.
set +u
source ~/env/mine/bin/activate
set -u

export TORCH_HOME=~/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

OBJECT_ID="$1"
OUTPUT_DIR="$2"
PROMPT="$3"

STAGEA_BACKEND="${STAGEA_BACKEND:-i2v}"
if [[ "${STAGEA_BACKEND}" == "fun_inp" ]]; then
  WAN_CKPT="${WAN_CKPT:-${HOME}/hf_models/Wan2.2-Fun-A14B-InP}"
else
  WAN_CKPT="${WAN_CKPT:-${HOME}/hf_models/Wan2.2-I2V-A14B}"
fi
IMAGE_PATH="${IMAGE_PATH:-${HOME}/hf_models/PartNet/${OBJECT_ID}/00_seg.png}"
IMAGE_END="${IMAGE_END:-${HOME}/hf_models/PartNet/${OBJECT_ID}/05_seg.png}"
STAGEA_SEED="${STAGEA_SEED:-42}"
if [[ "${STAGEA_BACKEND}" == "fun_inp" ]]; then
  STAGEA_GUIDE_SCALE="${STAGEA_GUIDE_SCALE:-6.0}"
else
  STAGEA_GUIDE_SCALE="${STAGEA_GUIDE_SCALE:-3.5}"
fi
STAGEA_SAMPLE_SHIFT="${STAGEA_SAMPLE_SHIFT:-5.0}"
VIDEOX_FUN_ROOT="${VIDEOX_FUN_ROOT:-${HOME}/VideoX-Fun}"
FUN_CONFIG="${FUN_CONFIG:-${VIDEOX_FUN_ROOT}/config/wan2.2/wan_civitai_i2v.yaml}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ "${STAGEA_BACKEND}" == "fun_inp" ]]; then
  if [[ ! -f "${IMAGE_END}" ]]; then
    echo "ERROR: IMAGE_END not found for STAGEA_BACKEND=fun_inp: ${IMAGE_END}" >&2
    exit 2
  fi
  if [[ ! -f "${FUN_CONFIG}" ]]; then
    echo "ERROR: FUN_CONFIG not found: ${FUN_CONFIG}" >&2
    echo "Set VIDEOX_FUN_ROOT or FUN_CONFIG to the local VideoX-Fun config." >&2
    exit 2
  fi
  EXTRA_ARGS+=(--image_end "${IMAGE_END}" --fun_config "${FUN_CONFIG}")
fi

python scripts/stagea.py \
    --backend    "${STAGEA_BACKEND}" \
    --image      "${IMAGE_PATH}" \
    "${EXTRA_ARGS[@]}" \
    --motion     "${PROMPT}" \
    --wan_ckpt   "${WAN_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed       "${STAGEA_SEED}" \
    --guide_scale "${STAGEA_GUIDE_SCALE}" \
    --sample_shift "${STAGEA_SAMPLE_SHIFT}"
