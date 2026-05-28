#!/usr/bin/env bash
#SBATCH --job-name=stagea
#SBATCH --partition=H800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=stagea_%j.out
#SBATCH --error=stagea_%j.err
#
# Stage A: Wan2.2 I2V 21-frame articulated-object video.
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

WAN_CKPT=~/hf_models/Wan2.2-I2V-A14B
IMAGE_PATH=~/hf_models/PartNet/${OBJECT_ID}/00_seg.png
STAGEA_SEED="${STAGEA_SEED:-42}"
STAGEA_GUIDE_SCALE="${STAGEA_GUIDE_SCALE:-3.5}"
STAGEA_SAMPLE_SHIFT="${STAGEA_SAMPLE_SHIFT:-5.0}"

mkdir -p "${OUTPUT_DIR}"

python scripts/stagea.py \
    --image      "${IMAGE_PATH}" \
    --motion     "${PROMPT}" \
    --wan_ckpt   "${WAN_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed       "${STAGEA_SEED}" \
    --guide_scale "${STAGEA_GUIDE_SCALE}" \
    --sample_shift "${STAGEA_SAMPLE_SHIFT}" \
    --no_offload_model
