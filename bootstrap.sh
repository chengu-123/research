#!/usr/bin/env bash
#SBATCH --job-name=bootstrap
#SBATCH --partition=H800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=bootstrap_%j.out
#SBATCH --error=bootstrap_%j.err
#
# Build the Stage D bootstrap bundle from Stage A video or six state images.
#
# Usage:
#   cd /path/to/mine
#   sbatch bootstrap.sh outputs/30857 "A brown wooden desk. The right drawer slides open, smoothly and completely pulling outward. "
#   BOOTSTRAP_INPUT_MODE=six_images IMAGE_DIR=~/hf_models/PartNet/30857 sbatch bootstrap.sh outputs/30857 "..."
#   PURE_IMAGE=~/hf_models/PartNet/30857/00_pure.png sbatch bootstrap.sh outputs/30857 "..."
#
# Stage A video input:
#   outputs/30857/stage_a/wan_video_target_3FHW_uint8.pt
#
# Six-image input:
#   IMAGE_DIR/rendering_joint_00_state_{00..05}.png by default
#
# Output:
#   outputs/30857/bootstrap/

set -euo pipefail

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

OUTPUT_ROOT="$1"
PROMPT="$2"

WAN_CKPT="${WAN_CKPT:-${HOME}/hf_models/Wan2.2-I2V-A14B}"
DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-0}"
BOOTSTRAP_INPUT_MODE="${BOOTSTRAP_INPUT_MODE:-stagea_video}"
STAGEA_VIDEO="${STAGEA_VIDEO:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
IMAGE_PATTERN="${IMAGE_PATTERN:-rendering_joint_00_state_{i:02d}.png}"
PURE_IMAGE="${PURE_IMAGE:-}"

INPUT_ARGS=(--input_mode "${BOOTSTRAP_INPUT_MODE}")
if [[ "${BOOTSTRAP_INPUT_MODE}" == "stagea_video" ]]; then
  if [[ -n "${STAGEA_VIDEO}" ]]; then
    INPUT_ARGS+=(--stagea_video "${STAGEA_VIDEO}")
  fi
elif [[ "${BOOTSTRAP_INPUT_MODE}" == "six_images" ]]; then
  if [[ -z "${IMAGE_DIR}" ]]; then
    echo "ERROR: IMAGE_DIR is required when BOOTSTRAP_INPUT_MODE=six_images" >&2
    exit 2
  fi
  INPUT_ARGS+=(--image_dir "${IMAGE_DIR}" --image_pattern "${IMAGE_PATTERN}")
else
  echo "ERROR: BOOTSTRAP_INPUT_MODE must be stagea_video or six_images" >&2
  exit 2
fi

if [[ -z "${PURE_IMAGE}" ]]; then
  if [[ -n "${IMAGE_DIR}" ]]; then
    PURE_IMAGE="${IMAGE_DIR}/00_pure.png"
  else
    RUN_ID="$(basename "${OUTPUT_ROOT}")"
    PURE_IMAGE="${HOME}/hf_models/PartNet/${RUN_ID}/00_pure.png"
  fi
fi
if [[ ! -f "${PURE_IMAGE}" ]]; then
  echo "ERROR: PURE_IMAGE not found: ${PURE_IMAGE}" >&2
  echo "Set PURE_IMAGE to the no-carpet 00_pure.png that matches the Stage A 00_seg.png." >&2
  exit 2
fi

python scripts/bootstrap.py \
    --output_dir "${OUTPUT_ROOT}" \
    "${INPUT_ARGS[@]}" \
    --s0_pure "${PURE_IMAGE}" \
    --motion "${PROMPT}" \
    --wan_ckpt "${WAN_CKPT}" \
    --device "${DEVICE}" \
    --device_id "${DEVICE_ID}"
