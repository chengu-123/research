#!/usr/bin/env bash
#SBATCH --job-name=bootstrap
#SBATCH --partition=H800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --time=8:00:00
#SBATCH --output=bootstrap_%j.out
#SBATCH --error=bootstrap_%j.err
#
# Build the Stage D bootstrap bundle from Stage A video or six state images.
#
# Usage:
#   cd /path/to/mine
#   sbatch bootstrap.sh outputs/30857 "A brown wooden desk. The right drawer slides open, smoothly and completely pulling outward. "
#   IMAGE_DIR=~/hf_models/PartNet/30857 sbatch bootstrap.sh outputs/30857 "..."
#   BOOTSTRAP_INPUT_MODE=stagea_video sbatch bootstrap.sh outputs/30857 "..."
#   PURE_IMAGE=~/hf_models/PartNet/30857/00_pure.png sbatch bootstrap.sh outputs/30857 "..."
#   WAN_BACKEND=fun_inp PURE_END_IMAGE=~/hf_models/PartNet/30857/05_pure.png sbatch bootstrap.sh outputs/30857 "..."
#
# Six-image input:
#   IMAGE_DIR/{00..05}_seg.png by default
#
# Stage A video input:
#   outputs/30857/stage_a/wan_video_target_3FHW_uint8.pt
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
TRELLIS_PRETRAINED="${TRELLIS_PRETRAINED:-${HOME}/hf_models/TRELLIS-image-large}"
DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-0}"
BOOTSTRAP_INPUT_MODE="${BOOTSTRAP_INPUT_MODE:-six_images}"
STAGEA_VIDEO="${STAGEA_VIDEO:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
IMAGE_PATTERN="${IMAGE_PATTERN:-}"
if [[ -z "${IMAGE_PATTERN}" ]]; then
  IMAGE_PATTERN="{i:02d}_seg.png"
fi
PURE_IMAGE="${PURE_IMAGE:-}"
PURE_END_IMAGE="${PURE_END_IMAGE:-}"
WAN_BACKEND="${WAN_BACKEND:-i2v}"

INPUT_ARGS=(--input_mode "${BOOTSTRAP_INPUT_MODE}")
if [[ "${BOOTSTRAP_INPUT_MODE}" == "stagea_video" ]]; then
  if [[ -n "${STAGEA_VIDEO}" ]]; then
    INPUT_ARGS+=(--stagea_video "${STAGEA_VIDEO}")
  fi
elif [[ "${BOOTSTRAP_INPUT_MODE}" == "six_images" ]]; then
  if [[ -z "${IMAGE_DIR}" ]]; then
    RUN_ID="$(basename "${OUTPUT_ROOT}")"
    IMAGE_DIR="${HOME}/hf_models/PartNet/${RUN_ID}"
  fi
  if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "ERROR: IMAGE_DIR not found for six-image bootstrap: ${IMAGE_DIR}" >&2
    echo "Set IMAGE_DIR to the folder containing 00_seg.png ... 05_seg.png." >&2
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
if [[ -z "${PURE_END_IMAGE}" ]]; then
  if [[ -n "${IMAGE_DIR}" ]]; then
    PURE_END_IMAGE="${IMAGE_DIR}/05_pure.png"
  else
    RUN_ID="$(basename "${OUTPUT_ROOT}")"
    PURE_END_IMAGE="${HOME}/hf_models/PartNet/${RUN_ID}/05_pure.png"
  fi
fi
PURE_END_ARGS=()
if [[ -f "${PURE_END_IMAGE}" ]]; then
  PURE_END_ARGS+=(--s5_pure "${PURE_END_IMAGE}")
elif [[ "${WAN_BACKEND}" == "fun_inp" ]]; then
  echo "ERROR: PURE_END_IMAGE not found for WAN_BACKEND=fun_inp: ${PURE_END_IMAGE}" >&2
  exit 2
fi
if [[ ! -f "${TRELLIS_PRETRAINED}/pipeline.json" ]]; then
  echo "ERROR: TRELLIS_PRETRAINED must point to local TRELLIS-image-large with pipeline.json." >&2
  echo "Current TRELLIS_PRETRAINED=${TRELLIS_PRETRAINED}" >&2
  exit 2
fi

python scripts/bootstrap.py \
    --output_dir "${OUTPUT_ROOT}" \
    "${INPUT_ARGS[@]}" \
    --s0_pure "${PURE_IMAGE}" \
    "${PURE_END_ARGS[@]}" \
    --wan_backend "${WAN_BACKEND}" \
    --motion "${PROMPT}" \
    --wan_ckpt "${WAN_CKPT}" \
    --trellis_pretrained "${TRELLIS_PRETRAINED}" \
    --device "${DEVICE}" \
    --device_id "${DEVICE_ID}"
