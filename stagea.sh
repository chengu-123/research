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

# Examples:
#   bash stagea.sh 30857 "the drawer slowly slides outward in a continuous motion"
#   bash stagea.sh 7201  "the oven door slowly tilts downward to open"
#   bash stagea.sh 7128  "the microwave door slowly swings open on its single hinge"

set -euo pipefail

source ~/env/mine/bin/activate

export TORCH_HOME=~/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# -------------------------- arguments ---------------------------------------
OBJECT_ID="${1:-}"
MOTION_PROMPT="${2:-}"

if [[ -z "$OBJECT_ID" || -z "$MOTION_PROMPT" ]]; then
  echo "Usage: $0 <object_id> \"<motion_prompt>\"" >&2
  echo "Example:" >&2
  echo "  $0 30857 \"the drawer slowly slides outward in a continuous motion\"" >&2
  exit 2
fi

# -------------------------- paths (env-overridable) -------------------------
WAN_CKPT="${WAN_CKPT:-${HOME}/hf_models/Wan2.2-I2V-A14B}"
INPUT_DIR="${INPUT_DIR:-${HOME}/hf_models/PartNet/${OBJECT_ID}}"
IMAGE_BASENAME="${IMAGE_BASENAME:-00_seg.png}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

# -------------------------- pipeline v3.3.1 fixed defaults ------------------
PROMPT_LANG="${PROMPT_LANG:-en}"
SEED="${SEED:-42}"
FRAME_NUM="${FRAME_NUM:-21}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
FPS="${FPS:-16}"
DEVICE_ID="${DEVICE_ID:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# -------------------------- derived paths -----------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_PATH="${INPUT_DIR}/${IMAGE_BASENAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/${OBJECT_ID}/stage_a"

# -------------------------- existence checks (fail fast) --------------------
if [[ ! -d "$WAN_CKPT" ]]; then
  echo "ERROR: Wan2.2 checkpoint dir not found: $WAN_CKPT" >&2
  echo "  Override via env var: WAN_CKPT=/your/path $0 ..." >&2
  exit 3
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: per-object input dir not found: $INPUT_DIR" >&2
  echo "  Override via env var: INPUT_DIR=/your/path $0 ..." >&2
  exit 3
fi
if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "ERROR: input image not found: $IMAGE_PATH" >&2
  echo "  Expected ${IMAGE_BASENAME} under ${INPUT_DIR}." >&2
  echo "  Override filename via: IMAGE_BASENAME=rendering_joint_00_state_00.png $0 ..." >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR"

# -------------------------- header echo -------------------------------------
echo "========== Stage A : Wan2.2 I2V -- object ${OBJECT_ID} =========="
echo "  WAN_CKPT       : ${WAN_CKPT}"
echo "  INPUT_DIR      : ${INPUT_DIR}"
echo "  IMAGE_PATH     : ${IMAGE_PATH}"
echo "  OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "  MOTION_PROMPT  : ${MOTION_PROMPT}"
echo "  PROMPT_LANG    : ${PROMPT_LANG}"
echo "  SEED           : ${SEED}"
echo "  FRAME_NUM      : ${FRAME_NUM}"
echo "  RESOLUTION HxW : ${HEIGHT} x ${WIDTH}"
echo "  SAMPLING_STEPS : ${SAMPLING_STEPS}"
echo "  GUIDE_SCALE    : ${GUIDE_SCALE}"
echo "  SAMPLE_SHIFT   : ${SAMPLE_SHIFT}"
echo "  SAMPLE_SOLVER  : ${SAMPLE_SOLVER}"
echo "  FPS            : ${FPS}"
echo "  DEVICE_ID      : ${DEVICE_ID}"
echo "  EXTRA_ARGS     : ${EXTRA_ARGS}"
echo "=================================================================="

cd "${SCRIPT_DIR}"

# -------------------------- run Stage A -------------------------------------
# scripts/stagea.py validates 4n+1 / 8-multiple constraints before loading Wan,
# so bad SEED / FRAME_NUM / HEIGHT / WIDTH env overrides fail fast.
# EXTRA_ARGS is unquoted on purpose so multi-flag strings split into tokens.
python scripts/stagea.py \
    --image          "${IMAGE_PATH}" \
    --motion         "${MOTION_PROMPT}" \
    --wan_ckpt       "${WAN_CKPT}" \
    --output_dir     "${OUTPUT_DIR}" \
    --lang           "${PROMPT_LANG}" \
    --seed           "${SEED}" \
    --frame_num      "${FRAME_NUM}" \
    --height         "${HEIGHT}" \
    --width          "${WIDTH}" \
    --sampling_steps "${SAMPLING_STEPS}" \
    --guide_scale    "${GUIDE_SCALE}" \
    --sample_shift   "${SAMPLE_SHIFT}" \
    --sample_solver  "${SAMPLE_SOLVER}" \
    --fps            "${FPS}" \
    --device_id      "${DEVICE_ID}" \
    ${EXTRA_ARGS}

# -------------------------- done --------------------------------------------
echo ""
echo "[stagea] done. Inspect:"
echo "  ${OUTPUT_DIR}/wan_video_target.mp4"
echo "  ${OUTPUT_DIR}/wan_video_grid.png"
echo "  ${OUTPUT_DIR}/keyframes_6.png"
echo "  ${OUTPUT_DIR}/optical_flow_per_frame.png"
echo "  ${OUTPUT_DIR}/meta.json"
echo ""
echo "Stage B input tensor: ${OUTPUT_DIR}/wan_video_target_3FHW_uint8.pt"
