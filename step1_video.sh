#!/usr/bin/env bash
#SBATCH --job-name=step1_video
#SBATCH --partition=H800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=step1_video_%j.out
#SBATCH --error=step1_video_%j.err
#
# Step 1/5: WAN2.2 Image-to-Video generation
#

source ~/env/mine/bin/activate

export TORCH_HOME=~/.cache/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

OBJECT_ID="${1:-30857}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WAN_CKPT=~/hf_models/Wan2.2-I2V-A14B
INPUT_DIR=~/hf_models/PartNet/${OBJECT_ID}

mkdir -p "${OUTPUT_BASE}"

echo "========== Step 1/5: WAN2.2 I2V — ${OBJECT_ID} =========="
echo "  Image  : ${IMAGE_PATH}"
echo "  Output : ${VIDEO_PATH}"

cd "${PIPELINE_DIR}"
python generate_video.py \
    --ckpt_dir "${WAN_CKPT}" \
    --image "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --save_file "${VIDEO_PATH}" \
    --size 832*480 \
    --frame_num 81 \
    --base_seed 42 \
    --offload_model True

echo "Step 1 complete. Video saved to ${VIDEO_PATH}"
