#!/usr/bin/env bash
# Run the v1 pipeline up to Stage C only (Stage B VGCF + Stage C SAJO).
#
# Usage:
#   bash run_stage_bc.sh <input_dir> <output_dir> [joint_type]
#
# Examples:
#   bash run_stage_bc.sh outputs/30857 outputs/30857_v1
#   bash run_stage_bc.sh outputs/30857 outputs/30857_v1 revolute
#
# Produces, under <output_dir>/:
#   config.yaml
#   inputs/
#   stage_b_vgcf/        # O_stack.npy, O_stack_soft.npy, z_final.pt,
#                        # vgcf_diagnostics.json, viz/*.html
#   stage_c_sajo/        # p_base, p_move, anchors, em_*.json, bic.json,
#                        # M_base, M_move, T_k, joint_info.json, viz/*.html
#
# Stages D (Stage-2 + mesh decode) and F (URDF + pybullet) are NOT executed.

set -euo pipefail

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
JOINT_TYPE="${3:-}"
CONFIG="${CONFIG:-configs/v1.yaml}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: bash run_stage_bc.sh <input_dir> <output_dir> [joint_type]" >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: input_dir not found: $INPUT_DIR" >&2
  exit 3
fi

CMD=(python run_v1.py
  --input_dir  "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --config     "$CONFIG"
  --device     "$DEVICE"
  --stage      c
)

if [[ -n "$JOINT_TYPE" ]]; then
  CMD+=(--joint_type "$JOINT_TYPE")
fi

echo "[run_stage_bc] $(printf '%q ' "${CMD[@]}")"
"${CMD[@]}"

echo ""
echo "[run_stage_bc] done. Inspect:"
echo "  $OUTPUT_DIR/stage_b_vgcf/viz/O_stack.html"
echo "  $OUTPUT_DIR/stage_b_vgcf/viz/vgcf_diagnostics.html"
echo "  $OUTPUT_DIR/stage_c_sajo/viz/p_base.html"
echo "  $OUTPUT_DIR/stage_c_sajo/viz/p_move.html"
echo "  $OUTPUT_DIR/stage_c_sajo/viz/M_base_vs_move.html"
echo "  $OUTPUT_DIR/stage_c_sajo/viz/anchors.html"
echo "  $OUTPUT_DIR/stage_c_sajo/viz/axis_overlay.html"
echo "  $OUTPUT_DIR/stage_c_sajo/viz/em_traces.html"
