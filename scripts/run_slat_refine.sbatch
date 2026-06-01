#!/bin/bash
#SBATCH --job-name=slatref
#SBATCH --partition=H800
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

set +u
source /lustre/1230003454/env/fa3d2/bin/activate
set -u

cd /lustre/1230003454/current/mine

BDIR=outputs/7201/bootstrap_optionA_s2/bootstrap
AGG=outputs/7201/texture_init_agg/aggregated_init.pt

# Args forwarded from sbatch command line ("$@").
python scripts/wan_texture_slat_refine.py \
  --bootstrap_dir "${BDIR}" \
  --agg_init "${AGG}" \
  "$@"
