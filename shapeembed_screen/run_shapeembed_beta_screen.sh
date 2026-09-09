#!/bin/bash
# ShapeEmbedLite exploratory screen
# Grid: norm {none, fro} x beta {1e-8, 1e-4, 1e-2, 1}, latent 128, lr 1e-4, e200, ReLU.
# TRAIN-ONLY exploration: no feature extraction.
# Work on features produced by the chosen config, by run_shapeembed_screen.sh.
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=8000M
#SBATCH --array=1-24
#SBATCH -J "se_beta_screen"
#SBATCH -o "se_beta_screen_%A_%a.out"
#SBATCH -e "se_beta_screen_%A_%a.err"
set -euo pipefail

BASE="/path/to/base"
PYTHON="${PYTHON:-$BASE/venv/bin/python}"
REPO="$BASE/ShapeEmbedLite"
TRAIN_DM="$BASE/DM_folder/train"
VAL_DM="$BASE/DM_folder/test"
OUT_BASE="$BASE/output"
FEATURIZE="$(cd "$(dirname "$0")" && pwd)/shapeembed_to_features.py"
NAME="experiment_name"
LR=0.0001; EPOCHS=200

# 3-axis grid: latent x norm x beta  (24 configs -> parallel SLURM array).
# NB the repo doubles the latent (-l 64 -> 128-d, -l 128 -> 256-d, -l 256 -> 512-d).

LATENTS=(64 128 256); NORMS=(none fro); BETAS=(1e-8 1e-7 1e-6 1e-5)
CONFIGS=()
for _L in "${LATENTS[@]}"; do for _N in "${NORMS[@]}"; do for _B in "${BETAS[@]}"; do
  CONFIGS+=("$_L $_N $_B")
done; done; done

run_one() {
  local L NORM BETA; read -r L NORM BETA <<< "$1"
  local TAG="ls${L}_b${BETA}_lr${LR}_${NORM}_e${EPOCHS}"
  local OUT="$OUT_BASE/$TAG"
  local NORMFLAG=""; [ "$NORM" != "none" ] && NORMFLAG="--preprocess-normalize $NORM"
  echo "=== $TAG ==="
  # TRAIN + built-in eval only.
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --train-test-dataset "$NAME" "$TRAIN_DM" "$VAL_DM" \
      $NORMFLAG -l "$L" -b "$BETA" -r "$LR" -e "$EPOCHS" -n 5 --classify-with-scale \
      -p 0.1 10 -o "$OUT"
  echo "[OK] $TAG -> $OUT  (loss_record.csv, run_report.txt, reconstruction reports)"
}

mkdir -p "$OUT_BASE"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run_one "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"
else
  for c in "${CONFIGS[@]}"; do run_one "$c"; done
fi
