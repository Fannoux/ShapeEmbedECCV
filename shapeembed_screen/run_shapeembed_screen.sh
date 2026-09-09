#!/bin/bash
# ShapeEmbedLite screen.
#   SLURM:  sbatch run_shapeembed_screen.sh          (one config per array task)
#   Local:  bash   run_shapeembed_screen.sh          (configs sequentially)
# Per config:
#   (1) train on TRAIN, evaluate on VAL 
#   (2) --skip-training on the TRAIN folder as "test"
#   (3) featurize both -> features_{train,val}.csv for standard_eval hold-out
#
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem=8000M
#SBATCH --array=1-2
#SBATCH -J "shapeembed_screen"
#SBATCH -o "runs/logs/shapeembed_screen_%A_%a.out"
#SBATCH -e "runs/logs/shapeembed_screen_%A_%a.err"
set -euo pipefail

BASE="/path/to/ShapeEmbed/"                 
PYTHON="${PYTHON:-$BASE/venv/bin/python}"
REPO="$BASE/ShapeEmbedLite"
DM_DIR="$BASE/DM_folder"
TRAIN_DM="$DM_DIR/train"                                 
VAL_DM="$DM_DIR/test"                                    # ShapeEmbed "test" = our validation
OUT_BASE="$BASE/output/"
FEATURIZE="$(cd "$(dirname "$0")" && pwd)/shapeembed_to_features.py"
NAME="experiment_name"

# NB: the repo doubles the latent (ls_sz = -l * 2), so -l 64 -> 128-d, -l 128 -> 256-d.
EPOCHS=200; LR=0.0001; BETA=1e-8; NORM=fro
CONFIGS=( "64" "128" )                                   # -l values

run_one() {
  local L="$1"
  local TAG="ls${L}_b${BETA}_lr${LR}_${NORM}_e${EPOCHS}"
  local OUT="$OUT_BASE/$TAG"
  local CFG="--preprocess-normalize $NORM -l $L -b $BETA -r $LR --classify-with-scale"
  echo "=== $TAG ==="

  # (1) train + evaluate on VAL  (keeps the VAL reconstruction reports)
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --train-test-dataset "$NAME" "$TRAIN_DM" "$VAL_DM" \
      $CFG -e "$EPOCHS" -n 5 -p 0.1 10 -o "$OUT"
  local MODEL; MODEL="$(ls "$OUT"/*_model_state_dict.pth | head -1)"

  # (2) FULL train latents: skip-training, TRAIN folder as "test" (unshuffled no 80% cut)
  #     Faster alternative: drop this block and add --extract-train-latent to (1).
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --skip-training -w "$MODEL" \
      --train-test-dataset "$NAME" "$TRAIN_DM" "$TRAIN_DM" $CFG -n 5 -o "$OUT/train_full"

  # (3) featurize (both come from the unshuffled test_loader -> DatasetFolder order)
  "$PYTHON" "$FEATURIZE" --latent "$OUT/train_full/test_latent_space.npy" \
      --label "$OUT/train_full/test_labels.npy" --dm-dir "$TRAIN_DM" --match-order \
      --out "$OUT/features_train.csv"
  "$PYTHON" "$FEATURIZE" --latent "$OUT/test_latent_space.npy" \
      --label "$OUT/test_labels.npy" --dm-dir "$VAL_DM" --match-order \
      --out "$OUT/features_val.csv"
  echo "[OK] $TAG -> $OUT/features_{train,val}.csv"
}

mkdir -p "$OUT_BASE"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run_one "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"
else
  for c in "${CONFIGS[@]}"; do run_one "$c"; done
fi
