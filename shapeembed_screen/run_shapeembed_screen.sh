#!/bin/bash
# ShapeEmbedLite screen — Anna's recommended config (from the call).
#   SLURM:  sbatch run_shapeembed_screen.sh          (one config per array task)
#   Local:  bash   run_shapeembed_screen.sh          (configs sequentially)
# Per config:
#   (1) train on TRAIN, evaluate on VAL  -> VAL latents (408) + model weights + reconstruction reports
#   (2) --skip-training on the TRAIN folder as "test" -> FULL 577 train latents (unshuffled -> fish_id)
#   (3) featurize both -> features_{train,val}.csv (fish_id, f0..fN, label) for standard_eval hold-out
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

BASE="/path/to/ShapeEmbed/"                 # -> .../Ziram
PYTHON="${PYTHON:-$BASE/venv_cnn/bin/python}"
REPO="$BASE/ShapeEmbedLite"
DM_DIR="$BASE/BinaryMask2DM_Ziram_TrainVal"
TRAIN_DM="$DM_DIR/train"                                 # 577 F0
VAL_DM="$DM_DIR/test"                                    # 408 F0+F2  (ShapeEmbed "test" = our validation)
OUT_BASE="$BASE/shapeembed_screen_out/version3"
FEATURIZE="$(cd "$(dirname "$0")" && pwd)/shapeembed_to_features.py"

# ---- Anna's config (call): fro, epochs 200, lr 1e-4, beta 1e-8, latent {64,128} ----
# NB: the repo doubles the latent (ls_sz = -l * 2), so -l 64 -> 128-d, -l 128 -> 256-d.
EPOCHS=200; LR=0.0001; BETA=1e-8; NORM=fro
CONFIGS=( "64" "128" )                                   # -l values

run_one() {
  local L="$1"
  local TAG="ls${L}_b${BETA}_lr${LR}_${NORM}_e${EPOCHS}"
  local OUT="$OUT_BASE/$TAG"
  local CFG="--preprocess-normalize $NORM -l $L -b $BETA -r $LR --classify-with-scale"
  echo "=== $TAG ==="

  # (1) train + evaluate on VAL  (keeps the VAL reconstruction reports -> the artifact to show Anna)
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --train-test-dataset Ziram "$TRAIN_DM" "$VAL_DM" \
      $CFG -e "$EPOCHS" -n 5 -p 0.1 10 -o "$OUT"
  local MODEL; MODEL="$(ls "$OUT"/*_model_state_dict.pth | head -1)"

  # (2) FULL 577 train latents: skip-training, TRAIN folder as "test" (unshuffled -> fish_id, no 80% cut)
  #     Faster alternative (461 subset, no IDs): drop this block and add --extract-train-latent to (1).
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --skip-training -w "$MODEL" \
      --train-test-dataset Ziram "$TRAIN_DM" "$TRAIN_DM" $CFG -n 5 -o "$OUT/train_full"

  # (3) featurize (both come from the unshuffled test_loader -> DatasetFolder order -> fish_id)
  "$PYTHON" "$FEATURIZE" --latent "$OUT/train_full/test_latent_space.npy" \
      --label "$OUT/train_full/test_labels.npy" --dm-dir "$TRAIN_DM" --match-order \
      --out "$OUT/features_train.csv"
  "$PYTHON" "$FEATURIZE" --latent "$OUT/test_latent_space.npy" \
      --label "$OUT/test_labels.npy" --dm-dir "$VAL_DM" --match-order \
      --out "$OUT/features_val.csv"
  echo "[OK] $TAG -> $OUT/features_{train,val}.csv  (train n=577, val n=408)"
}

mkdir -p "$OUT_BASE"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run_one "${CONFIGS[$((SLURM_ARRAY_TASK_ID-1))]}"
else
  for c in "${CONFIGS[@]}"; do run_one "$c"; done
fi
