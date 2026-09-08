#!/bin/bash
# ShapeEmbedLite exploratory screen — now that the decoder has a ReLU, test whether:
#   (a) it converges WITHOUT input normalisation, and
#   (b) a non-trivial beta (KL / variational term) helps the latent.
# Grid: norm {none, fro} x beta {1e-8, 1e-4, 1e-2, 1}, latent 128, lr 1e-4, e200, ReLU (current HEAD).
# TRAIN-ONLY exploration: no feature extraction. Pick the config from convergence (loss_record.csv),
# reconstruction, and the built-in classification (run_report.txt) — all produced by the training run.
# The full-577, deterministic, fish_id-mapped features are produced ONCE, on the chosen config, by
# run_shapeembed_screen.sh (after the z_mean fix). So nothing extracted here is wasted/re-done.
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

BASE="/nfs/research/birney/users/fanny/medaka/ziram_analysis/"
PYTHON="${PYTHON:-$BASE/scripts_cnn/venv/bin/python}"
REPO="$BASE/ShapeEmbed/ShapeEmbedLite"
TRAIN_DM="/nfs/research/birney/users/fanny/medaka/ziram_analysis/ECCV_frozen/DM_TrainVal/train"
VAL_DM="/nfs/research/birney/users/fanny/medaka/ziram_analysis/ECCV_frozen/DM_TrainVal/test"
OUT_BASE="${OUT_BASE:-$BASE/ShapeEmbed/shapeembed_beta_screen_out2}"
FEATURIZE="$(cd "$(dirname "$0")" && pwd)/shapeembed_to_features.py"
LR=0.0001; EPOCHS=200

# 3-axis grid: latent x norm x beta  (24 configs -> parallel SLURM array).
# NB the repo doubles the latent (-l 64 -> 128-d, -l 128 -> 256-d, -l 256 -> 512-d).
# Beta CALIBRATED: at -l 128/fro, recon~=0.0024, KL(unweighted)~=2678, so beta*KL == recon at ~9e-7;
# the band 1e-8..1e-5 spans autoencoder -> balanced -> strong-KL. (For -l 64/256 the balance shifts ~x2
# since KL scales with latent dim; the band still covers the useful range.)  Conventional beta >= 0.01
# is NOT used -- it collapses the posterior (KL dominates by 100-10^6x).
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
  # TRAIN + built-in eval only. Pick the config from convergence (loss_record.csv), reconstruction,
  # and the built-in classification in run_report.txt. NO featurization here: those latents are a
  # non-reproducible sampled z on the ~461 train subset and would be thrown away -- the full-577,
  # deterministic, fish_id-mapped features are produced ONCE on the CHOSEN config by
  # run_shapeembed_screen.sh (after the z_mean fix).
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --train-test-dataset Ziram "$TRAIN_DM" "$VAL_DM" \
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
