#!/bin/bash
# Determinism check

# Usage:  bash check_determinism.sh <path/to/model_state_dict.pth> [LATENT_-l]
#   LATENT_-l must match the -l used to TRAIN that checkpoint (default 128).
set -euo pipefail

MODEL="${1:?usage: check_determinism.sh <model_state_dict.pth> [latent_-l, default 128]}"
L="${2:-128}"
BASE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$BASE/venv/bin/python}"
REPO="$BASE/ShapeEmbedLite"
VAL_DM="$BASE/DM_folder/test"
OUT="$BASE/output/determinism_check"
CFG="--preprocess-normalize fro -l $L -b 1e-8 --classify-with-scale"
NAME="experiment_name"

for r in a b; do
  echo "=== extraction pass $r ==="
  "$PYTHON" "$REPO/ShapeEmbedLite.py" --skip-training -w "$MODEL" \
      --train-test-dataset "$NAME" "$VAL_DM" "$VAL_DM" $CFG -n 5 -o "$OUT/run_$r"
done

"$PYTHON" - "$OUT/run_a/test_latent_space.npy" "$OUT/run_b/test_latent_space.npy" <<'PY'
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
d = np.abs(a - b)
print(f"\nlatents shape: {a.shape}")
print(f"identical      : {np.array_equal(a, b)}")
print(f"max |a-b|      : {d.max():.3e}")
print(f"mean |a-b|     : {d.mean():.3e}")
print(f"% elems differ : {100*(d > 1e-8).mean():.1f}%")
print("\n=> REPRODUCIBLE (deterministic across passes)" if np.allclose(a, b)
      else "\n=> NOT reproducible: the exported latent is a random sample (differs between passes)")
PY
