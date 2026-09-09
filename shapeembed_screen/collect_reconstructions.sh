#!/bin/bash
# Gather the reconstruction artifacts from a run: the summary comparison
# and a spread of per-sample original-vs-reconstructed outline plots.
#
# Usage:  bash collect_reconstructions.sh <run_out_dir>   (e.g. .../shapeembed_screen_out/ls128_...)
set -euo pipefail

OUT="${1:?usage: collect_reconstructions.sh <run_out_dir>}"
DEST="$OUT/output"
mkdir -p "$DEST"

# 1) the multi-sample summary (original vs reconstructed distance matrices/contours)
[ -f "$OUT/summary_report.pdf" ] && cp "$OUT/summary_report.pdf" "$DEST/"
# 2) mean reconstructed shape per severity class (if present)
[ -f "$OUT/mean_shapes.pdf" ] && cp "$OUT/mean_shapes.pdf" "$DEST/"

# 3) a spread of per-sample comparison PDFs (original vs reconstructed contour)
mapfile -t reps < <(ls "$OUT"/report/*_report.pdf 2>/dev/null || true)
n=${#reps[@]}
if [ "$n" -gt 0 ]; then
  step=$(( n/10 > 0 ? n/10 : 1 ))
  for ((i=0; i<n; i+=step)); do cp "${reps[$i]}" "$DEST/"; done
fi

echo "[OK] $(ls "$DEST" 2>/dev/null | wc -l) files -> $DEST"
