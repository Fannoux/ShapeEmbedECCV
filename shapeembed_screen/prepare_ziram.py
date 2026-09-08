#!/usr/bin/env python3
"""
Convert Ziram F0 binary masks to ShapeEmbed distance matrices — script version of
prepare_ziram.ipynb, driven by the dataset's predefined split (NO random split here).

Pipeline per mask (exactly as the notebook):
    mask PNG -> find_longest_contour -> contour_spline_resample(n_samples) -> distance_matrix

Uses the manifest's `set` column directly:
    set == 'training'   -> OUT/train/<severity>/<stem>.npy
    set == 'validation' -> OUT/test/<severity>/<stem>.npy   (validation = ShapeEmbed's "test")
Only training + validation are processed (F2 test set not handled here).
Sub-folders are the severity_score_adjusted classes (0-4), so it plugs straight into
    python ShapeEmbedLite.py --train-test-dataset F0 OUT/train OUT/test ...

Usage:
    python prepare_ziram.py \
        --masks /path/to/masks \
        --manifest /path/to/manifest.csv \
        --out /path/to/output \
        --n-samples 64
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find helpers.py next to this script
from helpers import find_longest_contour, contour_spline_resample, distance_matrix

SET2DIR = {'training': 'train', 'validation': 'test'}            # validation -> ShapeEmbed "test"


def image_to_contour(fname, n_samples=64, sparsity=1):
    with Image.open(fname) as img:
        arr = np.array(img)
        cont = find_longest_contour(arr)
    return contour_spline_resample(cont, n_samples=n_samples, sparsity=sparsity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', default=None)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--out', default='./distance_matrices')
    ap.add_argument('--n-samples', type=int, default=64, help='contour points = distance-matrix size (power of 2!)')
    ap.add_argument('--sparsity', type=int, default=1)
    ap.add_argument('--label-col', default='severity_score_adjusted')
    args = ap.parse_args()

    assert (args.n_samples & (args.n_samples - 1)) == 0, \
        f'--n-samples must be a power of 2 (ShapeEmbed requirement), got {args.n_samples}'

    # match masks to manifest (label + set) by filename stem
    import glob
    files = pd.DataFrame({'file': glob.glob(os.path.join(args.masks, '*.png'))})
    files['stem'] = files['file'].map(lambda p: os.path.basename(p)[:-4])
    meta = pd.read_csv(args.manifest)
    if not 'stem' in meta.columns:
        meta['stem'] = meta['image_path'].map(lambda p: os.path.basename(p)[:-4])
    print(meta.head(), files.head())
    merge = meta.merge(files, on='stem', how='inner')
    merge = merge[merge['set'].isin(SET2DIR)]
    print(f"masks: {len(files)} | manifest: {len(meta)} | matched (training+validation): {len(merge)}")
    print(merge.groupby(['set', args.label_col]).size().unstack(fill_value=0).to_string())

    out = Path(args.out)
    placed, failed = {}, []
    for _, row in merge.iterrows():
        try:
            dm = distance_matrix(*(lambda c: (c, c))(image_to_contour(row['file'], args.n_samples, args.sparsity)))
            sub = SET2DIR[row['set']]
            cls = str(int(row[args.label_col]))
            dst_dir = out / sub / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            np.save(dst_dir / f"{row['stem']}.npy", dm)
            placed[(sub, cls)] = placed.get((sub, cls), 0) + 1
        except Exception as e:
            failed.append((row['stem'], str(e)))

    print(f"\n[OK] wrote {sum(placed.values())} distance matrices ({args.n_samples}x{args.n_samples}) -> {out}")
    for sub in ('train', 'test'):
        r = {c: placed.get((sub, c), 0) for c in '01234'}
        print(f"  {sub:5s}: " + " ".join(f"SC{c}={r[c]}" for c in '01234') + f"  total={sum(r.values())}")
    if failed:
        print(f"  failed: {len(failed)} (e.g. {failed[:2]})")
    print(f"\nNext -> python ShapeEmbedLite.py --train-test-dataset F0 {out}/train {out}/test "
          f"-l 64 -b 0.0 -e 150 --batch-size 16 -n 5 --classify-with-scale -p 0.1 10 -o results/F0_ls128_b0")


if __name__ == '__main__':
    main()
