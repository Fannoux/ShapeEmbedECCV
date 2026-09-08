#!/usr/bin/env python3
"""
ShapeEmbedLite latents -> shared features_to_csv format (fish_id, f0..fN, label)
for standard_eval.py --train-csv/--eval-csv.

Validation latents come from ShapeEmbedLite's test_loader (shuffle=False, batch_size=1), so their
row order == the torchvision DatasetFolder sorted order of the DM folder -> fish_id is reconstructed
from the folder with --dm-dir --match-order (and cross-checked against the saved labels).

Training latents come from a shuffled, drop_last loader -> no reliable order; fish_id is a placeholder
(this is fine: the train set is only used to FIT the probe, IDs are not needed for that).
"""
import argparse, os
import numpy as np
import pandas as pd


def folder_stems(dm_dir):
    """DatasetFolder sorted order: sorted class dirs, then sorted filenames within each."""
    classes = sorted(d for d in os.listdir(dm_dir) if os.path.isdir(os.path.join(dm_dir, d)))
    stems, labels = [], []
    for ci, c in enumerate(classes):
        cdir = os.path.join(dm_dir, c)
        for f in sorted(os.listdir(cdir)):
            if f.endswith('.npy'):
                stems.append(os.path.splitext(f)[0])
                labels.append(ci)
    return stems, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--dm-dir', help='DM folder to reconstruct fish_id from (validation/test only)')
    ap.add_argument('--match-order', action='store_true',
                    help='reconstruct fish_id from --dm-dir DatasetFolder order (unshuffled sets only)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    X = np.load(a.latent)
    y = np.load(a.label).astype(int)
    if X.ndim == 1:
        X = X.reshape(len(y), -1)
    n = len(X)

    if a.match_order and a.dm_dir:
        stems, flabels = folder_stems(a.dm_dir)
        if len(stems) != n:
            raise SystemExit(f"[ERR] {len(stems)} stems in {a.dm_dir} != {n} latents (order mismatch)")
        if not np.array_equal(np.asarray(flabels), y):
            raise SystemExit("[ERR] folder-order labels != saved labels -> fish_id mapping NOT safe")
        fish_id = stems
        src = 'folder'
    else:
        fish_id = [f'train_{i}' for i in range(n)]     # placeholder (fit-only set)
        src = 'placeholder'

    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    df.insert(0, 'fish_id', fish_id)
    df['label'] = y
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[OK] {n} rows, {X.shape[1]} dims, ids={src} -> {a.out}")


if __name__ == '__main__':
    main()
