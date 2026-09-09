#!/usr/bin/env python3
"""
Organise extracted distance-matrix .npy files into the class-subfolder layout that
ShapeEmbedLite's DatasetFolder expects, using the fixed adjusted split.

Produces:
    OUT/train/<class>/*.npy   (manifest set == 'training')
    OUT/test/<class>/*.npy    (manifest set == 'validation'  -> ShapeEmbed's "test")

Then run:
    python ShapeEmbedLite.py --train-test-dataset name OUT/train OUT/test ...

Matching: each .npy is matched to a manifest row by its filename stem (the full image
name). A trailing suffix like '_dm' / '_preprocessed_dm' is tolerated; a unique
substring match is used as a fallback. Unmatched (unlabelled) matrices are skipped.

Usage:
    python sort_matrices_to_classfolders.py --matrices distmat_raw \
        --manifest shapeembed_manifest.csv --out distmat [--copy]
"""

import argparse
import glob
import os
import shutil
import pandas as pd

SET2DIR = {'training': 'train', 'validation': 'test'}


def resolve_stem(npy_name, stems):
    base = os.path.basename(npy_name)
    base = base[:-4] if base.endswith('.npy') else base
    if base in stems:
        return base
    for suf in ('_preprocessed_dm', '_reconstructed_dm', '_dm', '_distance_matrix', '_matrix'):
        if base.endswith(suf) and base[:-len(suf)] in stems:
            return base[:-len(suf)]
    hits = [s for s in stems if s in base]           # unique substring fallback
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrices', required=True, help='folder of extracted .npy distance matrices')
    ap.add_argument('--manifest', default='shapeembed_manifest.csv')
    ap.add_argument('--out', default='distmat')
    ap.add_argument('--copy', action='store_true', help='copy files (default: symlink)')
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    info = {r['stem']: (str(SET2DIR[r['set']]), str(int(r['label'])))
            for _, r in man.iterrows()}
    stems = set(info)

    npys = sorted(glob.glob(os.path.join(args.matrices, '*.npy')))
    placed, skipped = {}, []
    for p in npys:
        stem = resolve_stem(p, stems)
        if stem is None:
            skipped.append(os.path.basename(p)); continue
        sub, cls = info[stem]
        dst_dir = os.path.join(args.out, sub, cls)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(p))
        if os.path.lexists(dst):
            os.remove(dst)
        (shutil.copy2 if args.copy else os.symlink)(os.path.abspath(p), dst)
        placed[(sub, cls)] = placed.get((sub, cls), 0) + 1

    print(f"matrices found: {len(npys)} | placed: {sum(placed.values())} | unmatched(skipped): {len(skipped)}")
#    for sub in ('train', 'test'):
#        row = {c: placed.get((sub, c), 0) for c in '01234'}
#        print(f"  {sub:5s}: " + " ".join(f"SC{c}={row[c]}" for c in '01234') + f"  total={sum(row.values())}")
    if skipped:
        print(f"  e.g. unmatched: {skipped[:3]}")
    print(f"\nReady -> python ShapeEmbedLite.py --train-test-dataset name {args.out}/train {args.out}/test ...")


if __name__ == '__main__':
    main()
