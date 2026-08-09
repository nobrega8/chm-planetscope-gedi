#!/usr/bin/env python3
"""Predict Canopy Height (CHM) and per-pixel uncertainty from a stacking ensemble.

Applies the trained 9-model stacking ensemble (RF, XGB, LGB, ET, SVR, Ridge,
Lasso, ElasticNet, KNN base learners + Ridge meta-learner) pixel-by-pixel to
one or more co-registered input rasters, producing a 2-band GeoTIFF:

  Band 1 - CHM_predicted_m   : predicted canopy height (m)
  Band 2 - CHM_uncertainty_m : std. deviation across the 9 base models (m)

Input rasters must share the exact same grid (shape + transform + CRS) and
must carry band descriptions matching the required feature names (see
docs/FEATURES.md). Features can be spread across several files (e.g. one
raster for spectral bands, one for indices, one for GLCM texture, one for
terrain) - the script matches features to bands by description, not by file.

Example
-------
    python predict_chm.py \\
        --inputs composite_8b.tif indices_8b.tif glcm.tif terrain_dgt.tif \\
        --models-dir models/ \\
        --output CHM_predicted.tif
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

INCLUDED_MODELS = ['RF', 'XGB', 'LGB', 'ET', 'SVR', 'Ridge', 'Lasso', 'ElasticNet', 'KNN']
NODATA = -9999.0
CM_TO_M = 0.01  # base/meta models were trained on the target in centimeters


def load_ensemble(models_dir: Path):
    base_models = {m: joblib.load(models_dir / f'{m}_base.joblib') for m in INCLUDED_MODELS}
    meta = joblib.load(models_dir / 'meta_ridge.joblib')
    feature_names = list(base_models['RF'].feature_names_in_)
    return base_models, meta, feature_names


def build_band_lookup(srcs: dict) -> dict:
    """Map feature name -> (source key, 1-based band index) using band descriptions."""
    lookup = {}
    for key, s in srcs.items():
        for i, name in enumerate(s.descriptions, start=1):
            if name:
                lookup[name] = (key, i)
    return lookup


def predict(
    input_paths: list[Path],
    models_dir: Path,
    out_path: Path,
    row_chunk: int = 300,
    max_height_m: float = 70.0,
) -> None:
    base_models, meta, feature_names = load_ensemble(models_dir)
    print(f'Loaded ensemble from {models_dir}  ({len(feature_names)} features, '
          f'{len(INCLUDED_MODELS)} base models)')

    srcs = {str(p): rasterio.open(p) for p in input_paths}
    ref = next(iter(srcs.values()))

    for key, s in srcs.items():
        if (s.height, s.width) != (ref.height, ref.width) or s.transform != ref.transform:
            raise ValueError(f'Grid mismatch: "{key}" does not match "{next(iter(srcs))}" '
                              '(shape/transform must be identical across all inputs)')

    band_lookup = build_band_lookup(srcs)
    missing = [f for f in feature_names if f not in band_lookup]
    if missing:
        raise ValueError(
            f'Missing {len(missing)} required feature(s) in input band descriptions: {missing}\n'
            'See docs/FEATURES.md for the full list of required features and their expected names.'
        )

    out_profile = ref.profile.copy()
    out_profile.update(count=2, dtype='float32', nodata=NODATA, compress='lzw')

    height, width = ref.height, ref.width
    t_start = time.time()

    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.set_band_description(1, 'CHM_predicted_m')
        dst.set_band_description(2, 'CHM_uncertainty_m')

        for row0 in range(0, height, row_chunk):
            nrows = min(row_chunk, height - row0)
            window = Window(0, row0, width, nrows)

            cols = []
            for feat in feature_names:
                key, bidx = band_lookup[feat]
                arr = srcs[key].read(bidx, window=window, masked=True).astype(np.float32)
                cols.append(np.ma.filled(arr, np.nan).ravel())
            x_block = np.column_stack(cols)  # (nrows*width, n_features)

            valid = np.all(np.isfinite(x_block), axis=1)
            pred_chm = np.full(x_block.shape[0], NODATA, dtype=np.float32)
            pred_std = np.full(x_block.shape[0], NODATA, dtype=np.float32)

            if valid.any():
                xv = pd.DataFrame(x_block[valid], columns=feature_names)
                base_preds_cm = np.column_stack([base_models[m].predict(xv) for m in INCLUDED_MODELS])
                stack_pred_cm = np.clip(meta.predict(base_preds_cm), 0, None)
                pred_chm[valid] = np.clip(stack_pred_cm * CM_TO_M, 0, max_height_m)
                pred_std[valid] = base_preds_cm.std(axis=1) * CM_TO_M

            dst.write(pred_chm.reshape(nrows, width), 1, window=window)
            dst.write(pred_std.reshape(nrows, width), 2, window=window)

            done = row0 + nrows
            elapsed = time.time() - t_start
            print(f'  {done}/{height} rows ({100 * done / height:5.1f}%)  {elapsed:6.0f}s elapsed', flush=True)

    for s in srcs.values():
        s.close()
    print(f'Saved to {out_path}  ({time.time() - t_start:.0f}s total)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', nargs='+', required=True, type=Path,
                         help='One or more co-registered raster files carrying the required '
                              'features as named bands (see docs/FEATURES.md).')
    parser.add_argument('--models-dir', type=Path, default=Path(__file__).parent / 'models',
                         help='Directory with the trained *_base.joblib / meta_ridge.joblib files '
                              '(default: ./models next to this script).')
    parser.add_argument('--output', type=Path, required=True, help='Output GeoTIFF path.')
    parser.add_argument('--row-chunk', type=int, default=300,
                         help='Rows processed per block (trade-off memory vs. overhead). Default: 300.')
    parser.add_argument('--max-height-m', type=float, default=70.0,
                         help='Cap applied to predicted heights (m). The models only saw forest '
                              'pixels during training; on out-of-distribution pixels (e.g. buildings, '
                              'roads) extrapolation can spike unrealistically. Default: 70.0, the max '
                              'plausible height observed in the reference LiDAR data used for training. '
                              'Set to a large value (e.g. 1e6) to disable capping.')
    args = parser.parse_args()

    missing_inputs = [p for p in args.inputs if not p.exists()]
    if missing_inputs:
        sys.exit(f'Input file(s) not found: {missing_inputs}')
    if not args.models_dir.exists():
        sys.exit(f'Models directory not found: {args.models_dir}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predict(args.inputs, args.models_dir, args.output, args.row_chunk, args.max_height_m)


if __name__ == '__main__':
    main()
