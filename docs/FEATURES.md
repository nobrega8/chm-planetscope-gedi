# Required Input Features

The model expects **99 named raster bands**, spread across one or more
co-registered GeoTIFFs (same grid: shape, transform and CRS). `predict_chm.py`
matches features to bands by **band description** (the name stored in the
raster metadata for each band), not by file or band order — so features can
be split across files however is convenient, as long as every band is
described with the exact name below (case-sensitive).

You can set band descriptions with `rasterio`:

```python
import rasterio

with rasterio.open('my_raster.tif', 'r+') as dst:
    dst.set_band_description(1, 'NDVI')
    dst.set_band_description(2, 'EVI')
    # ...
```

or with GDAL: `gdal_edit.py -mo` doesn't set per-band descriptions directly —
use `gdal.Band.SetDescription()` via the GDAL Python bindings, or `rasterio`
as above.

## 1. Spectral bands (8) — PlanetScope 8-band Surface Reflectance

CoastalBlue, Blue, GreenI, Green, Yellow, Red, RedEdge, NIR

## 2. GLCM texture (10) — computed on a single band (Green in the reference implementation), 5x5 or similar window

Contrast, Dissimilarity, Homogeneity, ASM, Energy, MAX, Entropy, Mean, Variance, Correlation

## 3. Terrain derivatives (30) — from a DEM (DGT LiDAR-derived DTM used in the reference model)

elevation, slope, aspect, tri, twi, roughness, tpi, curvature_plan, curvature_profile,
flow_accum, dist_stream, northness, eastness, heat_load_index, hillshade, hillshade_multi,
elev_range, elev_std, rel_slope_pos, valley_depth, convergence_idx, hand, ridge_dist,
drainage_density, slope_ms3, slope_ms7, tpi_ms3, tpi_ms7, curv_ms3, curv_ms7

`*_ms3` / `*_ms7` are multi-scale variants computed with 3x3 / 7x7 windows.

## 4. Spectral indices (51) — derived from the 8 spectral bands above

NDVI, EVI, SAVI, NGRDI, NDNIRGB, NDNIRGR, NDNIRRB, ATSAVI, TRANSFORMED_NDVI, PAN_NDVI,
MSAVI, GSAVI, GARI, GNDVI, RVI, RDVI, EVI2, OSAVI, WDRVI, VARI, NDRE, NDREI, CIre, CIgreen,
PSRI, NDYI, SIPI, MTCI, IRECI, MCARI, TCARI, REIP, VOG1, VOG2, VOG3, DATT1, Maccioni,
MNDVI705, REYVI, NDGE, PRI, CRI1, CRI2, LI, NYI, ARI1, ARI2, NLI, MNLI, RGVI, CBluNDVI

---

**Important caveats**

- The model was trained exclusively on **forest pixels within GEDI footprints**
  in two Portuguese study areas (Monsanto peri-urban forest, Serra da Lousã
  mountain range). Predictions on pixels far outside that distribution
  (different land cover, different sensors, different geographic/climatic
  context) are extrapolations and should be treated with caution — see the
  main [README](../README.md#limitations).
- The terrain derivatives were computed from a DGT (Direção-Geral do
  Território, Portugal) LiDAR-derived DTM. Substituting a coarser DEM source
  (e.g. SRTM, Copernicus GLO-30) as input will shift the terrain-derivative
  values and is expected to degrade accuracy relative to the reported
  performance.
- Any pixel with a missing/NaN value in **any** of the 99 features is written
  as nodata (`-9999`) in the output, on both bands.
