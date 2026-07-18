"""Add per-building UHImax to a prepared buildings GeoPackage.

Split out from ``prepare_pdok_model_geodata.py`` (2026-07-18) so that model-only edits
(``cdm/*.py``) don't retrigger a UHI resample: this rule's Snakefile inputs are the Habib
raster and this script alone, not ``model_src``, so every cdm iteration stops paying the
sampling cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def sample_uhi_max(
    buildings_gdf: gpd.GeoDataFrame,
    raster_path: Path,
    fallback_c: float,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Sample the UHImax raster in a neighbourhood around each building's representative point.

    The Habib raster defines UHImax on outdoor (street-canyon / pedestrian-level) cells only,
    not on building footprints, so a point sample at the representative point almost always
    lands on nodata (verified 2026-07-18: 199/200 sample-building points hit nodata, but
    200/200 had a valid cell within 30 m). This instead averages the valid cells in a square
    window of +/- ``buffer_m`` around the point -- also the physically relevant quantity, since
    it is the ambient air UHI immediately surrounding the building. Do not "simplify" this back
    to a point sample. Buildings whose window contains no valid cell at all (other
    municipalities, raster edges) receive ``fallback_c`` -- the citywide mean air UHI -- so the
    model keeps working for cities the Habib dataset does not cover.

    Vectorized (2026-07-18): rather than reading a window per building (~30+ min for the full
    stock), the raster's band is read once and a windowed mean is precomputed for every pixel
    via ``scipy.ndimage.uniform_filter``, then every building's window mean is a single
    vectorized row/col lookup -- seconds for the whole stock. ``uniform_filter`` computes a box
    *mean*, not a sum, but since both the numerator (sum of valid values) and denominator (count
    of valid cells) go through the identical filter, the shared 1/window_size**2 factor cancels
    in the ratio, leaving the true mean of valid cells in the window. Box size is
    ``2*round(buffer_m/res)+1`` pixels to match the old +/-``buffer_m`` window; the result is a
    close but not bit-identical match to the old per-building read (different edge/rounding
    handling) -- expected and acceptable, since this is a spatial average, not an exact value.
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)
        transform = src.transform
        res = src.res[0]

    valid = ~np.ma.getmaskarray(band)
    values = np.where(valid, band.filled(0.0), 0.0).astype(np.float64)

    window_size = 2 * round(buffer_m / res) + 1
    filtered_sum = uniform_filter(values, size=window_size, mode="constant", cval=0.0)
    filtered_count = uniform_filter(valid.astype(np.float64), size=window_size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        window_mean = np.where(filtered_count > 0, filtered_sum / filtered_count, fallback_c)

    xs = np.array([geom.representative_point().x for geom in buildings_gdf.geometry])
    ys = np.array([geom.representative_point().y for geom in buildings_gdf.geometry])
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)

    in_bounds = (rows >= 0) & (rows < band.shape[0]) & (cols >= 0) & (cols < band.shape[1])
    sampled = np.full(len(buildings_gdf), fallback_c, dtype=np.float64)
    sampled[in_bounds] = window_mean[rows[in_bounds], cols[in_bounds]]

    buildings_gdf["UHI_max_C"] = sampled
    return buildings_gdf


def main() -> None:
    """Add the UHImax column to an already-prepared buildings GeoPackage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--buildings", required=True, help="Prepared buildings GeoPackage (no UHI column yet).")
    parser.add_argument("--buildings-layer", required=True)
    parser.add_argument("--uhi-raster", required=True, help="Per-building UHImax GeoTIFF (Habib et al. 2025).")
    parser.add_argument(
        "--uhi-fallback-c",
        type=float,
        required=True,
        help="UHImax for buildings outside raster coverage.",
    )
    parser.add_argument(
        "--uhi-buffer-m",
        type=float,
        default=30,
        help="Radius (m) of the square window averaged around each building for UHImax sampling.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", required=True)
    args = parser.parse_args()

    buildings = gpd.read_file(args.buildings, layer=args.buildings_layer)
    buildings = sample_uhi_max(
        buildings,
        Path(args.uhi_raster),
        fallback_c=args.uhi_fallback_c,
        buffer_m=args.uhi_buffer_m,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(output, layer=args.layer, driver="GPKG")


if __name__ == "__main__":
    main()
