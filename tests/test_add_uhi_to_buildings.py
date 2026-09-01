"""Per-building UHImax sampling from the Habib raster."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.transform
from shapely.geometry import Point

from scripts.gis.add_uhi_to_buildings import sample_uhi_max

RASTER = Path("data/input/geodata/UHImax_habib_TheHague_5m.tif")
FALLBACK = 1.0
BUFFER_M = 30


@pytest.mark.skipif(not RASTER.exists(), reason="habib raster not fetched")
def test_sample_inside_city() -> None:
    """Buildings inside The Hague get real neighbourhood-mean values, not all the fallback.

    The raster only has valid cells on outdoor pixels, not building footprints, so a point
    sample at these coordinates would almost always miss -- the regression this guards against.
    """
    gdf = gpd.GeoDataFrame(geometry=[Point(81400, 455200), Point(80000, 454000)], crs="EPSG:28992")
    out = sample_uhi_max(gdf, RASTER, fallback_c=FALLBACK, buffer_m=BUFFER_M)
    assert out["UHI_max_C"].notna().all()
    assert (out["UHI_max_C"] >= 0).all()
    assert (out["UHI_max_C"] < 12).all()
    assert not (out["UHI_max_C"] == FALLBACK).all()


@pytest.mark.skipif(not RASTER.exists(), reason="habib raster not fetched")
def test_sample_outside_coverage_uses_fallback() -> None:
    """A point far outside the raster (Schiermonnikoog) gets the fallback value."""
    gdf = gpd.GeoDataFrame(geometry=[Point(221000, 609500)], crs="EPSG:28992")
    out = sample_uhi_max(gdf, RASTER, fallback_c=FALLBACK, buffer_m=BUFFER_M)
    assert float(out["UHI_max_C"].iloc[0]) == FALLBACK


@pytest.mark.skipif(not RASTER.exists(), reason="habib raster not fetched")
def test_sample_window_fully_on_nodata_uses_fallback() -> None:
    """A point just past the raster's edge, whose whole buffer window misses valid data, falls back."""
    with rasterio.open(RASTER) as src:
        edge_x = src.bounds.left - 200  # buffer window (+/- 30 m) stays clear of the raster entirely
        edge_y = (src.bounds.bottom + src.bounds.top) / 2
    gdf = gpd.GeoDataFrame(geometry=[Point(edge_x, edge_y)], crs="EPSG:28992")
    out = sample_uhi_max(gdf, RASTER, fallback_c=FALLBACK, buffer_m=BUFFER_M)
    assert float(out["UHI_max_C"].iloc[0]) == FALLBACK


def test_sample_window_inside_raster_but_all_nodata_uses_fallback(tmp_path: Path) -> None:
    """A point inside the raster's extent whose window covers only nodata cells falls back.

    Uses a synthetic 40x40, 5 m-pixel raster (200 x 200 m) rather than the real Habib raster:
    real Hague coordinates could not be found where a full +/- 30 m window is guaranteed nodata
    without depending on the raster's current content, and this also directly exercises the
    masked-mean/count logic that the real-data regeneration run (0/2504 fallback) never hit.
    """
    raster_path = tmp_path / "synthetic_uhi.tif"
    transform = rasterio.transform.from_origin(west=0, north=200, xsize=5, ysize=5)
    array = np.full((40, 40), -9999.0, dtype="float32")
    array[:, :20] = 3.0  # left half (x in [0, 100)) valid; right half (x in [100, 200)) nodata
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(array, 1)

    # x=150, y=100: deep in the nodata half; a +/- 30 m window ([120, 180]) never reaches x=100.
    gdf = gpd.GeoDataFrame(geometry=[Point(150, 100)], crs="EPSG:28992")
    out = sample_uhi_max(gdf, raster_path, fallback_c=FALLBACK, buffer_m=BUFFER_M)
    assert float(out["UHI_max_C"].iloc[0]) == FALLBACK


def test_sample_uhi_max_matches_manual_window_mean(tmp_path: Path) -> None:
    """The vectorized (uniform_filter) window mean matches a plain-numpy windowed mean.

    Regression guard for the 2026-07-18 vectorization: builds a 20x20, 5 m-pixel raster with
    ~30% of cells randomly set to nodata, then checks 4 hand-placed points (interior, two
    near-edge, one off-center) against a mean computed by directly slicing the raw array and
    excluding nodata cells -- an independent computation path from ``sample_uhi_max`` itself.
    """
    rng = np.random.default_rng(0)
    size = 20
    res = 5.0
    nodata = -9999.0
    array = rng.uniform(0.0, 5.0, size=(size, size)).astype("float32")
    array[rng.random((size, size)) < 0.3] = nodata

    raster_path = tmp_path / "synthetic_mixed.tif"
    transform = rasterio.transform.from_origin(west=0, north=size * res, xsize=res, ysize=res)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)

    buffer_m = 10.0  # window_size = 2*round(10/5)+1 = 5 -> half-width of 2 pixels
    half = round(buffer_m / res)
    points_rc = [(10, 10), (1, 1), (18, 18), (5, 15)]  # interior, two near-edge, off-center
    fallback_c = -1.0  # sentinel, distinct from any real sampled value

    expected = []
    for row, col in points_rc:
        window = array[max(row - half, 0) : row + half + 1, max(col - half, 0) : col + half + 1]
        valid = window != nodata
        expected.append(float(window[valid].mean()) if valid.any() else fallback_c)

    xs, ys = rasterio.transform.xy(transform, [rc[0] for rc in points_rc], [rc[1] for rc in points_rc])
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys, strict=True)], crs="EPSG:28992")

    out = sample_uhi_max(gdf, raster_path, fallback_c=fallback_c, buffer_m=buffer_m)
    for actual, exp in zip(out["UHI_max_C"], expected, strict=True):
        assert actual == pytest.approx(exp, abs=1e-3)
