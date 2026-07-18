"""Per-building UHImax sampling from the Habib raster."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from shapely.geometry import Point

from scripts.gis.prepare_pdok_model_geodata import sample_uhi_max

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
