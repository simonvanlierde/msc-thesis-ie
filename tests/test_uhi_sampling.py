"""Per-building UHImax sampling from the Habib raster."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from scripts.gis.prepare_pdok_model_geodata import sample_uhi_max

RASTER = Path("data/input/geodata/UHImax_habib_TheHague_5m.tif")
FALLBACK = 1.0


@pytest.mark.skipif(not RASTER.exists(), reason="habib raster not fetched")
def test_sample_inside_city() -> None:
    """Buildings inside The Hague get raster values, not the fallback."""
    gdf = gpd.GeoDataFrame(geometry=[Point(81400, 455200), Point(80000, 454000)], crs="EPSG:28992")
    out = sample_uhi_max(gdf, RASTER, fallback_c=FALLBACK)
    assert out["UHI_max_C"].notna().all()
    assert (out["UHI_max_C"] >= 0).all()
    assert (out["UHI_max_C"] < 12).all()


@pytest.mark.skipif(not RASTER.exists(), reason="habib raster not fetched")
def test_sample_outside_coverage_uses_fallback() -> None:
    """A point far outside the raster (Schiermonnikoog) gets the fallback value."""
    gdf = gpd.GeoDataFrame(geometry=[Point(221000, 609500)], crs="EPSG:28992")
    out = sample_uhi_max(gdf, RASTER, fallback_c=FALLBACK)
    assert float(out["UHI_max_C"].iloc[0]) == FALLBACK
