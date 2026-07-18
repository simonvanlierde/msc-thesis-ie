"""Tests for the Habib et al. (2025) UHImax raster fetch."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

RASTER = Path("data/input/geodata/UHImax_habib_TheHague_5m.tif")


@pytest.mark.skipif(not RASTER.exists(), reason="raster not fetched yet (network rule)")
def test_habib_raster_is_valid() -> None:
    """The fetched UHImax raster is RD New, 5 m, plausible-valued, and covers The Hague."""
    with rasterio.open(RASTER) as src:
        assert src.crs.to_epsg() == 28992
        assert abs(src.res[0] - 5.0) < 0.01
        band = src.read(1, masked=True)
        valid = band.compressed()
        assert valid.size > 0
        # canopy-air UHImax for a Dutch city: positive, and far below the old 8.3 C surface figure
        assert float(valid.min()) >= 0.0
        assert float(np.percentile(valid, 99)) < 12.0
        # The Hague city-centre coordinate (RD New) must fall inside the raster bounds
        x, y = 81400, 455200
        assert src.bounds.left < x < src.bounds.right
        assert src.bounds.bottom < y < src.bounds.top
