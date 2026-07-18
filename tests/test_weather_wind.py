"""The weather series carries hourly mean wind speed for the UHI weather factor."""

from pathlib import Path

import pandas as pd
import pytest

WEATHER = sorted(Path("results/weather").glob("knmi_330_*.csv"))


@pytest.mark.skipif(not WEATHER, reason="no weather CSV fetched")
def test_wind_speed_column() -> None:
    """wind_speed_m_s exists, is in m/s (not 0.1 m/s), and is plausible for Hoek van Holland."""
    df = pd.read_csv(WEATHER[-1])
    assert "wind_speed_m_s" in df.columns
    wind = df["wind_speed_m_s"].dropna()
    assert len(wind) > 0.99 * len(df)
    assert float(wind.min()) >= 0.0
    assert float(wind.max()) < 45.0
    # coastal-station annual mean is well above 3 m/s; a 0.1 m/s unit slip would make this ~50
    assert 3.0 < float(wind.mean()) < 10.0
