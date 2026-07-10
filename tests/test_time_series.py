"""Unit tests for the weather and time-series construction helpers."""

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import requests

from cdm import time_series as ts

if TYPE_CHECKING:
    from pathlib import Path


def test_add_uhi_effect_applies_day_and_night_boosts() -> None:
    df = pd.DataFrame({"H": [7, 9, 20, 21]})
    result = ts.add_UHI_effect(df, UHI_effect_day_C=2.0, UHI_effect_night_C=0.5)

    # Day is defined as 8 < H <= 20.
    assert result["UHI_effect_C"].tolist() == [0.5, 2.0, 2.0, 0.5]


def test_add_seasonal_temperature_boosts_maps_each_season() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2022-01-15", "2022-04-15", "2022-07-15", "2022-10-15"])})
    result = ts.add_seasonal_temperature_boosts(
        df,
        delta_T_winter_C=1.0,
        delta_T_spring_C=2.0,
        delta_T_summer_C=3.0,
        delta_T_autumn_C=4.0,
    )

    assert result["delta_T_season_C"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_add_seasonal_solar_radiation_boosts_converts_units() -> None:
    df = pd.DataFrame({"Q": [360.0], "date": pd.to_datetime(["2022-07-15"])})
    result = ts.add_seasonal_solar_radiation_boosts(df, delta_P_solar_summer=0.5, delta_P_solar_RoY=0.0)

    # 360 J/cm2 with a 50% summer boost, converted to W/m2: 360 * 1.5 / 3600 * 10000.
    assert result["P_sol_total_W_m2"].iloc[0] == pytest.approx(360.0 * 1.5 / 3600 * 10000)
    assert "Q_sol_raw_J_cm2" in result.columns


def test_add_multidirectional_solar_radiation(tmp_path: Path) -> None:
    fractions_path = tmp_path / "fractions.csv"
    pd.DataFrame(
        {direction: [0.1, 0.2] for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]},
    ).to_csv(fractions_path, index=False)

    df = pd.DataFrame({"P_sol_total_W_m2": [1000.0, 2000.0], "date": pd.to_datetime(["2022-06-01", "2022-06-02"])})
    result = ts.add_multidirectional_solar_radiation(df, str(fractions_path))

    assert result["P_sol_N_W_m2"].tolist() == pytest.approx([1000.0 * 0.1, 2000.0 * 0.2])


def test_add_presence_load_factors_joins_repeated_daily_profile(tmp_path: Path) -> None:
    factors_path = tmp_path / "presence.csv"
    pd.DataFrame({"presence_people_office": list(range(24))}).to_csv(factors_path, index=False)

    # A two-day hourly time series; the one-day profile is repeated to match its length.
    time_series_df = pd.DataFrame({"T_outdoor_C": list(range(48))})
    result = ts.add_presence_load_factors(time_series_df, str(factors_path))

    assert "presence_people_office" in result.columns
    assert result["presence_people_office"].tolist() == list(range(24)) * 2


def test_add_presence_load_factors_rejects_a_partial_day(tmp_path: Path) -> None:
    """A series that isn't a whole number of days must fail loud, not silently NaN the tail hours."""
    factors_path = tmp_path / "presence.csv"
    pd.DataFrame({"presence_people_office": list(range(24))}).to_csv(factors_path, index=False)

    # 30 hours: one full day plus a partial one. int(30/24)=1 tile would leave hours 24-29 unmatched.
    time_series_df = pd.DataFrame({"T_outdoor_C": list(range(30))})

    with pytest.raises(ValueError, match="whole number of 24-hour days"):
        ts.add_presence_load_factors(time_series_df, str(factors_path))


def test_read_dynamic_subsurface_temperature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the local backup by making the KNMI download fail.
    def fake_get(*_args: object, **_kwargs: object) -> None:
        raise requests.exceptions.ConnectionError

    monkeypatch.setattr(ts.requests, "get", fake_get)

    path = tmp_path / "subsurface.csv"
    header_skip = "\n".join(f"# comment line {i}" for i in range(16))
    rows = "\n".join(
        f"260,20200601,{hour}, 53, 56, 55, 58, {value}"
        for hour, value in zip([6, 12, 18, 24], [100, 110, 120, 130], strict=True)
    )
    path.write_text(f"{header_skip}\n# STN,YYYYMMDD,HH,  TB1,  TB2,  TB3,  TB4,  TB5\n{rows}\n")

    with pytest.warns(UserWarning, match="backup"):
        result = ts.read_dynamic_subsurface_temperature(
            str(path),
            start_year=2020,
            end_year=2020,
            measurement_depth_cm=100,
        )

    # Four 6-hourly TB5 readings, each repeated to hourly resolution and divided by 10.
    assert len(result) == 24
    assert result[0] == pytest.approx(10.0)
    assert result[-1] == pytest.approx(13.0)


def test_read_dynamic_subsurface_temperature_rejects_bad_depth(tmp_path: Path) -> None:
    path = tmp_path / "subsurface.csv"
    path.write_text("\n".join(f"# line {i}" for i in range(20)))

    with pytest.raises(ValueError, match="measurement_depth"):
        ts.read_dynamic_subsurface_temperature(str(path), start_year=2020, end_year=2020, measurement_depth_cm=42)


def test_get_raw_weather_data_falls_back_to_local_file_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
    global_parameters: dict,
) -> None:
    # The KNMI request times out, so the locally cached backup file must be used instead.
    def fake_post(*_args: object, **_kwargs: object) -> None:
        raise requests.exceptions.Timeout

    monkeypatch.setattr(ts.requests, "post", fake_post)
    params = {
        **global_parameters,
        "weather_data_start_year": 2018,
        "weather_data_end_year": 2022,
        "weather_station": 330,
    }

    with pytest.warns(UserWarning, match="backup"):
        result = ts.get_raw_weather_data(params)

    assert not result.empty
    assert "T_outdoor_raw_C" in result.columns
    assert "date" in result.columns
