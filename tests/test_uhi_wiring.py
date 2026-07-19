"""End-to-end wiring: per-building UHImax x hourly fraction reaches the heat balance."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cdm.parameters import add_derived_parameters_to_buildings
from cdm.thermodynamic import calc_cooling_demand_metrics_for_chunk
from cdm.time_series import create_time_series

if TYPE_CHECKING:
    from pathlib import Path

# The eight window/facade compass directions, in the order create_time_series expects.
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

PRESENCE_LOAD_FACTORS_PATH = "data/input/parameters/presence_load_factors.csv"


def _raw_weather() -> pd.DataFrame:
    """One synthetic day (24 hours) with the columns get_raw_weather_data produces."""
    hours = np.arange(1, 25)
    dates = pd.date_range("2021-07-01", periods=24, freq="h")
    t = 20.0 + 5.0 * np.sin((hours - 9) * np.pi / 12)
    q = np.where((hours > 6) & (hours < 22), 200.0 * np.sin((hours - 6) * np.pi / 15).clip(0), 0.0)
    return pd.DataFrame(
        {
            "date": dates,
            "H": hours,
            "T_outdoor_raw_C": t,
            "Q": q,
            "wind_speed_m_s": 3.0,
        },
    )


def _write_solar_fractions(tmp_path: Path) -> str:
    """A one-day multidirectional solar radiation fractions file matching the raw weather fixture's length."""
    path = tmp_path / "fractions.csv"
    pd.DataFrame({direction: [0.1] * 24 for direction in DIRECTIONS}).to_csv(path, index=False)
    return str(path)


def test_create_time_series_emits_fraction_not_uplift(tmp_path: Path, global_parameters: dict) -> None:
    """T_outdoor_C no longer contains any UHI term; UHI_fraction is exported instead."""
    weather = _raw_weather()
    params = {
        **global_parameters,
        "uhi_day_fraction": 0.25,
        "delta_T_winter_C": 0.0,
        "delta_T_spring_C": 0.0,
        "delta_T_summer_C": 0.0,
        "delta_T_autumn_C": 0.0,
        "delta_P_solar_summer": 0.0,
        "delta_P_solar_RoY": 0.0,
    }
    solar_fractions_path = _write_solar_fractions(tmp_path)

    ts = create_time_series(params, weather, solar_fractions_path, PRESENCE_LOAD_FACTORS_PATH)

    assert "UHI_fraction" in ts
    assert float(np.max(ts["UHI_fraction"])) <= 1.0
    assert float(np.min(ts["UHI_fraction"])) >= 0.0
    # base outdoor temperature equals raw + seasonal boost only (UHI applied per building later):
    assert np.allclose(ts["T_outdoor_C"], weather["T_outdoor_raw_C"].to_numpy() + ts["delta_T_season_C"])


def test_chunk_applies_per_building_uhi(
    building: pd.Series,
    time_series_full_year: dict,
    global_parameters: dict,
) -> None:
    """Two identical buildings differing only in UHI_max_C get demand ordered accordingly."""
    twins = pd.DataFrame([building.to_dict(), building.to_dict()])
    twins.loc[0, "UHI_max_C"] = 0.0
    twins.loc[1, "UHI_max_C"] = 4.0

    result = calc_cooling_demand_metrics_for_chunk(twins, time_series_full_year, global_parameters)

    assert result["E_cooling_kWh"].iloc[1] > result["E_cooling_kWh"].iloc[0]


def test_missing_uhi_max_c_column_falls_back_to_citywide_constant(
    buildings: pd.DataFrame,
    global_parameters: dict,
) -> None:
    """A stock prepared before the UHI raster join (no UHI_max_C column) gets uhi_fallback_C, not a KeyError."""
    # calculate_building_population's np.where evaluates both branches, so the residential
    # branch's column must exist even for this all-office fixture stock.
    stock = buildings.drop(columns=["UHI_max_C"]).assign(number_of_residences=0)

    result = add_derived_parameters_to_buildings(stock, global_parameters)

    assert (result["UHI_max_C"] == global_parameters["uhi_fallback_C"]).all()


def test_nan_uhi_max_c_falls_back_to_citywide_constant(
    buildings: pd.DataFrame,
    global_parameters: dict,
) -> None:
    """A row the raster join left null (outside coverage, no fallback baked in yet) also gets uhi_fallback_C."""
    stock = buildings.copy().assign(number_of_residences=0)
    stock.loc[0, "UHI_max_C"] = np.nan

    result = add_derived_parameters_to_buildings(stock, global_parameters)

    assert result["UHI_max_C"].iloc[0] == global_parameters["uhi_fallback_C"]
