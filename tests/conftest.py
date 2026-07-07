"""Shared fixtures for the cooling demand model test suite.

The fixtures provide small, hand-checkable inputs (a single building, a global
parameter dictionary and a short hourly time series) that mirror the structure of
the real model data without depending on the large Zenodo geospatial datasets.
"""

import numpy as np
import pandas as pd
import pytest

# The eight compass directions, in the order the model expects them.
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@pytest.fixture
def global_parameters() -> dict[str, float]:
    """A minimal but complete set of global parameters used across the model."""
    return {
        "air_density": 1.2,
        "air_heat_capacity": 1005.0,
        "ventilation_efficiency": 0.7,
        "alfa_i": 7.5,
        "alfa_o": 27.5,
        "T_thresh_C": 25.0,
        "T_sub_C": 12.0,
        "int_heat_gain_pp_W": 100.0,
        "int_heat_gain_light_W_m2": 5.0,
        "peak_cooling_percentile_cap": 98,
        "people_density_office": 0.1,
        "people_per_hh": 2.2,
        "building_type_height_cutoff_m": 20.0,
        "building_type_age_cutoff_yr": 1990,
        "building_stock_growth_residential_new": 0.10,
        "building_stock_growth_office_old": 0.20,
    }


@pytest.fixture
def building() -> pd.Series:
    """A single office building row with all attributes the model reads."""
    return pd.Series(
        {
            # Thermal envelope
            "window_area_total_m2": 50.0,
            "wall_area_total_m2": 150.0,
            "floor_area_ground_m2": 100.0,
            "floor_area_total_m2": 300.0,
            "Rc_wall_m2K_W": 2.0,
            "Rc_roof_m2K_W": 3.0,
            "Rc_floor_m2K_W": 2.5,
            "U_window_W_m2K": 1.8,
            "volume_m3": 900.0,
            "infiltration_ACH": 0.5,
            # Use and occupancy
            "end_use": "office",
            "population": 30.0,
            "ventilation_rate_pp_m3_h": 30.0,
            "pressure_drop_Pa": 100.0,
            "int_heat_gain_appliances_W_m2": 8.0,
            # Solar
            "window_area_per_orientation_m2": np.array([10.0, 5.0, 5.0, 5.0, 10.0, 5.0, 5.0, 5.0]),
            "g_window": 0.6,
            # Geometry
            "MBR_width_m": 10.0,
            "MBR_length_m": 20.0,
            "height_m": 15.0,
            "MBR_azimuth": 30.0,
            "f_wall": 0.7,
            "f_window": 0.3,
        },
    )


@pytest.fixture
def time_series() -> dict[str, np.ndarray]:
    """A four-hour time series with weather, solar and occupancy entries."""
    delta_t = np.array([0.0, 2.0, 4.0, 6.0])
    series = {
        "T_outdoor_minus_indoor_C": delta_t,
        "presence_people_office": np.array([0.2, 0.8, 1.0, 0.5]),
        "presence_lighting_office": np.array([0.3, 0.9, 1.0, 0.6]),
        "presence_appliances_office": np.array([0.4, 0.7, 1.0, 0.5]),
    }
    # Constant, distinct solar radiation per direction so per-orientation maths is checkable.
    for i, direction in enumerate(DIRECTIONS):
        series[f"P_sol_{direction}_W_m2"] = np.full(4, 100.0 + i * 10.0)
    return series


@pytest.fixture
def time_series_full_year(time_series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The four-hour pattern tiled into a full 8760-hour year for end-to-end demand tests."""
    repeats = 8760 // 4
    return {key: np.tile(value, repeats) for key, value in time_series.items()}
