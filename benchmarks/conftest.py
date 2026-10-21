"""Synthetic, deterministic inputs for the CodSpeed benchmark suite.

The benchmarks exercise the model's computational core on a building stock and an hourly
weather year that mirror the structure of the real data, without needing the Zenodo
geospatial datasets. A fixed random seed keeps every run identical, so CodSpeed can compare
runs reliably.
"""

import os

# A multi-threaded BLAS (used by the `@` products in the heat-flow functions) stalls under
# CodSpeed's CPU simulation, so pin it to one thread before numpy is first imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from cdm.constants import HOURS_PER_YEAR, SOLAR_DIRECTIONS  # noqa: E402

END_USES = ("residential", "office")
N_BUILDINGS_HEAT_FLOWS = 200  # (200 buildings, 8760 hours) blocks: realistic, yet fast to simulate
N_BUILDINGS_STOCK = 10_000  # Whole-stock vectorised operations
N_WEATHER_YEARS = 5  # The model runs on weather from 2021-2025


@pytest.fixture
def global_parameters() -> dict[str, float]:
    """The global model parameters read by the benchmarked functions."""
    return {
        "air_density": 1.2,
        "air_heat_capacity": 1005.0,
        "alfa_i": 7.5,
        "alfa_o": 27.5,
        "T_thresh_C": 25.0,
        "T_sub_C": 12.0,
        "int_heat_gain_pp_W": 100.0,
        "int_heat_gain_light_W_m2": 5.0,
        "peak_cooling_percentile_cap": 98,
        "people_density_office": 0.1,
        "people_per_hh": 2.2,
        "uhi_scale": 1.0,
        "uhi_fallback_C": 1.0,
        "carbon_intensity_electric_grid_kgCO2eq_kWh": 0.4,
        "gwp_refrigerant_kgCO2eq_kg": 2000.0,
        "carbon_intensity_production_kgCO2eq_kg": 5.0,
        "carbon_intensity_EoL_kgCO2eq_kg": 1.0,
        "adp_intensity_cooling_equipment_kgSbeq_kg": 0.01,
        "csi_intensity_cooling_equipment_kgSbeq_kg": 0.02,
    }


def make_buildings(n: int, seed: int = 0) -> pd.DataFrame:
    """A synthetic building stock with every attribute the heat-balance model reads."""
    rng = np.random.default_rng(seed)
    width = rng.uniform(5.0, 20.0, n)
    length = width * rng.uniform(1.0, 3.0, n)
    height = rng.uniform(3.0, 40.0, n)
    f_window = rng.uniform(0.2, 0.5, n)
    window_area_per_orientation = rng.uniform(0.0, 20.0, (n, len(SOLAR_DIRECTIONS)))
    floor_area_ground = width * length
    return pd.DataFrame(
        {
            "window_area_total_m2": window_area_per_orientation.sum(axis=1),
            "wall_area_total_m2": 2 * (width + length) * height * (1 - f_window),
            "floor_area_ground_m2": floor_area_ground,
            "floor_area_total_m2": floor_area_ground * np.maximum(1, height // 3),
            "Rc_wall_m2K_W": rng.uniform(0.5, 4.5, n),
            "Rc_roof_m2K_W": rng.uniform(0.5, 6.0, n),
            "Rc_floor_m2K_W": rng.uniform(0.5, 3.5, n),
            "U_window_W_m2K": rng.uniform(1.0, 5.0, n),
            "volume_m3": floor_area_ground * height,
            "infiltration_ACH": rng.uniform(0.2, 1.5, n),
            "UHI_max_C": rng.uniform(0.0, 4.0, n),
            "end_use": rng.choice(END_USES, n),
            "population": rng.uniform(1.0, 100.0, n),
            "ventilation_rate_pp_m3_h": rng.uniform(20.0, 40.0, n),
            "int_heat_gain_appliances_W_m2": rng.uniform(2.0, 10.0, n),
            "window_area_per_orientation_m2": list(window_area_per_orientation),
            "g_window": rng.uniform(0.3, 0.8, n),
            "MBR_width_m": width,
            "MBR_length_m": length,
            "height_m": height,
            "MBR_azimuth": rng.uniform(0.0, 180.0, n),
            "f_wall": 1 - f_window,
            "f_window": f_window,
        },
    )


@pytest.fixture
def buildings() -> pd.DataFrame:
    """A mixed residential/office stock sized for the hourly heat-flow benchmarks."""
    return make_buildings(N_BUILDINGS_HEAT_FLOWS)


@pytest.fixture
def building_stock() -> pd.DataFrame:
    """A larger stock for the whole-stock vectorised (non-hourly) benchmarks."""
    return make_buildings(N_BUILDINGS_STOCK, seed=1)


@pytest.fixture
def time_series() -> dict[str, np.ndarray]:
    """One hourly year of weather, solar radiation, UHI and presence profiles (float32, as the model uses)."""
    rng = np.random.default_rng(2)
    hours = np.arange(HOURS_PER_YEAR)
    seasonal = -10.0 * np.cos(2 * np.pi * hours / HOURS_PER_YEAR)
    diurnal = -4.0 * np.cos(2 * np.pi * hours / 24)
    daylight = np.clip(np.sin(2 * np.pi * (hours % 24 - 6) / 24), 0, None)
    series = {
        "T_outdoor_minus_indoor_C": seasonal + diurnal - 10.0 + rng.normal(0.0, 2.0, HOURS_PER_YEAR),
        "UHI_fraction": rng.uniform(0.0, 1.0, HOURS_PER_YEAR),
    }
    for i, direction in enumerate(SOLAR_DIRECTIONS):
        series[f"P_sol_{direction}_W_m2"] = daylight * (100.0 + 20.0 * i) * rng.uniform(0.5, 1.0, HOURS_PER_YEAR)
    for use in END_USES:
        for load in ("people", "lighting", "appliances"):
            series[f"presence_{load}_{use}"] = rng.uniform(0.0, 1.0, HOURS_PER_YEAR)
    return {name: values.astype(np.float32) for name, values in series.items()}


@pytest.fixture
def weather_df() -> pd.DataFrame:
    """Several years of hourly KNMI-style weather, as read before the time series is built."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2021-01-01", periods=N_WEATHER_YEARS * HOURS_PER_YEAR, freq="h")
    hour = np.arange(len(dates)) % 24  # The series starts at midnight
    daylight = np.clip(np.sin(2 * np.pi * (hour - 6) / 24), 0, None)
    return pd.DataFrame(
        {
            "date": dates,
            "H": hour + 1,
            "T_outdoor_raw_C": 10.0 + 8.0 * daylight + rng.normal(0.0, 3.0, len(dates)),
            "Q": np.round(daylight * rng.uniform(0.0, 300.0, len(dates))),
            "wind_speed_m_s": rng.uniform(0.0, 12.0, len(dates)),
        },
    )
