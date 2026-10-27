"""Inputs for the CodSpeed benchmark suite.

The benchmarks run the pipeline stages that ``scripts/run_cdm_stage.py`` and the sensitivity
analysis call, under the real SQ parameters and one year of the committed backup weather. The
building stock is the integration test's seeded synthetic stock, since the real BAG geometry
is a Zenodo download CI never sees. Every input is identical on every run, so CodSpeed can
compare runs reliably.
"""

import os

# A multi-threaded BLAS (used by the `@` products in the heat-flow functions) stalls under
# CodSpeed's CPU simulation, so pin it to one thread before numpy is first imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import requests  # noqa: E402

from cdm.parameters import add_parameters_to_buildings  # noqa: E402
from cdm.readers import read_global_parameters, read_parameter_specific_data  # noqa: E402
from cdm.sensitivity_analysis import run_impact_stage  # noqa: E402
from cdm.thermodynamic import calc_cooling_demand_metrics_for_df  # noqa: E402
from cdm.time_series import create_time_series, get_raw_weather_data  # noqa: E402
from tests.test_integration import PARAMETER_DIR, PARAMETERS_TOML, SCENARIO, make_synthetic_stock  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

# Small enough to simulate in seconds; the heat balance cost is linear in buildings x hours.
N_BUILDINGS = 200
WEATHER_YEAR = 2022  # One year of the backup file (2018-2022)


@pytest.fixture(scope="session")
def inputs() -> dict:
    """The real SQ parameters, file paths and backup weather, as the pipeline stages take them."""
    global_parameters = read_global_parameters(PARAMETERS_TOML, SCENARIO)
    global_parameters["weather_data_start_year"] = global_parameters["weather_data_end_year"] = WEATHER_YEAR

    def offline(*_args: object, **_kwargs: object) -> None:
        raise requests.exceptions.Timeout

    # Force the committed backup file: the benchmarks must never hit the KNMI API.
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        patch.setattr(requests, "post", offline)
        warnings.simplefilter("ignore", UserWarning)
        raw_weather_data = get_raw_weather_data(global_parameters)

    return {
        "global_parameters": global_parameters,
        "building_type_parameters": read_parameter_specific_data(
            PARAMETER_DIR / "parameters_building_type.csv",
            SCENARIO,
        ),
        "energy_class_parameters": read_parameter_specific_data(
            PARAMETER_DIR / "parameters_energy_class.csv",
            SCENARIO,
        ),
        "cooling_technology_parameters": read_parameter_specific_data(
            PARAMETER_DIR / "parameters_cooling_technology.csv",
            SCENARIO,
        ),
        "raw_weather_data": raw_weather_data,
        "solar_fractions_path": str(PARAMETER_DIR / "multidirectional_solar_radiation_fractions.csv"),
        "presence_load_factors_path": str(PARAMETER_DIR / "presence_load_factors.csv"),
    }


@pytest.fixture(scope="session")
def stock() -> pd.DataFrame:
    """The raw building stock, before parameter assignment."""
    return make_synthetic_stock(n=N_BUILDINGS).assign(id_BAG=np.arange(N_BUILDINGS))


@pytest.fixture(scope="session")
def buildings(stock: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    """The stock with its building type, energy class and derived parameters, as the heat balance takes it."""
    return add_parameters_to_buildings(
        stock.copy(),
        inputs["global_parameters"],
        inputs["building_type_parameters"],
        inputs["energy_class_parameters"],
    )


@pytest.fixture(scope="session")
def time_series(inputs: dict) -> dict[str, np.ndarray]:
    """The hourly model time series for the weather year."""
    return create_time_series(
        inputs["global_parameters"],
        inputs["raw_weather_data"],
        inputs["solar_fractions_path"],
        inputs["presence_load_factors_path"],
    )


@pytest.fixture(scope="session")
def demand(buildings: pd.DataFrame, time_series: dict[str, np.ndarray], inputs: dict) -> dict[bool, pd.DataFrame]:
    """Per-building cooling demand, keyed by whether the hourly series were kept."""
    return {
        include_time_series: calc_cooling_demand_metrics_for_df(
            buildings.copy(),
            time_series,
            inputs["global_parameters"],
            include_time_series,
        )
        for include_time_series in (False, True)
    }


@pytest.fixture(scope="session")
def impacts(demand: dict[bool, pd.DataFrame], inputs: dict) -> dict[bool, pd.DataFrame]:
    """Per-building cooling demand and environmental impacts, keyed like ``demand``."""
    return {
        include_time_series: run_impact_stage(
            buildings,
            inputs["global_parameters"],
            inputs["building_type_parameters"],
            inputs["cooling_technology_parameters"],
        )[0]
        for include_time_series, buildings in demand.items()
    }
