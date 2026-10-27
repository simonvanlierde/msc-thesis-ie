"""Benchmarks for the pipeline stages, in the order a model run calls them.

CodSpeed's flame graph for each benchmark shows the time per inner function (each calc_Q_*
heat flow, the percentile cap, the UHI kernel), so they need no benchmarks of their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cdm.aggregation import aggregate_results
from cdm.parameters import add_parameters_to_buildings
from cdm.sensitivity_analysis import run_impact_stage
from cdm.thermodynamic import calc_cooling_demand_metrics_for_df
from cdm.time_series import create_time_series

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from pytest_codspeed import BenchmarkFixture


def test_add_parameters_to_buildings(benchmark: BenchmarkFixture, stock: pd.DataFrame, inputs: dict) -> None:
    # The function writes into its input, so each round gets a fresh copy.
    result = benchmark(
        lambda: add_parameters_to_buildings(
            stock.copy(),
            inputs["global_parameters"],
            inputs["building_type_parameters"],
            inputs["energy_class_parameters"],
        ),
    )
    assert result["wall_area_total_m2"].notna().all()


def test_create_time_series(benchmark: BenchmarkFixture, inputs: dict) -> None:
    result = benchmark(
        create_time_series,
        inputs["global_parameters"],
        inputs["raw_weather_data"],
        inputs["solar_fractions_path"],
        inputs["presence_load_factors_path"],
    )
    assert len(result["UHI_fraction"]) == 8760


# False is the sensitivity analysis path; True keeps the hourly series for the figures.
@pytest.mark.parametrize("include_time_series", [False, True])
def test_calc_cooling_demand_metrics_for_df(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    inputs: dict,
    include_time_series: bool,
) -> None:
    result = benchmark(
        lambda: calc_cooling_demand_metrics_for_df(
            buildings.copy(),
            time_series,
            inputs["global_parameters"],
            include_time_series,
        ),
    )
    assert (result["E_cooling_kWh"] >= 0).all()


def test_run_impact_stage(benchmark: BenchmarkFixture, demand: dict[bool, pd.DataFrame], inputs: dict) -> None:
    # The sensitivity analysis repeats this stage once per cooling technology variable.
    result, summary = benchmark(
        run_impact_stage,
        demand[False],
        inputs["global_parameters"],
        inputs["building_type_parameters"],
        inputs["cooling_technology_parameters"],
    )
    assert result["GHG_emissions_total_kgCO2eq"].notna().all()
    assert summary


@pytest.mark.parametrize("include_time_series", [False, True])
def test_aggregate_results(
    benchmark: BenchmarkFixture,
    impacts: dict[bool, pd.DataFrame],
    inputs: dict,
    include_time_series: bool,
) -> None:
    result = benchmark(
        aggregate_results,
        impacts[include_time_series],
        inputs["global_parameters"],
        include_time_series=include_time_series,
    )
    assert result["id_BAG"].sum() == len(impacts[include_time_series])
