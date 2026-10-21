"""Benchmarks for the hourly heat-balance model (cdm/thermodynamic.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cdm.thermodynamic import (
    calc_cooling_demand_from_thermal_flows,
    calc_cooling_demand_metrics_for_df,
    calc_cooling_demand_percentile,
    calc_delta_T_air,
    calc_Q_infiltration,
    calc_Q_internal_heat,
    calc_Q_solar_radiation,
    calc_Q_transmission,
    calc_Q_ventilation,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from pytest_codspeed import BenchmarkFixture


@pytest.fixture
def delta_T_air(
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    return calc_delta_T_air(buildings, time_series, global_parameters)


@pytest.fixture
def heat_flows(
    buildings: pd.DataFrame,
    delta_T_air: np.ndarray,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> tuple[np.ndarray, ...]:
    return (
        calc_Q_transmission(buildings, delta_T_air, global_parameters),
        calc_Q_infiltration(buildings, delta_T_air, global_parameters),
        calc_Q_ventilation(buildings, delta_T_air, time_series, global_parameters),
        calc_Q_solar_radiation(buildings, time_series),
        calc_Q_internal_heat(buildings, time_series, global_parameters),
    )


def test_calc_delta_T_air(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> None:
    result = benchmark(calc_delta_T_air, buildings, time_series, global_parameters)
    assert result.shape == (len(buildings), len(time_series["UHI_fraction"]))


def test_calc_Q_transmission(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    delta_T_air: np.ndarray,
    global_parameters: dict[str, float],
) -> None:
    result = benchmark(calc_Q_transmission, buildings, delta_T_air, global_parameters)
    assert result.shape == delta_T_air.shape


def test_calc_Q_infiltration(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    delta_T_air: np.ndarray,
    global_parameters: dict[str, float],
) -> None:
    result = benchmark(calc_Q_infiltration, buildings, delta_T_air, global_parameters)
    assert result.shape == delta_T_air.shape


def test_calc_Q_ventilation(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    delta_T_air: np.ndarray,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> None:
    result = benchmark(calc_Q_ventilation, buildings, delta_T_air, time_series, global_parameters)
    assert result.shape == delta_T_air.shape


def test_calc_Q_solar_radiation(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
) -> None:
    result = benchmark(calc_Q_solar_radiation, buildings, time_series)
    assert result.shape == (len(buildings), len(time_series["UHI_fraction"]))


def test_calc_Q_internal_heat(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> None:
    result = benchmark(calc_Q_internal_heat, buildings, time_series, global_parameters)
    assert result.shape == (len(buildings), len(time_series["UHI_fraction"]))


def test_calc_cooling_demand_from_thermal_flows(
    benchmark: BenchmarkFixture,
    heat_flows: tuple[np.ndarray, ...],
) -> None:
    Q_cooling, E_cooling, P_peak = benchmark(calc_cooling_demand_from_thermal_flows, *heat_flows)
    assert Q_cooling.shape == heat_flows[0].shape
    assert (E_cooling >= 0).all()
    assert (P_peak >= 0).all()


@pytest.mark.parametrize("include_time_series", [False, True])
def test_calc_cooling_demand_percentile(
    benchmark: BenchmarkFixture,
    heat_flows: tuple[np.ndarray, ...],
    include_time_series: bool,
) -> None:
    Q_cooling, _, _ = calc_cooling_demand_from_thermal_flows(*heat_flows)

    # Without the hourly series the percentile partitions its input in place, so each round gets a fresh copy.
    def run() -> tuple:
        return calc_cooling_demand_percentile(Q_cooling.copy(), 98, include_time_series)

    _, P_peak, _, E_capped = benchmark(run)
    assert P_peak.shape == E_capped.shape == (len(Q_cooling),)


@pytest.mark.parametrize("include_time_series", [False, True])
def test_calc_cooling_demand_metrics_for_df(
    benchmark: BenchmarkFixture,
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
    include_time_series: bool,
) -> None:
    result = benchmark(
        calc_cooling_demand_metrics_for_df,
        buildings,
        time_series,
        global_parameters,
        include_time_series,
    )
    assert "E_cooling_capped_at_98th_percentile_kWh" in result.columns
