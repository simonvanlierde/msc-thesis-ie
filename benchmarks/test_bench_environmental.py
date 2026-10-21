"""Benchmarks for the environmental impact functions (cdm/environmental.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cdm.environmental import (
    calculate_environmental_parameters_for_cooling_technologies,
    calculate_impacts_from_cooling_energy_demand,
    calculate_impacts_from_peak_cooling_power_demand,
)

if TYPE_CHECKING:
    import pandas as pd
    from pytest_codspeed import BenchmarkFixture

COOLING_TECHNOLOGIES = ("ASHP", "GSHP", "WSHP", "chiller", "AC_split", "AC_mobile")


@pytest.fixture
def cooling_technology_parameters() -> list[dict[str, float]]:
    return [
        {
            "SEER": 3.0 + i,
            "refrigerant_leakage_rate_relative": 0.02 + 0.01 * i,
            "refrigerant_density_kg_kW": 0.2 + 0.05 * i,
            "material_density_kg_kW": 30.0 + 5.0 * i,
            "average_lifetime_yr": 12.0 + i,
        }
        for i in range(len(COOLING_TECHNOLOGIES))
    ]


@pytest.fixture
def buildings_with_demand(building_stock: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = len(building_stock)
    columns = {
        "E_cooling_capped_at_98th_percentile_kWh": rng.uniform(0.0, 50_000.0, n),
        "P_cooling_peak_98th_percentile_kW": rng.uniform(0.0, 100.0, n),
        "avg_SEER_inv": rng.uniform(0.15, 0.35, n),
        "avg_GHG_emissions_electricity_kgCO2eq_kWh_cooling": rng.uniform(0.05, 0.15, n),
        "avg_GHG_emissions_refrigerant_leaks_kgCO2eq_kW": rng.uniform(10.0, 60.0, n),
        "avg_GHG_emissions_production_phase_kgCO2eq_kW": rng.uniform(5.0, 20.0, n),
        "avg_GHG_emissions_EoL_phase_kgCO2eq_kW": rng.uniform(1.0, 5.0, n),
        "avg_material_density_kg_kW": rng.uniform(20.0, 60.0, n),
        "avg_ADP_kgSbeq_kW": rng.uniform(0.01, 0.05, n),
        "avg_CSI_kgSieq_kW": rng.uniform(0.01, 0.05, n),
    }
    return building_stock.assign(**columns)


def test_calculate_environmental_parameters_for_cooling_technologies(
    benchmark: BenchmarkFixture,
    cooling_technology_parameters: list[dict[str, float]],
    global_parameters: dict[str, float],
) -> None:
    result = benchmark(
        calculate_environmental_parameters_for_cooling_technologies,
        cooling_technology_parameters,
        global_parameters,
    )
    assert len(result) == len(COOLING_TECHNOLOGIES)


def test_calculate_impacts_from_cooling_energy_demand(
    benchmark: BenchmarkFixture,
    buildings_with_demand: pd.DataFrame,
) -> None:
    result = benchmark(calculate_impacts_from_cooling_energy_demand, buildings_with_demand, 98)
    assert "GHG_emissions_electricity_kgCO2eq" in result.columns


def test_calculate_impacts_from_peak_cooling_power_demand(
    benchmark: BenchmarkFixture,
    buildings_with_demand: pd.DataFrame,
) -> None:
    result = benchmark(calculate_impacts_from_peak_cooling_power_demand, buildings_with_demand, 98)
    assert "CSI_kgSieq" in result.columns
