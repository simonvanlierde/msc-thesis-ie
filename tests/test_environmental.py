"""Unit tests for environmental impact functions."""

import pandas as pd
import pytest

from cdm.environmental import (
    calculate_environmental_parameters_for_cooling_technologies,
    calculate_impacts_from_cooling_energy_demand,
    calculate_impacts_from_peak_cooling_power_demand,
)


@pytest.fixture
def global_parameters() -> dict[str, float]:
    return {
        "carbon_intensity_electric_grid_kgCO2eq_kWh": 0.4,
        "gwp_refrigerant_kgCO2eq_kg": 2000.0,
        "carbon_intensity_production_kgCO2eq_kg": 5.0,
        "carbon_intensity_EoL_kgCO2eq_kg": 1.0,
        "adp_intensity_cooling_equipment_kgSbeq_kg": 0.01,
        "csi_intensity_cooling_equipment_kgSbeq_kg": 0.02,
    }


@pytest.fixture
def cooling_technology_parameters() -> list[dict[str, float]]:
    return [
        {
            "SEER": 4.0,
            "refrigerant_leakage_rate_relative": 0.05,
            "refrigerant_density_kg_kW": 0.3,
            "material_density_kg_kW": 40.0,
            "average_lifetime_yr": 15.0,
        },
    ]


def test_derived_parameters_match_hand_calculation(
    global_parameters: dict[str, float],
    cooling_technology_parameters: list[dict[str, float]],
) -> None:
    df = calculate_environmental_parameters_for_cooling_technologies(
        cooling_technology_parameters,
        global_parameters,
    )
    row = df.iloc[0]

    # SEER_inv is the electricity needed per unit of cooling.
    assert row["SEER_inv"] == pytest.approx(1 / 4.0)
    assert row["GHG_emissions_electricity_kgCO2eq_kWh_cooling"] == pytest.approx(0.4 / 4.0)

    # Refrigerant leakage and its climate impact.
    assert row["refrigerant_leakage_kg_kW"] == pytest.approx(0.05 * 0.3)
    assert row["GHG_emissions_refrigerant_leaks_kgCO2eq_kW"] == pytest.approx(2000.0 * 0.05 * 0.3)

    # Annualised production / end-of-life impacts (material_density / lifetime).
    annualised_material = 40.0 / 15.0
    assert row["GHG_emissions_production_phase_kgCO2eq_kW"] == pytest.approx(5.0 * annualised_material)
    assert row["GHG_emissions_EoL_phase_kgCO2eq_kW"] == pytest.approx(1.0 * annualised_material)
    assert row["ADP_kgSbeq_kW"] == pytest.approx(0.01 * annualised_material)
    assert row["CSI_kgSieq_kW"] == pytest.approx(0.02 * annualised_material)


def test_handles_multiple_technologies(
    global_parameters: dict[str, float],
    cooling_technology_parameters: list[dict[str, float]],
) -> None:
    second = dict(cooling_technology_parameters[0], SEER=8.0)
    df = calculate_environmental_parameters_for_cooling_technologies(
        [*cooling_technology_parameters, second],
        global_parameters,
    )
    assert len(df) == 2
    # A more efficient unit (higher SEER) needs less electricity per unit of cooling.
    assert df.iloc[1]["SEER_inv"] < df.iloc[0]["SEER_inv"]


def test_impacts_from_cooling_energy_demand_scale_with_demand() -> None:
    buildings = pd.DataFrame(
        {
            "E_cooling_capped_at_98th_percentile_kWh": [1000.0, 2000.0],
            "avg_SEER_inv": [0.25, 0.5],
            "avg_GHG_emissions_electricity_kgCO2eq_kWh_cooling": [0.1, 0.2],
        },
    )
    result = calculate_impacts_from_cooling_energy_demand(buildings)

    assert result["electricity_use_kWh"].tolist() == pytest.approx([250.0, 1000.0])
    assert result["GHG_emissions_electricity_kgCO2eq"].tolist() == pytest.approx([100.0, 400.0])


def test_impacts_from_peak_power_demand_scale_with_peak_power() -> None:
    buildings = pd.DataFrame(
        {
            "P_cooling_peak_98th_percentile_kW": [10.0],
            "avg_GHG_emissions_refrigerant_leaks_kgCO2eq_kW": [3.0],
            "avg_GHG_emissions_production_phase_kgCO2eq_kW": [2.0],
            "avg_GHG_emissions_EoL_phase_kgCO2eq_kW": [1.0],
            "avg_material_density_kg_kW": [40.0],
            "avg_ADP_kgSbeq_kW": [0.01],
            "avg_CSI_kgSieq_kW": [0.02],
        },
    )
    result = calculate_impacts_from_peak_cooling_power_demand(buildings)

    assert result["GHG_emissions_refrigerant_leaks_kgCO2eq"].iloc[0] == pytest.approx(30.0)
    assert result["GHG_emissions_production_phase_kgCO2eq"].iloc[0] == pytest.approx(20.0)
    assert result["GHG_emissions_EoL_phase_kgCO2eq"].iloc[0] == pytest.approx(10.0)
    assert result["mass_cooling_equipment_kg"].iloc[0] == pytest.approx(400.0)
    assert result["ADP_kgSbeq"].iloc[0] == pytest.approx(0.1)
    assert result["CSI_kgSieq"].iloc[0] == pytest.approx(0.2)
