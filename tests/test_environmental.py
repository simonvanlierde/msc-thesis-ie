"""Unit tests for environmental impact functions."""

import pytest

from functions.environmental import (
    calculate_environmental_parameters_for_cooling_technologies,
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
        }
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
