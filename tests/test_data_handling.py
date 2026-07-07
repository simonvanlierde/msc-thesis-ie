"""Unit tests for the deterministic data-handling helpers.

The heavy GeoPackage readers depend on the external Zenodo datasets and are not
exercised here; these tests cover the pure mapping and scaling logic.
"""

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from cdm.aggregation import scale_results_with_building_stock
from cdm.constants import REQUIRED_GLOBAL_PARAMETERS
from cdm.parameters import (
    assign_building_type_parameters,
    assign_energy_class_parameters,
    calculate_building_population,
    determine_building_type,
    determine_energy_label_to_class_mappings,
)
from cdm.readers import read_global_parameters, read_parameter_specific_data

if TYPE_CHECKING:
    from pathlib import Path


def test_read_global_parameters_merges_base_and_scenario(tmp_path: Path) -> None:
    path = tmp_path / "parameters.toml"
    # A complete [base] (all required keys, T_thresh_C among them) plus the specifics under test.
    base = dict.fromkeys(REQUIRED_GLOBAL_PARAMETERS, 0.0)
    base["T_thresh_C"] = 24.0
    base_lines = "\n".join(f"{key} = {value}" for key, value in base.items())
    path.write_text(
        f'[base]\n{base_lines}\nenergy_class_ranges = [[0, 1], [1, 2]]\n\n[scenario."SQ"]\nT_thresh_C = 25.0\n',
    )

    parameters = read_global_parameters(path, "SQ")

    assert parameters["T_thresh_C"] == pytest.approx(25.0)  # scenario override wins over [base]
    # List-valued parameters (energy class ranges) are kept as native lists.
    assert parameters["energy_class_ranges"] == [[0, 1], [1, 2]]


def test_read_parameter_specific_data_keeps_mixed_value_types(tmp_path: Path) -> None:
    path = tmp_path / "parameters_building_type.csv"
    path.write_text(
        "building_type_int,building_type,Rc_wall_m2K_W,energy_labels_included_residential,scenario\n"
        '1,residential_lowrise,2.0,"[7, 6]",SQ\n'
        '1,residential_lowrise,9.9,"[7, 6]",2030\n',
    )

    records = read_parameter_specific_data(path, "SQ")

    assert len(records) == 1  # only the SQ row is kept
    assert records[0]["building_type_int"] == 1  # first column -> int
    assert records[0]["building_type"] == "residential_lowrise"  # second column -> str
    assert records[0]["Rc_wall_m2K_W"] == pytest.approx(2.0)  # numeric column -> float
    assert records[0]["energy_labels_included_residential"] == [7, 6]  # label list -> JSON


@pytest.mark.parametrize(
    ("end_use", "height_m", "construction_year", "expected_type"),
    [
        ("residential", 30.0, 2010, 1),  # residential, high-rise, new -> 000
        ("residential", 10.0, 1950, 4),  # residential, low-rise, old  -> 011
        ("office", 30.0, 2010, 5),  # office, high-rise, new        -> 100
        ("office", 10.0, 1950, 8),  # office, low-rise, old         -> 111
    ],
)
def test_determine_building_type(
    end_use: str,
    height_m: float,
    construction_year: int,
    expected_type: int,
    global_parameters: dict,
) -> None:
    building = pd.Series({"end_use": end_use, "height_m": height_m, "construction_year": construction_year})
    assert determine_building_type(building, global_parameters) == expected_type


def test_calculate_building_population_office_uses_floor_area(global_parameters: dict) -> None:
    building = pd.Series({"end_use": "office", "floor_area_total_m2": 1000.0, "number_of_residences": 0})
    assert calculate_building_population(building, global_parameters) == pytest.approx(1000.0 * 0.1)


def test_calculate_building_population_residential_uses_households(global_parameters: dict) -> None:
    building = pd.Series({"end_use": "residential", "floor_area_total_m2": 1000.0, "number_of_residences": 5})
    assert calculate_building_population(building, global_parameters) == pytest.approx(5 * 2.2)


def test_assign_building_type_parameters_selects_row_by_type() -> None:
    building_parameters = [{"Rc_wall_m2K_W": 1.0}, {"Rc_wall_m2K_W": 2.0}, {"Rc_wall_m2K_W": 3.0}]
    building = pd.Series({"building_type_int": 2})

    # The list is 1-indexed by building type, so type 2 selects the second record.
    assert assign_building_type_parameters(building, building_parameters)["Rc_wall_m2K_W"] == 2.0


def test_assign_energy_class_parameters_selects_row_by_class() -> None:
    energy_class_parameters = [{"U_window_W_m2K": 5.0}, {"U_window_W_m2K": 1.2}]
    building = pd.Series({"energy_class_int": 1})

    assert assign_energy_class_parameters(building, energy_class_parameters)["U_window_W_m2K"] == 5.0


def test_determine_energy_label_to_class_mappings() -> None:
    energy_class_parameters = [
        {"energy_class_int": 1, "energy_labels_included_residential": [7, 6], "energy_labels_included_office": [7]},
        {"energy_class_int": 2, "energy_labels_included_residential": [5], "energy_labels_included_office": [6, 5]},
    ]
    residential, office = determine_energy_label_to_class_mappings(energy_class_parameters)

    assert residential == {7: 1, 6: 1, 5: 2}
    assert office == {7: 1, 6: 2, 5: 2}


def test_scale_results_with_building_stock_only_scales_growth_types(global_parameters: dict) -> None:
    buildings_agg = pd.DataFrame(
        {
            "building_type_int": [1, 2, 3, 6, 8],
            "E_cooling_kWh": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
    )
    scaled = scale_results_with_building_stock(buildings_agg, global_parameters, ["E_cooling_kWh"])

    # Residential growth types (1, 3) scale by 1.10, office growth types (6, 8) by 1.20, the rest are untouched.
    assert scaled.set_index("building_type_int")["E_cooling_kWh"].to_dict() == pytest.approx(
        {1: 110.0, 2: 100.0, 3: 110.0, 6: 120.0, 8: 120.0},
    )
