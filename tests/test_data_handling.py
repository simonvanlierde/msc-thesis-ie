"""Unit tests for the deterministic data-handling helpers.

The heavy GeoPackage readers depend on the external Zenodo datasets and are not
exercised here; these tests cover the pure mapping and scaling logic.
"""

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from cdm.aggregation import scale_results_with_building_stock
from cdm.constants import REQUIRED_GLOBAL_PARAMETERS
from cdm.parameters import (
    assign_parameters_by_class,
    calculate_building_population,
    determine_building_type,
    determine_energy_label_to_class_mappings,
)
from cdm.readers import read_buildings, read_global_parameters, read_parameter_specific_data

if TYPE_CHECKING:
    from pathlib import Path


def test_read_buildings_translates_end_use_without_touching_other_columns(tmp_path: Path) -> None:
    """The Dutch->English translation must be scoped to end_use.

    A frame-wide .replace() rewrites "woonfunctie"/"kantoorfunctie" in *every* column, so a
    sibling column that happens to hold one of those strings is silently corrupted.
    """
    buildings = gpd.GeoDataFrame(
        {
            "end_use": ["woonfunctie", "kantoorfunctie"],
            "status": ["Pand in gebruik", "Pand in gebruik"],
            "energy_label": ["A", "B"],
            # A pass-through column that survives every filter and holds the Dutch term.
            "source_note": ["woonfunctie", "kantoorfunctie"],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )
    path = tmp_path / "buildings.gpkg"
    buildings.to_file(path, layer="BAG_buildings", driver="GPKG")

    result = read_buildings(path, "BAG_buildings")

    assert sorted(result["end_use"]) == ["office", "residential"]
    assert sorted(result["source_note"]) == ["kantoorfunctie", "woonfunctie"]


def test_read_buildings_filters_rows_at_read_time(tmp_path: Path) -> None:
    """Only in-use residential/office buildings with a known energy label are modelled.

    The predicate is pushed into GDAL as an attribute filter, so this pins the SQL against
    the pandas semantics it replaced -- in particular that `energy_label IS NOT NULL` drops
    the same rows as the `.dropna()` it stands in for.
    """
    buildings = gpd.GeoDataFrame(
        {
            "end_use": ["woonfunctie", "kantoorfunctie", "winkelfunctie", "woonfunctie", "woonfunctie"],
            "status": ["Pand in gebruik", "Pand in gebruik", "Pand in gebruik", "Bouw gestart", "Pand in gebruik"],
            # The last row's label is NULL in the GeoPackage: it must not survive the read.
            "energy_label": ["A", "B", "C", "D", None],
        },
        geometry=[Point(i, i) for i in range(5)],
        crs="EPSG:28992",
    )
    path = tmp_path / "buildings.gpkg"
    buildings.to_file(path, layer="BAG_buildings", driver="GPKG")

    result = read_buildings(path, "BAG_buildings")

    # Kept: the residential and the office row. Dropped: retail, not-yet-built, no label.
    assert sorted(result["end_use"]) == ["office", "residential"]
    assert sorted(result["energy_label"]) == ["A", "B"]


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
    buildings = pd.DataFrame([{"end_use": end_use, "height_m": height_m, "construction_year": construction_year}])
    assert determine_building_type(buildings, global_parameters).iloc[0] == expected_type


def test_calculate_building_population_office_uses_floor_area(global_parameters: dict) -> None:
    buildings = pd.DataFrame([{"end_use": "office", "floor_area_total_m2": 1000.0, "number_of_residences": 0}])
    assert calculate_building_population(buildings, global_parameters)[0] == pytest.approx(1000.0 * 0.1)


def test_calculate_building_population_residential_uses_households(global_parameters: dict) -> None:
    buildings = pd.DataFrame([{"end_use": "residential", "floor_area_total_m2": 1000.0, "number_of_residences": 5}])
    assert calculate_building_population(buildings, global_parameters)[0] == pytest.approx(5 * 2.2)


def test_calculate_building_population_mixes_end_uses_in_one_call(global_parameters: dict) -> None:
    buildings = pd.DataFrame(
        {
            "end_use": ["office", "residential"],
            "floor_area_total_m2": [1000.0, 1000.0],
            "number_of_residences": [0, 5],
        },
    )
    assert calculate_building_population(buildings, global_parameters) == pytest.approx([100.0, 11.0])


def test_assign_parameters_by_class_selects_the_record_matching_each_building() -> None:
    building_parameters = [
        {"building_type_int": 1, "Rc_wall_m2K_W": 1.0},
        {"building_type_int": 2, "Rc_wall_m2K_W": 2.0},
        {"building_type_int": 3, "Rc_wall_m2K_W": 3.0},
    ]
    building_type_ints = pd.Series([2, 1], index=[10, 11])

    assigned = assign_parameters_by_class(building_type_ints, building_parameters, "building_type_int")

    # The class column becomes the lookup key, so only the parameters it selects come back.
    assert list(assigned.columns) == ["Rc_wall_m2K_W"]
    assert assigned["Rc_wall_m2K_W"].tolist() == [2.0, 1.0]
    # The result stays aligned with the buildings it was looked up for.
    assert assigned.index.tolist() == [10, 11]


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
