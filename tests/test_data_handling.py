"""Unit tests for the deterministic data-handling helpers.

The heavy GeoPackage readers depend on the external Zenodo datasets and are not
exercised here; these tests cover the pure mapping and scaling logic.
"""

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from cdm.aggregation import aggregate_results, scale_results_with_building_stock
from cdm.constants import REQUIRED_GLOBAL_PARAMETERS
from cdm.parameters import (
    add_cooling_technology_data_to_buildings,
    add_energy_class_data_to_buildings,
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


_ENERGY_CLASS_PARAMS = [
    {"energy_class_int": 1, "energy_labels_included_residential": [7], "energy_labels_included_office": [7], "R": 1.0},
]


def test_add_energy_class_data_raises_on_unmapped_label() -> None:
    """A label outside every energy_labels_included_* list must fail loud, not vanish from totals."""
    buildings = pd.DataFrame({"energy_label_int": [7.0, 3.0], "end_use": ["residential", "residential"]})

    with pytest.raises(ValueError, match="not covered"):
        add_energy_class_data_to_buildings(buildings, _ENERGY_CLASS_PARAMS)


def test_add_energy_class_data_raises_on_null_label() -> None:
    """A null energy_label_int would crash the int cast; fail with a clear message instead."""
    buildings = pd.DataFrame({"energy_label_int": [7.0, None], "end_use": ["residential", "residential"]})

    with pytest.raises(ValueError, match="null energy_label_int"):
        add_energy_class_data_to_buildings(buildings, _ENERGY_CLASS_PARAMS)


def test_add_energy_class_data_assigns_known_labels() -> None:
    """The happy path still classifies a covered label (guards do not block valid data)."""
    buildings = pd.DataFrame({"energy_label_int": [7.0], "end_use": ["residential"]})

    result = add_energy_class_data_to_buildings(buildings, _ENERGY_CLASS_PARAMS)

    assert result["energy_class_int"].tolist() == [1]


_COOLING_TECHS = pd.DataFrame(
    {
        "cooling_technology": ["A", "B"],
        "cooling_technology_int": [1, 2],
        "SEER": [3.0, 4.0],  # dropped before weighting
        "refrigerant_density_kg_kW": [0.1, 0.2],  # dropped before weighting
        "average_lifetime_yr": [10.0, 20.0],
        "some_param": [100.0, 200.0],
    },
)


def test_add_cooling_technology_data_weights_params_by_share() -> None:
    """Each avg_<param> is the building's share-weighted average of that technology parameter."""
    buildings = pd.DataFrame(
        {"cooling_technology_share_A": [0.25, 1.0], "cooling_technology_share_B": [0.75, 0.0]},
    )

    result = add_cooling_technology_data_to_buildings(buildings, _COOLING_TECHS)

    assert result["avg_some_param"].iloc[0] == pytest.approx(0.25 * 100 + 0.75 * 200)
    assert result["avg_some_param"].iloc[1] == pytest.approx(100.0)
    # average_lifetime_yr is renamed to the model's column name.
    assert result["avg_lifetime_cooling_technology_yr"].iloc[0] == pytest.approx(0.25 * 10 + 0.75 * 20)


def test_add_cooling_technology_data_rejects_unknown_tech() -> None:
    """A share column for a technology missing from the parameter table fails loud, not with KeyError."""
    buildings = pd.DataFrame({"cooling_technology_share_A": [1.0], "cooling_technology_share_Z": [0.0]})

    with pytest.raises(ValueError, match="no row in the cooling-technology parameters"):
        add_cooling_technology_data_to_buildings(buildings, _COOLING_TECHS)


def test_determine_energy_label_to_class_mappings() -> None:
    energy_class_parameters = [
        {"energy_class_int": 1, "energy_labels_included_residential": [7, 6], "energy_labels_included_office": [7]},
        {"energy_class_int": 2, "energy_labels_included_residential": [5], "energy_labels_included_office": [6, 5]},
    ]
    residential, office = determine_energy_label_to_class_mappings(energy_class_parameters)

    assert residential == {7: 1, 6: 1, 5: 2}
    assert office == {7: 1, 6: 2, 5: 2}


def _aggregatable_stock(building_type_ints: list[int], energy_class_ints: list[int]) -> pd.DataFrame:
    """A DataFrame carrying every column aggregate_results reads, one row per building type/class."""
    n = len(building_type_ints)
    # Each summed physical quantity is 10.0 per building; every weighted-mean input is 1.0.
    sum_cols = [
        "floor_area_ground_m2",
        "volume_m3",
        "floor_area_total_m2",
        "number_of_residences",
        "population",
        "E_cooling_kWh",
        "E_cooling_capped_at_98th_percentile_kWh",
        "P_cooling_peak_kW",
        "P_cooling_peak_98th_percentile_kW",
        "electricity_use_kWh",
        "GHG_emissions_electricity_kgCO2eq",
        "GHG_emissions_refrigerant_leaks_kgCO2eq",
        "GHG_emissions_production_phase_kgCO2eq",
        "GHG_emissions_EoL_phase_kgCO2eq",
        "mass_cooling_equipment_kg",
        "ADP_kgSbeq",
        "CSI_kgSieq",
        "GHG_emissions_total_kgCO2eq",
    ]
    mean_cols = [
        "construction_year",
        "height_m",
        "energy_label_int",
        "E_cooling_capped_at_98th_percentile_Wh_m2",
        "P_cooling_peak_98th_percentile_W_m2",
        "electricity_use_intensity_kWh_m2",
        "material_use_intensity_kg_m2",
        "GHG_emissions_intensity_kgCO2eq_m2",
        "avg_SEER_inv",
        "total_MPR",
        "avg_lifetime_cooling_technology_yr",
    ]
    data = {
        "id_BAG": [f"b{i}" for i in range(n)],
        "building_type_int": building_type_ints,
        "building_type": [f"type_{t}" for t in building_type_ints],
        "energy_class_int": energy_class_ints,
        "energy_class": [f"class_{c}" for c in energy_class_ints],
        **{col: [10.0] * n for col in sum_cols},
        **{col: [1.0] * n for col in mean_cols},
    }
    return pd.DataFrame(data)


def test_aggregate_scales_a_weighted_sample_back_to_stock_totals(global_parameters: dict) -> None:
    """A sample carrying stock_weight must aggregate to the same summed totals as the full stock."""
    # Full stock: 30 identical buildings in one (type, class) stratum.
    full = _aggregatable_stock([1] * 30, [3] * 30)
    full_agg = aggregate_results(full, global_parameters, scale_with_building_stock=False)

    # Sample: one building standing in for all 30 (weight 30).
    sample = _aggregatable_stock([1], [3])
    sample["stock_weight"] = [30.0]
    sample_agg = aggregate_results(sample, global_parameters, scale_with_building_stock=False)

    assert sample_agg["E_cooling_kWh"].iloc[0] == pytest.approx(full_agg["E_cooling_kWh"].iloc[0])
    # The reported building count is the represented stock count, not the single sampled row.
    assert sample_agg["id_BAG"].iloc[0] == pytest.approx(30.0)


def test_aggregate_without_stock_weight_reports_a_plain_count(global_parameters: dict) -> None:
    """The full-stock path (no stock_weight) is unchanged: id_BAG is an integer building count."""
    full = _aggregatable_stock([1] * 5, [3] * 5)

    full_agg = aggregate_results(full, global_parameters, scale_with_building_stock=False)

    assert full_agg["id_BAG"].iloc[0] == 5
    assert full_agg["E_cooling_kWh"].iloc[0] == pytest.approx(50.0)


def test_aggregate_weighted_mean_survives_a_zero_floor_area_group(global_parameters: dict) -> None:
    """A group with zero total floor area must yield NaN intensities, not crash the whole aggregation."""
    stock = _aggregatable_stock([1, 2], [3, 3])
    stock["floor_area_total_m2"] = [0.0, 100.0]  # first group has no floor area to weight by

    agg = aggregate_results(stock, global_parameters, scale_with_building_stock=False)

    by_type = agg.set_index("building_type_int")
    assert np.isnan(by_type.loc[1, "electricity_use_intensity_kWh_m2"])  # degenerate group -> NaN
    assert not np.isnan(by_type.loc[2, "electricity_use_intensity_kWh_m2"])  # healthy group unaffected


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
