"""Snakemake wrappers for the thesis cooling-demand notebook stages.

The code below follows the executable cells in ``main.ipynb`` and delegates the
scientific calculations to the existing ``functions`` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np

from functions.data_handling import (
    add_cooling_technology_data_to_buildings,
    add_parameters_to_buildings,
    aggregate_results,
    read_buildings,
    read_global_parameters,
    read_parameter_specific_data,
)
from functions.environmental import (
    calculate_environmental_impacts_from_cooling_demand,
    calculate_environmental_parameters_for_cooling_technologies,
)
from functions.thermodynamic import calc_cooling_demand_metrics_for_df
from functions.time_series import create_time_series, get_raw_weather_data

PARAMETER_COLUMNS_TO_DROP_AFTER_CALCULATIONS = [
    "avg_ADP_kgSbeq_kW",
    "avg_CSI_kgSieq_kW",
    "avg_GHG_emissions_electricity_kgCO2eq_kWh_cooling",
    "avg_GHG_emissions_EoL_phase_kgCO2eq_kW",
    "avg_GHG_emissions_production_phase_kgCO2eq_kW",
    "avg_GHG_emissions_refrigerant_leaks_kgCO2eq_kW",
    "avg_material_density_kg_kW",
    "avg_refrigerant_leakage_kg_kW",
    "avg_refrigerant_leakage_rate_relative",
    "energy_class_int",
    "energy_labels_included_residential",
    "energy_labels_included_office",
    "cooling_technology_share_ASHP",
    "cooling_technology_share_GSHP",
    "cooling_technology_share_WSHP",
    "cooling_technology_share_chiller",
    "cooling_technology_share_AC_split",
    "cooling_technology_share_AC_mobile",
    "f_wall",
    "f_window",
    "facade_area_per_orientation_m2",
    "g_window",
    "ground_elevation_m",
    "infiltration_ACH",
    "int_heat_gain_appliances_W_m2",
    "MBR_length_m",
    "MBR_width_m",
    "pressure_drop_Pa",
    "Rc_floor_m2K_W",
    "Rc_roof_m2K_W",
    "Rc_wall_m2K_W",
    "roof_elevation_m",
    "status",
    "U_window_W_m2K",
    "ventilation_rate_pp_m3_h",
    "wall_area_total_m2",
    "window_area_per_orientation_m2",
    "window_area_total_m2",
]


def _scenario_paths(scenario: str) -> dict[str, Path]:
    scenario_folder = Path("data/input/parameters") / f"parameters_{scenario}"
    return {
        "global": scenario_folder / "parameters_global.csv",
        "building_type": scenario_folder / "parameters_building_type.csv",
        "energy_class": scenario_folder / "parameters_energy_class.csv",
        "cooling_technology": scenario_folder / "parameters_cooling_technology.csv",
    }


def _drop_array_columns(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    array_columns = [
        column
        for column in buildings.columns
        if len(buildings) > 0 and isinstance(buildings[column].iloc[0], np.ndarray)
    ]
    return buildings.drop(columns=array_columns)


def run_cooling_demand(args: argparse.Namespace) -> None:
    """Run the notebook's parameter assignment, time series and CDM steps."""
    paths = _scenario_paths(args.scenario)
    global_parameters = read_global_parameters(paths["global"])
    building_type_parameters = read_parameter_specific_data(paths["building_type"])
    energy_class_parameters = read_parameter_specific_data(paths["energy_class"])
    cooling_technology_parameters = read_parameter_specific_data(paths["cooling_technology"])

    buildings = read_buildings(Path(args.buildings), args.buildings_layer)
    raw_weather_data = get_raw_weather_data(global_parameters)

    buildings = add_parameters_to_buildings(
        buildings,
        global_parameters,
        building_type_parameters,
        energy_class_parameters,
    )
    cooling_technologies = calculate_environmental_parameters_for_cooling_technologies(
        cooling_technology_parameters,
        global_parameters,
    )
    buildings = add_cooling_technology_data_to_buildings(buildings, cooling_technologies)
    time_series = create_time_series(
        global_parameters,
        raw_weather_data,
        args.solar_fractions,
        args.presence_load_factors,
    )
    buildings = calc_cooling_demand_metrics_for_df(
        buildings,
        time_series,
        global_parameters,
        args.include_time_series,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _drop_array_columns(buildings).to_file(output_path, layer=args.output_layer, driver="GPKG")


def run_lca(args: argparse.Namespace) -> None:
    """Run the notebook's environmental-impact and aggregation steps."""
    paths = _scenario_paths(args.scenario)
    global_parameters = read_global_parameters(paths["global"])

    buildings = gpd.read_file(args.cooling_demand, layer=args.cooling_demand_layer)
    buildings, _impact_summary = calculate_environmental_impacts_from_cooling_demand(
        buildings,
        global_parameters,
    )
    buildings_agg = aggregate_results(buildings, global_parameters)

    geodata_output_path = Path(args.geodata_output)
    csv_output_path = Path(args.csv_output)
    geodata_output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    buildings_to_file = _drop_array_columns(buildings)
    columns_to_drop = [
        column for column in PARAMETER_COLUMNS_TO_DROP_AFTER_CALCULATIONS if column in buildings_to_file.columns
    ]
    buildings_to_file = buildings_to_file.drop(columns=columns_to_drop)

    buildings_to_file.to_file(geodata_output_path, layer=args.geodata_output_layer, driver="GPKG")
    buildings_agg.to_csv(csv_output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single pipeline stage."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    cooling = subparsers.add_parser("cooling-demand")
    cooling.add_argument("--scenario", required=True)
    cooling.add_argument("--buildings", required=True)
    cooling.add_argument("--buildings-layer", required=True)
    cooling.add_argument("--solar-fractions", required=True)
    cooling.add_argument("--presence-load-factors", required=True)
    cooling.add_argument("--output", required=True)
    cooling.add_argument("--output-layer", required=True)
    cooling.add_argument("--include-time-series", action="store_true")
    cooling.set_defaults(func=run_cooling_demand)

    lca = subparsers.add_parser("lca")
    lca.add_argument("--scenario", required=True)
    lca.add_argument("--cooling-demand", required=True)
    lca.add_argument("--cooling-demand-layer", required=True)
    lca.add_argument("--geodata-output", required=True)
    lca.add_argument("--geodata-output-layer", required=True)
    lca.add_argument("--csv-output", required=True)
    lca.set_defaults(func=run_lca)

    return parser.parse_args()


def main() -> None:
    """Run the requested stage."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
