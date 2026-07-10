"""Snakemake wrappers for the thesis cooling-demand notebook stages.

The code below follows the executable cells in ``main.ipynb`` and delegates the
scientific calculations to the existing ``functions`` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from cdm.aggregation import aggregate_results
from cdm.constants import PARAMETER_COLUMNS_TO_DROP_AFTER_CALCULATIONS
from cdm.environmental import (
    calculate_environmental_impacts_from_cooling_demand,
    calculate_environmental_parameters_for_cooling_technologies,
)
from cdm.parameters import add_cooling_technology_data_to_buildings, add_parameters_to_buildings
from cdm.readers import read_buildings, read_global_parameters, read_parameter_specific_data
from cdm.thermodynamic import calc_cooling_demand_metrics_for_df
from cdm.time_series import create_time_series

PARAMETER_DIR = Path("data/input/parameters")
PARAMETERS_TOML = PARAMETER_DIR / "parameters.toml"


def _load_parameters(scenario: str) -> dict:
    """Load the scenario's parameters from the consolidated parameters.toml + per-group CSVs."""
    return {
        "global": read_global_parameters(PARAMETERS_TOML, scenario),
        "building_type": read_parameter_specific_data(PARAMETER_DIR / "parameters_building_type.csv", scenario),
        "energy_class": read_parameter_specific_data(PARAMETER_DIR / "parameters_energy_class.csv", scenario),
        "cooling_technology": read_parameter_specific_data(
            PARAMETER_DIR / "parameters_cooling_technology.csv",
            scenario,
        ),
    }


def _drop_array_columns(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # GeoPackage cannot serialize array-valued cells (the hourly Q_* series). Only object-dtype
    # columns can hold arrays (the geometry column has its own dtype and is left alone); scan the
    # whole column, not just iloc[0], so a column whose first cell is None but whose later cells
    # are arrays is still dropped rather than crashing the write.
    array_columns = [
        column
        for column in buildings.columns
        if buildings[column].dtype == object
        and buildings[column].map(lambda value: isinstance(value, np.ndarray)).any()
    ]
    return buildings.drop(columns=array_columns)


def run_cooling_demand(args: argparse.Namespace) -> None:
    """Run the notebook's parameter assignment, time series and CDM steps."""
    parameters = _load_parameters(args.scenario)
    global_parameters = parameters["global"]
    building_type_parameters = parameters["building_type"]
    energy_class_parameters = parameters["energy_class"]
    cooling_technology_parameters = parameters["cooling_technology"]

    buildings = read_buildings(Path(args.buildings), args.buildings_layer)
    raw_weather_data = pd.read_csv(args.weather_csv, parse_dates=["date"])

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
    global_parameters = read_global_parameters(PARAMETERS_TOML, args.scenario)

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

    buildings_to_file = _drop_array_columns(buildings).drop(
        columns=list(PARAMETER_COLUMNS_TO_DROP_AFTER_CALCULATIONS),
        errors="ignore",
    )

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
    cooling.add_argument("--weather-csv", required=True)
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
