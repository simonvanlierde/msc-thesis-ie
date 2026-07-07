"""Regenerate the cooling-technology-mix elasticities table.

This reproduces the sensitivity-analysis loop in ``main.ipynb`` (the cells that
build ``cooling_mix_elasticities_table``): for every ordered pair of the six
cooling technologies, it sweeps a 0->100% mix share between the two, runs the
full cooling-demand + LCA model at each step, and records the mean elasticity of
electricity demand, carbon emissions and material demand with respect to the mix.

Cost note: with the defaults this is 30 tech pairs x ``calculation_steps`` full
model runs over the whole building stock, so it is deliberately heavy.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # post_process_SA_cooling_tech_mix always writes figures; run headless before pyplot loads

import pandas as pd

from cdm.figures_sensitivity import SA_IMAGE_DIR, post_process_SA_cooling_tech_mix
from cdm.readers import (
    read_buildings,
    read_global_parameters,
    read_parameter_specific_data,
)
from cdm.sensitivity_analysis import run_SA_for_cooling_technology_mix

COOLING_TECHS = ["ASHP", "GSHP", "WSHP", "chiller", "AC_split", "AC_mobile"]
PARAMETER_DIR = Path("data/input/parameters")
PARAMETERS_TOML = PARAMETER_DIR / "parameters.toml"


def _scenario_parameters(scenario: str) -> dict:
    return {
        "global": read_global_parameters(PARAMETERS_TOML, scenario),
        "building_type": read_parameter_specific_data(PARAMETER_DIR / "parameters_building_type.csv", scenario),
        "energy_class": read_parameter_specific_data(PARAMETER_DIR / "parameters_energy_class.csv", scenario),
        "cooling_technology": read_parameter_specific_data(
            PARAMETER_DIR / "parameters_cooling_technology.csv",
            scenario,
        ),
    }


def build_elasticities_table(
    building_type_parameters: list[dict],
    static_parameters: dict,
    calculation_steps: int,
) -> pd.DataFrame:
    """Run the mix sweep for every ordered technology pair and collect mean elasticities."""
    Path(SA_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    rows = []
    for cooling_one, cooling_two in permutations(COOLING_TECHS, 2):
        sa_results = run_SA_for_cooling_technology_mix(
            cooling_tech_one=cooling_one,
            cooling_tech_two=cooling_two,
            building_type_parameters=building_type_parameters,
            static_parameters=static_parameters,
            calculation_steps=calculation_steps,
        )
        # Returns mean elasticity per impact, ordered [electricity, GHG, material]
        elasticities = post_process_SA_cooling_tech_mix(
            SA_results=sa_results,
            cooling_tech_one=cooling_one,
            cooling_tech_two=cooling_two,
            show_plots=False,
        )
        rows.append(
            {
                "cooling_tech_one": cooling_one,
                "cooling_tech_two": cooling_two,
                "elasticity_electricity_demand": elasticities.iloc[0, 0],
                "elasticity_carbon_emissions": elasticities.iloc[1, 0],
                "elasticity_material_demand": elasticities.iloc[2, 0],
            },
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Regenerate the cooling-technology-mix elasticities table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="SQ")
    parser.add_argument("--buildings", required=True)
    parser.add_argument("--buildings-layer", required=True)
    parser.add_argument("--solar-fractions", required=True)
    parser.add_argument("--presence-load-factors", required=True)
    parser.add_argument("--weather-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calculation-steps", type=int, default=20)
    args = parser.parse_args()

    parameters = _scenario_parameters(args.scenario)
    buildings = read_buildings(Path(args.buildings), args.buildings_layer)
    raw_weather_data = pd.read_csv(args.weather_csv, parse_dates=["date"])

    static_parameters = {
        "buildings": buildings,
        "raw_weather_data": raw_weather_data,
        "global_parameters": parameters["global"],
        "energy_class_parameters": parameters["energy_class"],
        "cooling_technology_parameters": parameters["cooling_technology"],
        "multi_directional_solar_radiation_fractions_path": args.solar_fractions,
        "presence_load_factors_path": args.presence_load_factors,
    }
    table = build_elasticities_table(parameters["building_type"], static_parameters, args.calculation_steps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


if __name__ == "__main__":
    main()
