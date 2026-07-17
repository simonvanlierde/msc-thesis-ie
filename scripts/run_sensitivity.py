"""Run the curated sensitivity-analysis set and emit SM figures + a summary table.

Mirrors scripts/run_cooling_mix_sensitivity.py. The full ~20-SA notebook section is
replaced by the 15 SAs the paper draft actually leans on (see the design spec). Runs
on the ``sample`` subset at coarse ``--steps`` resolution, off the notebook critical path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: the SA plot helpers call plt.show(); Agg makes that a no-op

import pandas as pd

from cdm import figures_sensitivity
from cdm.constants import SCENARIOS
from cdm.figures_sensitivity import post_process_SA_results
from cdm.parameters import add_parameters_to_buildings
from cdm.readers import read_buildings, read_global_parameters, read_parameter_specific_data
from cdm.sensitivity_analysis import (
    run_SA_for_total_market_penetration,
    run_SA_for_variable_in_cooling_technology_parameters,
    run_SA_for_variable_in_global_parameters,
)

PARAMETER_DIR = Path("data/input/parameters")
PARAMETERS_TOML = PARAMETER_DIR / "parameters.toml"


@dataclass(frozen=True)
class SASpec:
    """One sensitivity analysis: which variable, which runner, and the swept range."""

    variable_name: str
    kind: str  # "global" | "cooling_tech" | "market_penetration"
    start: float
    end: float
    print_name: str = ""
    unit: str = ""


# The 15 SAs the paper draft leans on. Ranges copied from the corresponding main.ipynb cells.
SA_SPECS: list[SASpec] = [
    SASpec("carbon_intensity_electric_grid_kgCO2eq_kWh", "global", 0, 1, "grid carbon intensity", "kgCO2eq/kWh"),
    SASpec("gwp_refrigerant_kgCO2eq_kg", "global", 0, 20000, "refrigerant GWP", "kgCO2eq/kg"),
    SASpec("carbon_intensity_production_kgCO2eq_kg", "global", 1, 10, "production carbon intensity", "kgCO2eq/kg"),
    # range must bracket the 0.3 kgCO2eq/kg reference: the elasticity is read at the sweep row
    # closest to the reference, and the first row of a sweep is always NaN (pct_change)
    SASpec("carbon_intensity_EoL_kgCO2eq_kg", "global", 0, 3, "end-of-life carbon intensity", "kgCO2eq/kg"),
    SASpec("people_density_office", "global", 0.04, 0.2, "office people density", "people/m2"),
    SASpec("int_heat_gain_light_W_m2", "global", 0.5, 25, "lighting internal heat gain", "W/m2"),
    SASpec("T_thresh_C", "global", 15, 30, "cooling threshold temperature", "degC"),
    SASpec("UHI_effect_day_C", "global", 0, 15, "daytime UHI effect", "degC"),
    SASpec("peak_cooling_percentile_cap", "global", 80, 99.9, "peak-cooling percentile cap", "percentile"),
    SASpec("delta_T_summer_C", "global", -2, 8, "summer temperature shift", "degC"),
    SASpec("SEER", "cooling_tech", 0.5, 3, "SEER", "x reference"),
    SASpec("refrigerant_leakage_rate_relative", "cooling_tech", 0.05, 2, "refrigerant leakage rate", "x reference"),
    SASpec("material_density_kg_kW", "cooling_tech", 0.2, 4, "material intensity", "x reference"),
    SASpec("average_lifetime_yr", "cooling_tech", 0.2, 4, "equipment lifetime", "x reference"),
    SASpec("market_penetration", "market_penetration", 0, 5.86, "total market penetration", "x reference"),
]


def reference_values_for(
    spec: SASpec,
    global_param_dict: dict[str, dict],
    cooling_tech_param_dict: dict[str, list[dict]] | None = None,
    building_type_param_dict: dict[str, list[dict]] | None = None,
    building_type_prevalence: pd.Series | None = None,
) -> dict[str, float]:
    """The per-scenario reference value of the swept variable, in the runner's index units.

    The reference must match the axis the runner indexes on, so ``post_process_SA_results``
    normalises at the right point:
    - global: the scenario's own parameter value;
    - cooling_tech: the mean of the variable across cooling technologies (the runner indexes on
      ``mean * multiplier``, so the reference is the mean, i.e. the multiplier-1 point);
    - market_penetration: the weighted total market-penetration rate in percent (the runner
      re-indexes its results to the measured total MPR * 100).
    """
    if spec.kind == "global":
        return {s: global_param_dict[s][spec.variable_name] for s in global_param_dict}
    if spec.kind == "cooling_tech":
        if cooling_tech_param_dict is None:
            msg = "cooling_tech_param_dict is required for cooling_tech specs"
            raise ValueError(msg)
        return {s: pd.DataFrame(cooling_tech_param_dict[s])[spec.variable_name].mean() for s in global_param_dict}
    # market_penetration
    if building_type_param_dict is None or building_type_prevalence is None:
        msg = "building_type_param_dict and building_type_prevalence are required for market_penetration specs"
        raise ValueError(msg)
    refs = {}
    for s in global_param_dict:
        df = pd.DataFrame(building_type_param_dict[s]).set_index("building_type")
        share_cols = [c for c in df.columns if c.startswith("cooling_technology_share")]
        total_mpr = df[share_cols].sum(axis=1)
        refs[s] = float(total_mpr.mul(building_type_prevalence).sum() * 100)
    return refs


def build_static_parameters(
    buildings: pd.DataFrame,
    raw_weather_data: pd.DataFrame,
    params: dict,
    solar: str,
    presence: str,
) -> dict:
    """Assemble the static_parameters dict the cdm.sensitivity_analysis runners expect."""
    return {
        "buildings": buildings,
        "raw_weather_data": raw_weather_data,
        "global_parameters": params["global"],
        "building_type_parameters": params["building_type"],
        "energy_class_parameters": params["energy_class"],
        "cooling_technology_parameters": params["cooling_technology"],
        "multi_directional_solar_radiation_fractions_path": solar,
        "presence_load_factors_path": presence,
    }


def _run_sa(spec: SASpec, static_parameters: dict, steps: int, building_type_prevalence: pd.Series) -> pd.DataFrame:
    """Dispatch one SA to the runner matching its kind."""
    if spec.kind == "global":
        return run_SA_for_variable_in_global_parameters(
            variable_name=spec.variable_name,
            variable_start=spec.start,
            variable_end=spec.end,
            global_parameters=static_parameters["global_parameters"],
            static_parameters=static_parameters,
            calculation_steps=steps,
        )
    if spec.kind == "cooling_tech":
        return run_SA_for_variable_in_cooling_technology_parameters(
            variable_name=spec.variable_name,
            multiplier_start=spec.start,
            multiplier_end=spec.end,
            cooling_technology_parameters=static_parameters["cooling_technology_parameters"],
            static_parameters=static_parameters,
            calculation_steps=steps,
        )
    # market_penetration
    return run_SA_for_total_market_penetration(
        multiplier_start=spec.start,
        multiplier_end=spec.end,
        building_type_parameters=static_parameters["building_type_parameters"],
        building_type_prevalence=building_type_prevalence,
        static_parameters=static_parameters,
        calculation_steps=steps,
    )


def main() -> None:
    """Run every SA in SA_SPECS, save a figure per SA, and write the elasticity summary table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="SQ")
    parser.add_argument("--buildings", required=True)
    parser.add_argument("--buildings-layer", required=True)
    parser.add_argument("--solar-fractions", required=True)
    parser.add_argument("--presence-load-factors", required=True)
    parser.add_argument("--weather-csv", required=True)
    parser.add_argument("--image-dir", required=True, help="Directory for the SA figures; declared as a rule output.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=21)
    args = parser.parse_args()

    building_type_csv = PARAMETER_DIR / "parameters_building_type.csv"
    cooling_tech_csv = PARAMETER_DIR / "parameters_cooling_technology.csv"
    global_param_dict = {s: read_global_parameters(PARAMETERS_TOML, s) for s in SCENARIOS}
    cooling_tech_param_dict = {s: read_parameter_specific_data(cooling_tech_csv, s) for s in SCENARIOS}
    building_type_param_dict = {s: read_parameter_specific_data(building_type_csv, s) for s in SCENARIOS}
    params = {
        "global": global_param_dict[args.scenario],
        "building_type": building_type_param_dict[args.scenario],
        "energy_class": read_parameter_specific_data(PARAMETER_DIR / "parameters_energy_class.csv", args.scenario),
        "cooling_technology": cooling_tech_param_dict[args.scenario],
    }
    buildings = read_buildings(Path(args.buildings), args.buildings_layer)
    # read_buildings returns the raw stock; the `building_type` archetype is added by
    # add_parameters_to_buildings. Parameterise a copy once to get the stock prevalence
    # the market-penetration SA and its reference need.
    buildings_with_type = add_parameters_to_buildings(
        buildings.copy(),
        params["global"],
        params["building_type"],
        params["energy_class"],
    )
    building_type_prevalence = buildings_with_type["building_type"].value_counts(normalize=True)
    raw_weather_data = pd.read_csv(args.weather_csv, parse_dates=["date"])
    static_parameters = build_static_parameters(
        buildings,
        raw_weather_data,
        params,
        args.solar_fractions,
        args.presence_load_factors,
    )

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    # The plot helpers in figures_sensitivity save to the module-level SA_IMAGE_DIR (three figures
    # per SA). Point it at our declared output dir instead of the module default (data/output/images/SA).
    figures_sensitivity.SA_IMAGE_DIR = str(image_dir)

    rows = []
    for spec in SA_SPECS:
        sa_results = _run_sa(spec, static_parameters, args.steps, building_type_prevalence)
        refs = reference_values_for(
            spec,
            global_param_dict,
            cooling_tech_param_dict,
            building_type_param_dict,
            building_type_prevalence,
        )
        elasticities = post_process_SA_results(
            SA_results=sa_results,
            reference_values=refs,
            variable_name_print=spec.print_name,
            variable_unit_print=spec.unit,
        )
        # post_process_SA_results returns a Series (elasticity per impact at the reference value).
        row = {"variable": spec.variable_name}
        row.update({f"elasticity_{label}": float(elasticities[label]) for label in elasticities.index})
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
