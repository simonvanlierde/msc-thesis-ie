"""Regenerate the headline numbers quoted in the journal-paper draft.

Reads the scenario results (results/CDM_results_{scenario}_full.csv), writes every
number the manuscript quotes to JSON at full precision, and renders the same values
as a human-readable report on stdout. The JSON is the artifact: rounding happens in
the renderer (and in the prose), never in the stored values, so the file stays usable
for checking the draft against the pipeline.

    python scripts/paper_numbers.py [--results-dir results] [--json-output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCENARIOS = ["SQ", "2030", "2050_L", "2050_M", "2050_H"]

# Float slack for the GHG phase-share self-check.
TOLERANCE = 1e-6

GHG_PHASES = {
    "electricity": "GHG_emissions_electricity_kgCO2eq",
    "refrigerant": "GHG_emissions_refrigerant_leaks_kgCO2eq",
    "production": "GHG_emissions_production_phase_kgCO2eq",
    "EoL": "GHG_emissions_EoL_phase_kgCO2eq",
}

# The manuscript quotes demand on the uncapped basis (990 GWh); the capped variant
# is carried alongside for the peak/equipment-sizing story.
TRAJECTORY_METRICS = {
    "demand": "E_cooling_kWh",
    "electricity": "electricity_use_kWh",
    "GHG": "GHG_emissions_total_kgCO2eq",
    "mass": "mass_cooling_equipment_kg",
}


def load(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the per-scenario results into a dict of DataFrames, keyed by scenario name."""
    frames = {}
    for s in SCENARIOS:
        df = pd.read_csv(results_dir / f"CDM_results_{s}_full.csv")
        df["is_office"] = df["building_type"].str.contains("office", case=False)
        frames[s] = df
    return frames


def pct(x: float) -> str:
    """Format a fraction as a percentage string with no decimal places."""
    return f"{100 * x:.0f}%"


def signed_pct(x: float) -> str:
    """Format a relative change as a signed percentage string with no decimal places."""
    return f"{100 * x:+.0f}%"


def intensity(df: pd.DataFrame) -> float:
    """Cooling energy demand per unit floor area (kWh/m2, uncapped basis)."""
    return float(df["E_cooling_kWh"].sum() / df["floor_area_total_m2"].sum())


def unmet_share(df: pd.DataFrame) -> float:
    """Cooling gap: demand-weighted share of demand not covered by installed equipment (D2)."""
    served = (df["E_cooling_kWh"] * df["total_MPR"]).sum()
    return float(1 - served / df["E_cooling_kWh"].sum())


def collect(frames: dict[str, pd.DataFrame]) -> dict:
    """Every number the manuscript quotes, unrounded. Floats are cast for JSON."""
    sq = frames["SQ"]
    off, res = sq[sq["is_office"]], sq[~sq["is_office"]]
    ghg_total = float(sq["GHG_emissions_total_kgCO2eq"].sum())
    population = float(sq["population"].sum())
    office_mi = float(off["mass_cooling_equipment_kg"].sum() / off["floor_area_total_m2"].sum())
    residential_mi = float(res["mass_cooling_equipment_kg"].sum() / res["floor_area_total_m2"].sum())

    status_quo = {
        "archetype_groups": len(sq),
        "cooling_demand_kWh": float(sq["E_cooling_kWh"].sum()),
        "cooling_demand_capped_kWh": float(sq["E_cooling_capped_at_98th_percentile_kWh"].sum()),
        "peak_power_kW": float(sq["P_cooling_peak_kW"].sum()),
        "peak_power_capped_kW": float(sq["P_cooling_peak_98th_percentile_kW"].sum()),
        "intensity_office_kWh_m2": intensity(off),
        "intensity_residential_kWh_m2": intensity(res),
        "office_share_floor_area": float(off["floor_area_total_m2"].sum() / sq["floor_area_total_m2"].sum()),
        "office_share_demand": float(off["E_cooling_kWh"].sum() / sq["E_cooling_kWh"].sum()),
        "office_share_electricity": float(off["electricity_use_kWh"].sum() / sq["electricity_use_kWh"].sum()),
        "office_share_GHG": float(off["GHG_emissions_total_kgCO2eq"].sum() / ghg_total),
        "unmet_demand_share": unmet_share(sq),
        "GHG_total_kgCO2eq": ghg_total,
        # NB: population is the model's own (residences x household size), which exceeds
        # the municipal population quoted in the manuscript's introduction.
        "population": population,
        "GHG_per_capita_kgCO2eq": ghg_total / population,
        "GHG_phase_shares": {phase: float(sq[col].sum()) / ghg_total for phase, col in GHG_PHASES.items()},
        "equipment_mass_kg": float(sq["mass_cooling_equipment_kg"].sum()),
        "material_intensity_office_kg_m2": office_mi,
        "material_intensity_residential_kg_m2": residential_mi,
        "material_intensity_ratio": office_mi / residential_mi,
        "ADP_kgSbeq": float(sq["ADP_kgSbeq"].sum()),
        "CSI_kgSieq": float(sq["CSI_kgSieq"].sum()),
    }

    scenarios = {}
    for s in SCENARIOS[1:]:
        df = frames[s]
        scenarios[s] = {
            **{f"{name}_change": float(df[col].sum() / sq[col].sum() - 1) for name, col in TRAJECTORY_METRICS.items()},
            "unmet_demand_share": unmet_share(df),
            "office_share_demand": float(df.loc[df["is_office"], "E_cooling_kWh"].sum() / df["E_cooling_kWh"].sum()),
            "intensity_residential_kWh_m2": intensity(df[~df["is_office"]]),
        }

    return {"status_quo": status_quo, "scenarios": scenarios}


def render(data: dict) -> str:
    """The human-readable report, rendered from the collected values."""
    sq = data["status_quo"]
    lines = [
        f"== Status quo (SQ), {sq['archetype_groups']:,} archetype groups ==",
        f"annual cooling demand:        {sq['cooling_demand_kWh'] / 1e6:,.0f} GWh (uncapped)",
        f"                              {sq['cooling_demand_capped_kWh'] / 1e6:,.0f} GWh (98p-capped)",
        f"peak power, uncapped:         {sq['peak_power_kW'] / 1e3:,.0f} MW",
        f"peak power, 98p-capped:       {sq['peak_power_capped_kW'] / 1e3:,.0f} MW",
        f"office / residential intensity: {sq['intensity_office_kWh_m2']:.0f} / "
        f"{sq['intensity_residential_kWh_m2']:.0f} kWh/m2 (uncapped)",
        f"office floor-area share:      {pct(sq['office_share_floor_area'])}",
        f"office demand share:          {pct(sq['office_share_demand'])}",
        f"office electricity share:     {pct(sq['office_share_electricity'])}",
        f"office GHG share:             {pct(sq['office_share_GHG'])}",
        f"cooling gap (unmet demand):   {pct(sq['unmet_demand_share'])}",
        f"GHG total:                    {sq['GHG_total_kgCO2eq'] / 1e6:.1f} kt CO2-eq  "
        f"({sq['GHG_per_capita_kgCO2eq']:.0f} kg/cap over {sq['population']:,.0f} modelled residents)",
    ]
    lines += [f"  GHG share {phase:<12} {100 * share:.1f}%" for phase, share in sq["GHG_phase_shares"].items()]
    lines += [
        f"equipment mass:               {sq['equipment_mass_kg'] / 1e3:,.0f} t",
        f"office/res material intensity: {sq['material_intensity_ratio']:.1f}x  "
        f"({sq['material_intensity_office_kg_m2']:.2f} vs {sq['material_intensity_residential_kg_m2']:.2f} kg/m2)",
        f"ADP:                          {sq['ADP_kgSbeq']:.0f} kg Sb-eq",
        f"CSI:                          {sq['CSI_kgSieq']:.2e} kg Si-eq",
        "",
        "== Scenario trajectory (% vs SQ) ==",
        f"{'metric':<14}" + "".join(f"{s:>10}" for s in SCENARIOS[1:]),
    ]
    rows = data["scenarios"]
    lines += [
        f"{name:<14}" + "".join(f"{signed_pct(rows[s][f'{name}_change']):>10}" for s in SCENARIOS[1:])
        for name in TRAJECTORY_METRICS
    ]
    lines += [
        f"{'unmet demand':<14}" + "".join(f"{pct(rows[s]['unmet_demand_share']):>10}" for s in SCENARIOS[1:]),
        f"{'office share':<14}" + "".join(f"{pct(rows[s]['office_share_demand']):>10}" for s in SCENARIOS[1:]),
        f"{'res kWh/m2':<14}" + "".join(f"{rows[s]['intensity_residential_kWh_m2']:>10.0f}" for s in SCENARIOS[1:]),
    ]
    return "\n".join(lines)


def main() -> None:
    """Write the numbers as JSON and render the human-readable report to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Where to write the JSON artifact. Defaults to <results-dir>/paper_numbers.json.",
    )
    args = parser.parse_args()

    data = collect(load(args.results_dir))

    # self-check: GHG phase shares must add up
    shares = sum(data["status_quo"]["GHG_phase_shares"].values())
    if abs(shares - 1) > TOLERANCE:
        msg = f"GHG phase shares sum to {shares}, not 1"
        raise ValueError(msg)

    json_path = args.json_output or args.results_dir / "paper_numbers.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(render(data))  # noqa: T201 -- the rendered report is this script's console output


if __name__ == "__main__":
    main()
