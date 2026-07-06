#!/usr/bin/env python3
"""Convert the thesis CDM result CSVs into one compact JSON for the web dashboard.

Source of truth: ``data/output/CDM_results_{scenario}_full.csv`` — real thesis
model output, aggregated per building archetype (building_type x energy_class).
This step only reshapes and rolls up those numbers; it invents nothing.

Run:  python dashboard/scripts/build_scenarios.py
Out:  dashboard/public/data/scenarios.json

No third-party deps (stdlib csv/json) so it runs anywhere without the model env.
"""

# CLI build script (not the linted scientific package): prints, asserts, deferred imports are intentional.
# ruff: noqa: D103, E741, EXE001, PLR2004, S101, T201
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "output"
OUT = REPO / "dashboard" / "public" / "data" / "scenarios.json"

# Scenario key -> human label. Order matters for the UI timeline.
SCENARIOS = {
    "SQ": "Status quo (present)",
    "2030": "2030",
    "2050_L": "2050 — low",
    "2050_M": "2050 — medium",
    "2050_H": "2050 — high",
}

# The four life-cycle GHG stages, in life-cycle order (production -> use -> EoL).
# refrigerant leaks are a use-phase direct emission, kept separate from grid use.
GHG_STAGES = {
    "production_phase": "Equipment production",
    "electricity": "Operational electricity",
    "refrigerant_leaks": "Refrigerant leakage",
    "EoL_phase": "End-of-life",
}

# Impact categories carried through to the LCA view. label + unit for the axis.
CATEGORIES = {
    "GHG_emissions_total_kgCO2eq": ("Climate change", "kg CO₂-eq"),
    "ADP_kgSbeq": ("Abiotic resource depletion", "kg Sb-eq"),
    "CSI_kgSieq": ("Crustal scarcity", "kg Si-eq"),
}


def _f(v: str) -> float:
    return float(v) if v not in ("", "nan") else 0.0


def classify(building_type: str) -> dict[str, str]:
    """Split the archetype name into the orthogonal facets the UI filters on."""
    bt = building_type.lower()
    return {
        "use": "Office" if "office" in bt else "Residential",
        "age": "New" if bt.startswith("new") else "Old",
        "form": "Highrise" if "highrise" in bt else "Lowrise",
    }


def read_scenario(key: str) -> list[dict]:
    path = SRC / f"CDM_results_{key}_full.csv"
    rows: list[dict] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            facets = classify(r["building_type"])
            rows.append(
                {
                    "building_type": r["building_type"],
                    "energy_class": r["energy_class"],
                    **facets,
                    "floor_area_m2": _f(r["floor_area_total_m2"]),
                    "E_cooling_kWh": _f(r["E_cooling_kWh"]),
                    "P_peak_kW": _f(r["P_cooling_peak_kW"]),
                    "electricity_kWh": _f(r["electricity_use_kWh"]),
                    "ghg": {
                        "production_phase": _f(r["GHG_emissions_production_phase_kgCO2eq"]),
                        "electricity": _f(r["GHG_emissions_electricity_kgCO2eq"]),
                        "refrigerant_leaks": _f(r["GHG_emissions_refrigerant_leaks_kgCO2eq"]),
                        "EoL_phase": _f(r["GHG_emissions_EoL_phase_kgCO2eq"]),
                    },
                    "GHG_emissions_total_kgCO2eq": _f(r["GHG_emissions_total_kgCO2eq"]),
                    "ADP_kgSbeq": _f(r["ADP_kgSbeq"]),
                    "CSI_kgSieq": _f(r["CSI_kgSieq"]),
                },
            )
    return rows


def summarise(rows: list[dict]) -> dict:
    def s(key: str) -> float:
        return sum(r[key] for r in rows)

    totals = {
        "floor_area_m2": s("floor_area_m2"),
        "E_cooling_kWh": s("E_cooling_kWh"),
        "electricity_kWh": s("electricity_kWh"),
        "GHG_emissions_total_kgCO2eq": s("GHG_emissions_total_kgCO2eq"),
        "ADP_kgSbeq": s("ADP_kgSbeq"),
        "CSI_kgSieq": s("CSI_kgSieq"),
    }
    lca_by_stage = {stage: sum(r["ghg"][stage] for r in rows) for stage in GHG_STAGES}
    return {"totals": totals, "lca_by_stage": lca_by_stage}


def build() -> dict:
    scenarios = {}
    for key, label in SCENARIOS.items():
        rows = read_scenario(key)
        scenarios[key] = {"label": label, **summarise(rows), "archetypes": rows}
    return {
        "meta": {
            "source": "MSc thesis model output — data/output/CDM_results_*.csv",
            "doi_data": "10.5281/zenodo.8344580",
            "ghg_stages": GHG_STAGES,
            "categories": {k: {"label": l, "unit": u} for k, (l, u) in CATEGORIES.items()},
            "scenario_order": list(SCENARIOS),
        },
        "scenarios": scenarios,
    }


def self_check(data: dict) -> None:
    """Reproduce the README headline findings from the built data (real numbers)."""
    sq = data["scenarios"]["SQ"]["archetypes"]
    area = sum(r["floor_area_m2"] for r in sq)
    demand = sum(r["E_cooling_kWh"] for r in sq)
    ghg = sum(r["GHG_emissions_total_kgCO2eq"] for r in sq)
    off = [r for r in sq if r["use"] == "Office"]

    def share(sub: list[float], tot: float) -> float:
        return sum(sub) / tot

    office_area = share([r["floor_area_m2"] for r in off], area)
    office_demand = share([r["E_cooling_kWh"] for r in off], demand)
    office_ghg = share([r["GHG_emissions_total_kgCO2eq"] for r in off], ghg)

    # README: offices ~13% of floor area, ~34% of demand, ~65% of GHG.
    assert 0.11 < office_area < 0.15, f"office area share {office_area:.2%}"
    assert 0.30 < office_demand < 0.38, f"office demand share {office_demand:.2%}"
    assert 0.60 < office_ghg < 0.70, f"office GHG share {office_ghg:.2%}"
    print(f"self-check OK  offices: {office_area:.0%} area, {office_demand:.0%} demand, {office_ghg:.0%} GHG")


if __name__ == "__main__":
    data = build()
    self_check(data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)}  ({kb:.1f} kB)")
