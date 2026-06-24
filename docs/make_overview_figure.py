"""Generate the headline scenario-overview figure used in the README.

Reads the aggregated model results from ``data/output/`` and plots total cooling
energy demand and total greenhouse-gas emissions across the modelled scenarios.

Run from the repository root:

    uv run python docs/make_overview_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output"
FIGURE_PATH = Path(__file__).resolve().parent / "scenario_overview.png"

# Scenario file suffix -> human-readable label (in display order).
SCENARIOS = {
    "SQ": "Current",
    "2030": "2030",
    "2050_L": "2050\nLow",
    "2050_M": "2050\nMid",
    "2050_H": "2050\nHigh",
}


def load_scenario_totals() -> pd.DataFrame:
    """Aggregate each scenario's results into stock-wide totals."""
    rows = []
    for suffix, label in SCENARIOS.items():
        df = pd.read_csv(OUTPUT_DIR / f"CDM_results_{suffix}_full.csv")
        rows.append(
            {
                "scenario": label,
                "cooling_demand_GWh": df["E_cooling_capped_at_98th_percentile_kWh"].sum() / 1e6,
                "ghg_emissions_ktonne": df["GHG_emissions_total_kgCO2eq"].sum() / 1e6,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the two-panel scenario-overview figure and save it as a PNG."""
    totals = load_scenario_totals()

    fig, (ax_demand, ax_ghg) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax_demand.bar(totals["scenario"], totals["cooling_demand_GWh"], color="#2b6cb0")
    ax_demand.set_title("Annual cooling energy demand")
    ax_demand.set_ylabel("GWh per year")

    ax_ghg.bar(totals["scenario"], totals["ghg_emissions_ktonne"], color="#c05621")
    ax_ghg.set_title("Cooling-related GHG emissions")
    ax_ghg.set_ylabel("kilotonne CO₂-eq per year")

    for ax in (ax_demand, ax_ghg):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cooling demand and emissions in The Hague across scenarios",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {FIGURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
