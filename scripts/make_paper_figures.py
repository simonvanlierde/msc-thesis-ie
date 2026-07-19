"""Generate the journal-paper main-text figures from the scenario results.

F1: model pipeline schematic (no data input).
F2: office vs residential contribution across the impact chain (100%-stacked bars).
F5: distribution of building-level cooling intensity by end use.

Run from the repository root:

    python scripts/make_paper_figures.py [--input-dir results] [--figures-dir results/figures]

F3 (cooling gap vs income/UHI map) and F4 (scenario overview, docs/make_overview_figure.py)
are produced elsewhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyogrio
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Repo figure palette (validated CVD-safe pair, see docs/make_overview_figure.py).
BLUE = "#2b6cb0"  # residential
ORANGE = "#c05621"  # office

# Below this share (%), an F2 bar label no longer fits inside its segment.
INSIDE_LABEL_MIN_PCT = 15

F2_METRICS = {
    "Floor area": "floor_area_total_m2",
    "Cooling demand": "E_cooling_kWh",
    "Electricity use": "electricity_use_kWh",
    "GHG emissions": "GHG_emissions_total_kgCO2eq",
    "Equipment mass": "mass_cooling_equipment_kg",
}


def load_sq(input_dir: Path) -> pd.DataFrame:
    """Load the SQ scenario results into a DataFrame."""
    df = pd.read_csv(input_dir / "CDM_results_SQ_full.csv")
    df["is_office"] = df["building_type"].str.contains("office", case=False)
    return df


def make_f1_schematic(path: Path) -> None:
    """F1: the three model layers and their data flows."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, lines: list[str], face: str) -> None:
        """Draw a rounded box with a title and a list of lines inside."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=face, ec="#4a5568", lw=1))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top", fontsize=7.8, linespacing=1.4)

    def arrow(x0: float, x1: float, y: float) -> None:
        """Draw a horizontal arrow from (x0, y) to (x1, y)."""
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16, color="#4a5568", lw=1.4))

    box(
        0.15,
        0.75,
        2.6,
        2.3,
        "Geospatial layer",
        [
            "BAG footprints & attributes",
            "3D BAG heights",
            "EP-Online energy labels",
            "→ stock of 8 archetypes with",
            "envelope parameters",
        ],
        "#e8eef7",
    )
    box(
        3.65,
        0.75,
        2.6,
        2.3,
        "Thermodynamic layer",
        [
            "hourly heat balance per building",
            "KNMI weather + UHI correction",
            "transmission · air exchange ·",
            "solar · internal gains",
            "→ cooling demand & peak power",
        ],
        "#e8eef7",
    )
    box(
        7.15,
        0.75,
        2.7,
        2.3,
        "Environmental-impact layer",
        [
            "technology mix & SEER",
            "refrigerant leakage",
            "ecoinvent production/EoL proxies",
            "grid carbon intensity",
            "→ electricity · GHG · mass · ADP · CSI",
        ],
        "#fbeee4",
    )
    arrow(2.85, 3.55, 1.9)
    arrow(6.35, 7.05, 1.9)
    ax.text(
        5.0,
        0.3,
        "Scenario drivers (stock growth & renovation, climate, technology mix & efficiency, "
        "market penetration, comfort set-points, grid) enter the two model layers",
        ha="center",
        va="center",
        fontsize=7.8,
        style="italic",
        color="#4a5568",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_f2_contribution(sq: pd.DataFrame, path: Path) -> None:
    """F2: office vs residential share of each impact-chain metric (SQ)."""
    shares = {label: sq.loc[sq["is_office"], col].sum() / sq[col].sum() for label, col in F2_METRICS.items()}
    labels = list(reversed(list(shares)))
    office = [shares[label] * 100 for label in labels]
    residential = [100 - o for o in office]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(labels, office, color=ORANGE, label="Office", edgecolor="white", linewidth=2)
    ax.barh(labels, residential, left=office, color=BLUE, label="Residential", edgecolor="white", linewidth=2)
    for i, o in enumerate(office):
        if o >= INSIDE_LABEL_MIN_PCT:
            ax.text(o / 2, i, f"{o:.0f}%", va="center", ha="center", fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(
                o + 1.2,
                i,
                f"{o:.0f}%",
                va="center",
                ha="left",
                fontsize=9,
                color=ORANGE,
                fontweight="bold",
                bbox={"fc": "white", "ec": "none", "pad": 1},
            )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Office share of citywide total (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False)
    fig.suptitle("Offices concentrate cooling impacts far beyond their floor area", fontsize=11, y=1.06)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_buildings(input_dir: Path) -> pd.DataFrame:
    """Per-building results (the CSV is aggregated per archetype x energy class)."""
    gpkg = input_dir / "geodata" / "buildings_with_CDM_results_SQ_full.gpkg"
    df = pyogrio.read_dataframe(
        gpkg,
        layer="buildings_with_CDM_results_SQ_full",
        columns=["building_type", "E_cooling_kWh", "floor_area_total_m2"],
        read_geometry=False,
    )
    df["is_office"] = df["building_type"].str.contains("office", case=False)
    return df


def make_f5_intensity(sq: pd.DataFrame, path: Path) -> None:
    """F5: building-level cooling-intensity distribution by end use (SQ)."""
    intensity = sq["E_cooling_kWh"] / sq["floor_area_total_m2"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # per-group 99.5th percentile so the office tail is shown, not piled into the last bin
    xmax = max(intensity[m].quantile(0.995) for m in (sq["is_office"], ~sq["is_office"]))
    bins = 60
    for mask, color, label in [(~sq["is_office"], BLUE, "Residential"), (sq["is_office"], ORANGE, "Office")]:
        ax.hist(intensity[mask], bins=bins, range=(0, xmax), density=True, alpha=0.55, color=color, label=label)
        med = intensity[mask].median()
        ax.axvline(med, color=color, ls="--", lw=1.5)
        ax.text(med, ax.get_ylim()[1] * 0.97, f"  median {med:.0f}", color=color, fontsize=8.5, va="top")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Cooling energy demand intensity (kWh/m²·yr)")
    ax.set_ylabel("Density of buildings")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Cooling intensity distribution across the building stock", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the paper's main-text figures F1, F2 and F5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    make_f1_schematic(args.figures_dir / "F1_pipeline_schematic.png")
    make_f2_contribution(load_sq(args.input_dir), args.figures_dir / "F2_contribution_by_type.png")
    make_f5_intensity(load_buildings(args.input_dir), args.figures_dir / "F5_intensity_distribution.png")
    print(f"Saved F1, F2, F5 to {args.figures_dir}")  # noqa: T201 -- intentional user-facing console output


if __name__ == "__main__":
    main()
