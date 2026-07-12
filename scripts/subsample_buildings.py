"""Draw a representative subset of the prepared building stock.

``main.ipynb`` cannot hold the hourly time series for the full ~59k-building stock in
memory (~165 GB), so the notebook's figure section runs on a stratified sample and scales
the results back up. This script is that sampling step, made reproducible: it takes the
full prepared GeoPackage and writes a smaller one with the same schema plus a single
``stock_weight`` column, so every downstream rule treats ``building_subset: sample`` like
``full`` and aggregate_results can still recover stock-wide totals (see below).

Representative = proportional stratified sample over (building_type, energy_class), the two
axes that drive the cooling model, with at least one building kept per occupied stratum so
rare combinations are not dropped. A plain ``.head(n)`` or unstratified ``.sample(n)`` would
bias the stock-wide thermal flows the figures aggregate.

``stock_weight`` is each drawn building's multiplicity in the full stock (full stratum size /
number drawn from it). aggregate_results multiplies the summed physical quantities by it, so a
pipeline run on the sample produces stock-scale totals; the full stock omits the column
(weight 1), leaving that path unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from cdm.parameters import add_parameters_to_buildings
from cdm.readers import read_buildings, read_global_parameters, read_parameter_specific_data

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

PARAMETER_DIR = Path("data/input/parameters")
PARAMETERS_TOML = PARAMETER_DIR / "parameters.toml"

# The sample is one fixed set of buildings shared across all scenarios, so the strata are
# computed once under a single reference scenario. building_type/energy_class depend on the
# scenario's classification thresholds, not on its cooling parameters, so the status-quo
# scenario is the natural reference.
STRATIFY_SCENARIO = "SQ"
STRATIFY_COLUMNS = ["building_type_int", "energy_class_int"]


def draw_representative_sample(
    buildings: gpd.GeoDataFrame,
    strata: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> gpd.GeoDataFrame:
    """Proportional stratified sample: each stratum gets round(fraction * sample_size), min 1.

    ``strata`` carries the classification columns for the same rows as ``buildings`` (same
    index); only ``buildings`` columns are returned, plus a ``stock_weight`` giving each drawn
    building's multiplicity in the full stock (full stratum size / drawn) so aggregate_results
    can scale sample sums back up to stock totals. The full stock has no such column (weight 1).
    """
    counts = strata.groupby(STRATIFY_COLUMNS, dropna=False).size()
    fractions = counts / counts.sum()

    picked_index = []
    weights = []
    for stratum, fraction in fractions.items():
        target = max(round(fraction * sample_size), 1)
        members = strata.index[(strata[STRATIFY_COLUMNS] == stratum).all(axis=1)]
        take = min(target, len(members))
        # Seed offset by the stratum's own count keeps strata from drawing correlated rows.
        drawn = members.to_series().sample(n=take, random_state=seed + len(members)).tolist()
        picked_index.extend(drawn)
        weights.extend([len(members) / take] * take)

    sample = buildings.loc[picked_index].reset_index(drop=True)
    sample["stock_weight"] = weights
    return sample


def main() -> None:
    """Read the full stock, draw the stratified sample, write it with the input schema."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", required=True, help="Full prepared buildings GeoPackage")
    parser.add_argument("--buildings-layer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-layer", required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    buildings = read_buildings(Path(args.buildings), args.buildings_layer)

    # Reuse the model's own classification so the strata are identical to what it computes;
    # the extra parameter columns are only used for grouping and never written out.
    strata = add_parameters_to_buildings(
        buildings.copy(),
        read_global_parameters(PARAMETERS_TOML, STRATIFY_SCENARIO),
        read_parameter_specific_data(PARAMETER_DIR / "parameters_building_type.csv", STRATIFY_SCENARIO),
        read_parameter_specific_data(PARAMETER_DIR / "parameters_energy_class.csv", STRATIFY_SCENARIO),
    )

    sample = draw_representative_sample(buildings, strata, args.sample_size, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_file(output_path, layer=args.output_layer, driver="GPKG")


if __name__ == "__main__":
    main()
