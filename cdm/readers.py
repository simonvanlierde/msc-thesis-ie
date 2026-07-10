"""Input readers for the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

import csv
import json
import tomllib
from typing import TYPE_CHECKING

import geopandas as gpd

from cdm.constants import REQUIRED_GLOBAL_PARAMETERS

if TYPE_CHECKING:
    from pathlib import Path

# The stock we model: in-use residential and office buildings with a known energy label.
# Handed to GDAL as an attribute filter so the ~60% of rows we would immediately drop are
# never decoded. `IS NOT NULL` is the SQL spelling of the .dropna() this replaces.
BUILDING_ROW_FILTER = (
    "status = 'Pand in gebruik' "
    "AND energy_label IS NOT NULL "
    "AND end_use IN ('woonfunctie', 'kantoorfunctie', 'residential', 'office')"
)


def read_buildings(buildings_path: Path, layer_name: str = "BAG_buildings") -> gpd.GeoDataFrame:
    """Reads the buildings for which the cooling demand is calculated from a GeoPackage file.

    Rows outside the modelled stock are excluded by ``BUILDING_ROW_FILTER`` during the read itself.

    Args:
        buildings_path (Path): The path to the GeoPackage file containing the buildings for which the cooling demand is calculated.
        layer_name (str, optional): The name of the layer in the GeoPackage file containing the buildings for which the cooling demand is calculated. Defaults to "BAG_buildings".

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame containing the buildings for which the cooling demand is calculated.
    """
    return gpd.read_file(filename=buildings_path, layer=layer_name, where=BUILDING_ROW_FILTER).assign(
        # Translate end use from Dutch to English. Scoped to end_use: a frame-wide .replace()
        # would rewrite these strings in any column that ever came to hold them.
        end_use=lambda df: df["end_use"].replace({"woonfunctie": "residential", "kantoorfunctie": "office"}),
    )


def read_global_parameters(global_parameters_path: Path, scenario: str | None = None) -> dict[str, float]:
    """Reads the global parameters for the cooling demand model from the consolidated parameters.toml.

    ``scenario`` selects the ``[base]`` values merged with the ``[scenario.<scenario>]`` overrides.

    Args:
        global_parameters_path (Path): The path to the parameters.toml file.
        scenario (str, optional): The scenario whose overrides to merge onto ``[base]``. Defaults to None.

    Returns:
        dict[str, float]: The dictionary containing the global parameters for the cooling demand model.
    """
    with global_parameters_path.open("rb") as toml_file:
        config = tomllib.load(toml_file)
    merged = {**config.get("base", {}), **config.get("scenario", {}).get(scenario, {})}

    # Fail loudly at load if the config is missing a parameter the model needs, rather than
    # deep inside the model with a cryptic KeyError.
    missing = REQUIRED_GLOBAL_PARAMETERS - merged.keys()
    if missing:
        msg = f"parameters.toml (scenario {scenario!r}) is missing required global parameters: {sorted(missing)}"
        raise ValueError(msg)

    # Keep list-valued parameters (e.g. energy_class_ranges) as lists; coerce the rest to float
    return {key: value if isinstance(value, list) else float(value) for key, value in merged.items()}


def read_parameter_specific_data(parameters_path: Path, scenario: str | None = None) -> list[dict]:
    """Reads the building, energy label or cooling technology parameters for the cooling demand model from a csv file.

    Each record mixes value types (an integer id, a string name, optional list-valued columns and float parameters), so the dictionaries are intentionally left untyped at the value level. The consolidated tidy csv carries a ``scenario`` column; rows are filtered to ``scenario`` and the marker column is dropped.

    Args:
        parameters_path (Path): The path to the csv file containing the parameters for the cooling demand model.
        scenario (str, optional): The scenario to keep. Defaults to None.

    Returns:
        list[dict]: The list of dictionaries containing the parameters for the cooling demand model.
    """
    with parameters_path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        parameter_specific_data = []
        for row in reader:
            if row["scenario"] != scenario:  # keep only the requested scenario, then drop the marker column
                continue
            del row["scenario"]
            parameter_specific_data.append(
                {
                    key: int(value)
                    if i == 0  # First parameter (building type, energy class or cooling technology as an integer)
                    else str(value)
                    if i == 1  # Second parameter (building type, energy class or cooling technology name)
                    else json.loads(value)
                    if key.startswith("energy_labels_included_")  # Energy label lists
                    else float(value)  # All other parameter values
                    for i, (key, value) in enumerate(row.items())
                },
            )
        return parameter_specific_data
