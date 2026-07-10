"""Prepare model geodata from PDOK BAG and 3D Basisvoorziening sources."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from cdm.geometric import azimuth_rectangle

ENERGY_LABEL_TO_INT = {
    "A+++++": 1,
    "A++++": 2,
    "A+++": 3,
    "A++": 4,
    "A+": 5,
    "A": 6,
    "B": 7,
    "C": 8,
    "D": 9,
    "E": 10,
    "F": 11,
    "G": 12,
}


def _first_existing_column(columns: pd.Index, candidates: list[str], purpose: str) -> str:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    msg = f"Could not find {purpose} column. Tried {candidates}. Available columns: {list(columns)}"
    raise ValueError(msg)


# The only BAG verblijfsobject attributes the model needs, out of the 28 the fetch stores.
RESIDENCE_SOURCE_COLUMNS = ["identificatie", "oppervlakte", "gebruiksdoel", "status"]

_EP_WANTED_COLUMNS = {"bagverblijfsobjectid", "energieklasse"}
_MAX_PREAMBLE_SCAN = 20  # preamble is 2 rows in practice; scan a few extra for drift


def _ep_header_skiprows(path: Path) -> int:
    """Locate the EP-Online CSV header row (the export begins with a short key;value preamble)."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if "bagverblijfsobjectid" in line.lower():
                return index
            if index > _MAX_PREAMBLE_SCAN:
                break
    return 0


def _read_ep_online_labels(path: Path) -> pd.DataFrame:
    """Read the two needed columns from the semicolon-separated EP-Online export (~1.5 GB uncompressed)."""
    return pd.read_csv(
        path,
        sep=";",
        skiprows=_ep_header_skiprows(path),
        dtype="str",
        usecols=lambda name: name.strip().lower() in _EP_WANTED_COLUMNS,
    )


def _tile_bbox(gpkg_path: Path, boundary: gpd.GeoDataFrame | None) -> tuple[float, float, float, float] | None:
    """Boundary extent in the tile's own CRS, which is what pyogrio's bbox filter requires."""
    if boundary is None:
        return None
    crs = pyogrio.read_info(gpkg_path)["crs"]  # cheap: metadata only, no features read
    return tuple(boundary.to_crs(crs).total_bounds)


def _read_height_tiles(manifest_path: Path, boundary: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    manifest = json.loads(manifest_path.read_text())
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for tile in manifest["tiles"]:
            local_path = Path(tile["local_path"])
            if local_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(local_path) as archive:
                    archive.extractall(tmpdir_path / local_path.stem)
                gpkg_paths = sorted((tmpdir_path / local_path.stem).rglob("*.gpkg"))
            else:
                gpkg_paths = [local_path]
            if not gpkg_paths:
                msg = f"No GeoPackage found in {local_path}"
                raise ValueError(msg)
            # Let OGR drop out-of-city features before they ever reach pandas.
            frames.extend(
                gpd.read_file(gpkg_path, bbox=_tile_bbox(gpkg_path, boundary)) for gpkg_path in gpkg_paths
            )

    if not frames:
        msg = "No PDOK 3D height tile data was loaded."
        raise ValueError(msg)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def _prepare_buildings(height_tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Candidate names verified against a 2025 PDOK 3D Basisvoorziening
    # ``hoogtestatistieken_gebouwen`` tile (layer "buildings"): BAG id is
    # ``identificatie``, status ``status``, build year ``oorspronkelijkbouwjaar``,
    # ground/roof heights come as ``rf_h_ground`` / ``rf_h_roof_70p`` (the 70th
    # percentile roof height, matching the model's original ``h_dak_70p``).
    bag_id = _first_existing_column(height_tiles.columns, ["identificatie", "bagpandid", "id_bag"], "BAG pand id")
    status = _first_existing_column(height_tiles.columns, ["status", "pandstatus"], "building status")
    construction_year = _first_existing_column(
        height_tiles.columns,
        ["oorspronkelijkbouwjaar", "bouwjaar", "construction_year"],
        "construction year",
    )
    ground = _first_existing_column(
        height_tiles.columns,
        ["rf_h_ground", "h_maaiveld", "hoogte_maaiveld", "ground_elevation_m", "b3_h_maaiveld"],
        "ground elevation",
    )
    roof = _first_existing_column(
        height_tiles.columns,
        ["rf_h_roof_70p", "h_dak_70p", "dakhoogte_70p", "roof_elevation_m", "b3_h_70p", "rf_h_roof_max"],
        "roof elevation",
    )

    buildings = height_tiles.rename(
        columns={
            bag_id: "id_BAG",
            status: "status",
            construction_year: "construction_year",
            ground: "ground_elevation_m",
            roof: "roof_elevation_m",
        },
    ).copy()
    buildings["status"] = buildings["status"].replace({"bestaand": "Pand in gebruik", "Bestaand": "Pand in gebruik"})
    buildings["height_m"] = buildings["roof_elevation_m"].astype(float) - buildings["ground_elevation_m"].astype(float)
    buildings["floor_area_ground_m2"] = buildings.geometry.area
    buildings["volume_m3"] = buildings["floor_area_ground_m2"] * buildings["height_m"]
    buildings["MBR_geometry"] = buildings.geometry.minimum_rotated_rectangle()
    (
        buildings["MBR_azimuth"],
        buildings["MBR_width_m"],
        buildings["MBR_length_m"],
    ) = zip(*buildings["MBR_geometry"].map(azimuth_rectangle), strict=True)
    # MBR_geometry was only needed to derive the azimuth/width/length; drop it so
    # the GeoPackage keeps a single geometry column.
    return buildings.drop(columns="MBR_geometry")


def _prepare_residences(residences_path: Path, energy_labels_path: Path) -> gpd.GeoDataFrame:
    # Only 4 of the source's 28 attribute columns are used. pyogrio silently ignores
    # names it cannot find, so validate against the real schema before reading.
    available = set(pyogrio.read_info(residences_path)["fields"])
    missing = [column for column in RESIDENCE_SOURCE_COLUMNS if column not in available]
    if missing:
        msg = f"BAG verblijfsobject data is missing columns {missing}. Available columns: {sorted(available)}"
        raise ValueError(msg)

    residences = gpd.read_file(residences_path, columns=RESIDENCE_SOURCE_COLUMNS)
    residences = residences.rename(
        columns={
            "identificatie": "id_BAG",
            "oppervlakte": "floor_area_m2",
            "gebruiksdoel": "end_use",
        },
    )
    residences = residences[["id_BAG", "floor_area_m2", "status", "end_use", "geometry"]]
    residences = residences[residences["status"] == "Verblijfsobject in gebruik"].copy()

    labels = _read_ep_online_labels(energy_labels_path)
    label_id = _first_existing_column(
        labels.columns,
        ["BAGVerblijfsobjectID", "bagverblijfsobjectid"],
        "EP-Online BAG verblijfsobject id",
    )
    label_value = _first_existing_column(
        labels.columns,
        ["Energieklasse", "energieklasse", "energy_label"],
        "EP-Online energy label",
    )
    labels = labels[[label_id, label_value]].rename(columns={label_id: "id_BAG", label_value: "energy_label"})
    labels["energy_label_int"] = labels["energy_label"].map(ENERGY_LABEL_TO_INT)
    labels = labels.dropna(subset=["id_BAG", "energy_label_int"])

    return residences.merge(labels[["id_BAG", "energy_label_int"]], on="id_BAG", how="left")


def _join_buildings_residences(buildings: gpd.GeoDataFrame, residences: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # BAG residences arrive in lon/lat (CRS84); the height tiles are in RD New
    # (EPSG:28992). Reproject before the spatial join so "within" is meaningful.
    residences = residences.to_crs(buildings.crs)
    joined = gpd.sjoin(residences, buildings, predicate="within")
    joined["energy_label_int"] = joined["energy_label_int"].astype("float")

    # Floor-area-weighted mean energy label per building. Vectorised group sums
    # (no groupby-apply) so it behaves the same across pandas 2.0-3.0.
    labelled = joined.dropna(subset="energy_label_int")
    by_building = labelled["index_right"]
    weighted_label_sum = (labelled["energy_label_int"] * labelled["floor_area_m2"]).groupby(by_building).sum()
    floor_area_sum = labelled["floor_area_m2"].groupby(by_building).sum()
    mean_label = (weighted_label_sum / floor_area_sum).round().rename("energy_label_int")
    end_use = (
        joined.groupby(["index_right", "end_use"])["floor_area_m2"]
        .sum()
        .reset_index()
        .loc[lambda df: df.groupby("index_right")["floor_area_m2"].idxmax()]
        .set_index("index_right")["end_use"]
    )
    floor_area_and_count = joined.groupby("index_right").agg(
        floor_area_total_m2=("floor_area_m2", "sum"),
        number_of_residences=("id_BAG_left", pd.Series.nunique),
    )

    buildings = buildings.merge(mean_label, left_index=True, right_index=True, how="left")
    buildings = buildings.merge(end_use, left_index=True, right_index=True, how="left")
    buildings = buildings.merge(floor_area_and_count, left_index=True, right_index=True, how="left")
    buildings["energy_label"] = buildings["energy_label_int"].replace(
        {value: key for key, value in ENERGY_LABEL_TO_INT.items()},
    )
    return buildings


def _clip_to_boundary(buildings: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only buildings whose footprint sits inside the municipal boundary polygon."""
    boundary_geom = boundary.to_crs(buildings.crs).union_all()
    inside = buildings.geometry.representative_point().within(boundary_geom)
    # Contiguous index, as the pre-clip concat(ignore_index=True) used to guarantee downstream.
    return buildings[inside].reset_index(drop=True)


def main() -> None:
    """Prepare model input geodata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--height-manifest", required=True)
    parser.add_argument("--bag-residences", required=True)
    parser.add_argument("--energy-labels", required=True)
    parser.add_argument("--boundary", help="City boundary GeoJSON; buildings outside it are dropped.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", required=True)
    args = parser.parse_args()

    # Clip before deriving geometry: the MBR/azimuth/volume work and the sjoin then run
    # only on buildings that survive, instead of on the whole national tile extent.
    boundary = gpd.read_file(args.boundary) if args.boundary else None
    height_tiles = _read_height_tiles(Path(args.height_manifest), boundary)
    if boundary is not None:
        height_tiles = _clip_to_boundary(height_tiles, boundary)

    buildings = _prepare_buildings(height_tiles)
    residences = _prepare_residences(Path(args.bag_residences), Path(args.energy_labels))
    prepared = _join_buildings_residences(buildings, residences)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_file(output, layer=args.layer, driver="GPKG")


if __name__ == "__main__":
    main()
