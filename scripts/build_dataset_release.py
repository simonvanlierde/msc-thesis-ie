"""Package open building inputs without energy labels or environmental results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import geopandas as gpd

RD_NEW_EPSG = 28992

# An allowlist prevents new model-output or restricted-source columns from leaking.
FIELDS = {
    "id_BAG": ("string", "BAG building identifier; may repeat for differing source footprints."),
    "status": ("string", "BAG building status."),
    "construction_year": ("integer", "BAG year of construction."),
    "height_m": ("number", "Roof elevation minus ground elevation, metres."),
    "floor_area_ground_m2": ("number", "Footprint area, square metres."),
    "volume_m3": ("number", "Footprint area times height, cubic metres."),
    "MBR_azimuth": ("number", "Minimum bounding rectangle orientation, degrees."),
    "MBR_width_m": ("number", "Minimum bounding rectangle width, metres."),
    "MBR_length_m": ("number", "Minimum bounding rectangle length, metres."),
    "end_use": ("string", "Dominant BAG residence use, weighted by floor area."),
    "floor_area_total_m2": ("number", "Summed BAG residence floor area, square metres."),
    "number_of_residences": ("number", "Number of distinct in-use BAG residences within the building."),
    "UHI_max_C": ("number", "Habib UHImax, averaged within a 30 metre square radius; fallback 1 K."),
}


def sha256(path: Path) -> str:
    """Return the file's SHA-256 checksum."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_release(source: Path, output: Path) -> Path:
    """Build a new release folder and ZIP; refuse to overwrite an existing folder."""
    archive = output.parent / f"{output.name}.zip"
    review = output.parent / f"{output.name}-review"
    for destination in (output, archive, review):
        if destination.exists():
            raise FileExistsError(destination)
    buildings = gpd.read_file(source, layer="BAG_buildings_full", columns=list(FIELDS))
    missing = set(FIELDS) - set(buildings.columns)
    if missing or buildings.empty or (buildings.crs is None or buildings.crs.to_epsg() != RD_NEW_EPSG):
        msg = f"Invalid building inputs: missing={sorted(missing)}, rows={len(buildings)}, crs={buildings.crs}"
        raise ValueError(msg)
    buildings = buildings[[*FIELDS, "geometry"]].sort_values("id_BAG", kind="stable")
    invalid = ~buildings.geometry.is_valid | buildings.geometry.is_empty | buildings.geometry.isna()
    excluded = buildings.loc[invalid, ["id_BAG"]].reset_index(names="source_row")
    source_rows = len(buildings)
    buildings = buildings.loc[~invalid].reset_index(drop=True)
    if buildings.empty:
        msg = "No valid building geometries remain."
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=False)
    review.mkdir(exist_ok=False)
    excluded.assign(reason="invalid or empty geometry").to_csv(review / "excluded-records.csv", index=False)
    buildings = buildings.astype(
        {
            name: {"string": "string", "integer": "Int64", "number": "Float64"}[kind]
            for name, (kind, _) in FIELDS.items()
        },
    )
    buildings.to_file(output / "buildings.gpkg", layer="buildings", driver="GPKG")
    buildings.to_parquet(output / "buildings.parquet", compression="zstd", index=False)
    buildings.to_wkt().to_csv(output / "buildings.csv", index=False, lineterminator="\n", encoding="utf-8")
    fields = [{"name": name, "type": kind, "description": description} for name, (kind, description) in FIELDS.items()]
    fields.append({"name": "geometry", "type": "string", "description": "Building polygon as WKT in EPSG:28992."})
    package = {
        "$schema": "https://datapackage.org/profiles/2.0/datapackage.json",
        "name": "hague-uhi-building-inputs",
        "title": "The Hague building inputs with local UHImax (open subset)",
        "version": "0.1.0",
        "licenses": [{"name": "CC-BY-SA-4.0", "path": "https://creativecommons.org/licenses/by-sa/4.0/"}],
        "resources": [
            {
                "name": "buildings",
                "path": "buildings.csv",
                "type": "table",
                "format": "csv",
                "bytes": (output / "buildings.csv").stat().st_size,
                "hash": "sha256:" + sha256(output / "buildings.csv"),
                "schema": {"fields": fields, "missingValues": [""]},
            },
            {
                "name": "buildings-geoparquet",
                "type": "file",
                "path": "buildings.parquet",
                "format": "parquet",
                "description": "GeoParquet copy of buildings.csv with EPSG:28992 geometry and zstd compression.",
            },
            {"name": "buildings-geopackage", "type": "file", "path": "buildings.gpkg", "format": "gpkg"},
        ],
    }
    (output / "datapackage.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    project = Path(__file__).resolve().parents[1]
    card = (
        (project / "data/README.md")
        .read_text(encoding="utf-8")
        .replace(
            "## Composition",
            f"## Composition\n\nThis version contains {len(buildings):,} rows from {source_rows:,} source rows. "
            f"It excludes {len(excluded):,} invalid or empty geometries.",
        )
    )
    (output / "README.md").write_text(card, encoding="utf-8")
    (output / "LICENSES").mkdir()
    shutil.copyfile(project / "LICENSES/CC-BY-SA-4.0.txt", output / "LICENSES/CC-BY-SA-4.0.txt")
    (output / "REUSE.toml").write_text(
        'version = 1\n\n[[annotations]]\npath = ["**"]\nSPDX-License-Identifier = "CC-BY-SA-4.0"\n'
        'SPDX-FileCopyrightText = "Kadaster; Habib et al. (2025); Simon van Lierde"\n',
        encoding="utf-8",
    )
    shutil.copyfile(project / "data/CITATION.cff", output / "CITATION.cff")
    inventory = {
        "source_rows": source_rows,
        "excluded_invalid_geometry": len(excluded),
        "rows": len(buildings),
        "unique_bag_ids": buildings["id_BAG"].nunique(),
        "nulls": buildings.isna().sum().to_dict(),
        "crs": "EPSG:28992",
        "source_sha256": sha256(source),
        "columns": list(buildings.columns),
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    restored = gpd.read_parquet(output / "buildings.parquet")
    if len(restored) != len(buildings) or list(restored.columns) != list(buildings.columns):
        msg = "GeoParquet row count or columns changed during writing."
        raise ValueError(msg)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)} *{path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(path for path in output.rglob("*") if path.is_file()):
            handle.write(path, arcname=path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as handle:
        for path in sorted(path for path in output.rglob("*") if path.is_file()):
            if hashlib.sha256(handle.read(path.relative_to(output).as_posix())).hexdigest() != sha256(path):
                msg = f"Archive checksum mismatch: {path.name}"
                raise ValueError(msg)
    return archive


def main() -> int:
    """Build the private release package; do not contact a deposit service."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Exit codes: 0 success, 1 invalid inputs, 2 usage error, 3 I/O error.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Prepared UHI-corrected full-stock GeoPackage.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the open release subset.")
    args = parser.parse_args()
    try:
        archive = build_release(args.source, args.output)
    except ValueError as error:
        print(f"error: {error!s}", file=sys.stderr)  # noqa: T201
        return 1
    except OSError as error:
        print(f"error: {error!s}", file=sys.stderr)  # noqa: T201
        return 3
    print(archive)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
