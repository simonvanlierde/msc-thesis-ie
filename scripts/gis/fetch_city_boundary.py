"""Fetch a municipal boundary from the PDOK Bestuurlijke gebieden API by name.

Writes the boundary as GeoJSON (used later to clip buildings to the municipality)
and a ``bbox.txt`` holding the CRS84 ``minx,miny,maxx,maxy`` extent, which the
BAG / 3D acquisition rules feed to their OGC queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin

from scripts.gis.pdok_http import retrying_session


def _bbox_from_geometry(geometry: dict) -> tuple[float, float, float, float]:
    """Compute a lon/lat bounding box by walking every coordinate pair in a GeoJSON geometry."""
    xs: list[float] = []
    ys: list[float] = []

    def walk(coords: object) -> None:
        if not isinstance(coords, (list, tuple)):
            return
        # A position is a [lon, lat, ...] pair of numbers; anything else nests further.
        first = coords[0] if coords else None
        second = coords[1] if len(coords) > 1 else None
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            xs.append(float(first))
            ys.append(float(second))
        else:
            for item in coords:
                walk(item)

    walk(geometry.get("coordinates", []))
    if not xs:
        msg = "Boundary geometry contained no coordinates."
        raise ValueError(msg)
    return min(xs), min(ys), max(xs), max(ys)


def fetch_boundary(base_url: str, collection: str, name: str) -> dict:
    """Return the GeoJSON feature for the municipality named ``name``."""
    url = urljoin(base_url.rstrip("/") + "/", f"collections/{collection}/items")
    response = retrying_session().get(url, params={"f": "json", "limit": 10, "naam": name}, timeout=60)
    response.raise_for_status()
    features = [f for f in response.json().get("features", []) if f.get("properties", {}).get("naam") == name]
    if not features:
        msg = f"No municipality named {name!r} in PDOK collection {collection!r}. Check the official name."
        raise SystemExit(msg)
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    """Fetch a city boundary and write the GeoJSON plus its bbox."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True, help="Boundary GeoJSON output path.")
    parser.add_argument("--bbox-output", required=True, help="Path for the CRS84 bbox string.")
    args = parser.parse_args()

    boundary = fetch_boundary(args.base_url, args.collection, args.name)
    minx, miny, maxx, maxy = _bbox_from_geometry(boundary["features"][0]["geometry"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(boundary))
    Path(args.bbox_output).write_text(f"{minx},{miny},{maxx},{maxy}")


if __name__ == "__main__":
    main()
