"""Fetch a paginated OGC API Features collection to a GeoPackage.

GeoJSON was the original output, but the BAG residences layer is ~460k features: as
GeoJSON it is 719 MB and takes ~18 s to re-parse on every downstream read. The same
data as a GeoPackage is 296 MB and reads in ~4 s, so the fetch writes GPKG directly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
from pdok_http import retrying_session

# The PDOK BAG API allows up to 50 requests/s; stay well under it.
REQUEST_INTERVAL_S = 0.1


def fetch_features(
    base_url: str, collection: str, bbox: str, limit: int, max_pages: int | None, timeout: int = 60
) -> dict:
    """Fetch all pages for an OGC collection and return a FeatureCollection."""
    url = urljoin(base_url.rstrip("/") + "/", f"collections/{collection}/items")
    params: dict[str, str | int] = {"f": "json", "limit": limit, "bbox": bbox}
    features = []
    links = []

    session = retrying_session()
    page = 0
    while url:
        if page:
            time.sleep(REQUEST_INTERVAL_S)
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        page += 1
        features.extend(payload.get("features", []))
        links.extend(payload.get("links", []))

        if max_pages is not None and page >= max_pages:
            break

        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                next_url = link["href"]
                break
        url = next_url
        params = {}

    return {
        "type": "FeatureCollection",
        "source": f"{base_url.rstrip('/')}/collections/{collection}",
        "bbox_filter": bbox,
        "page_limit": max_pages,
        "numberReturned": len(features),
        "links": links,
        "features": features,
    }


def to_geodataframe(features: list[dict]) -> gpd.GeoDataFrame:
    """Build a GeoPackage-writable frame from OGC features (which are served in CRS84)."""
    # from_features reads only `properties` and `geometry`, so the feature-level `id`
    # member (which the GeoJSON driver used to surface as an `id` column) would be lost.
    for feature in features:
        if "id" in feature:
            feature["properties"].setdefault("id", feature["id"])

    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    # GeoPackage has no list column type, and BAG's `pand.href` is a list of URLs.
    for column in frame.columns:
        if column != frame.geometry.name and frame[column].map(lambda value: isinstance(value, list)).any():
            frame[column] = frame[column].map(lambda value: ",".join(value) if isinstance(value, list) else value)
    return frame


def main() -> None:
    """Run the OGC fetcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--bbox", required=True, help="CRS84 bbox as minx,miny,maxx,maxy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-request read timeout (s). Some PDOK collections are slow to serve large pages.",
    )
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    collection = fetch_features(
        args.base_url, args.collection, args.bbox, args.limit, args.max_pages, args.timeout
    )
    if not collection["features"]:
        msg = f"No features returned for collection {args.collection!r} in bbox {args.bbox}."
        raise SystemExit(msg)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Provenance (source URL, bbox, page limit) is not embedded: it is fully determined by
    # config/sources.yaml plus the committed bbox file, both declared inputs of this rule.
    to_geodataframe(collection["features"]).to_file(output, driver="GPKG", layer=args.collection)


if __name__ == "__main__":
    main()
