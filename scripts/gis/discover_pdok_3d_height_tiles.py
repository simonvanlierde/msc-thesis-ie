"""Discover PDOK 3D Basisvoorziening height-attribute tile downloads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

from pdok_http import retrying_session

# The PDOK APIs allow up to 50 requests/s; stay well under it.
REQUEST_INTERVAL_S = 0.1


def _year_datetime(year: int) -> str:
    return f"{year}-07-01T00:00:00Z"


def discover_tiles(base_url: str, collection: str, bbox: str, year: int, limit: int) -> dict:
    """Return tile download metadata for a bbox/year."""
    url = urljoin(base_url.rstrip("/") + "/", f"collections/{collection}/items")
    params: dict[str, str | int] = {
        "f": "json",
        "limit": limit,
        "bbox": bbox,
        "datetime": _year_datetime(year),
    }
    tiles = []

    session = retrying_session()
    first = True
    while url:
        if not first:
            time.sleep(REQUEST_INTERVAL_S)
        first = False
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            download_link = properties.get("download_link")
            if not download_link:
                for link in feature.get("links", []):
                    if link.get("rel") == "enclosure":
                        download_link = link.get("href")
                        break
            if download_link:
                tiles.append(
                    {
                        "id": feature.get("id"),
                        "bladnr": properties.get("bladnr"),
                        "year": year,
                        "download_link": download_link,
                        "download_size_bytes": properties.get("download_size_bytes"),
                        "geometry": feature.get("geometry"),
                    },
                )

        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                next_url = link["href"]
                break
        url = next_url
        params = {}

    return {
        "source": f"{base_url.rstrip('/')}/collections/{collection}",
        "bbox_filter": bbox,
        "year": year,
        "tiles": tiles,
    }


def main() -> None:
    """Discover selected height tiles and write a JSON manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = discover_tiles(args.base_url, args.collection, args.bbox, args.year, args.limit)
    if not manifest["tiles"]:
        msg = "No PDOK 3D Basisvoorziening height tiles found for the configured bbox/year."
        raise SystemExit(msg)
    output.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
