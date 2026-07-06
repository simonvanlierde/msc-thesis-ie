"""Fetch a paginated OGC API Features collection to GeoJSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

from pdok_http import retrying_session

# The PDOK BAG API allows up to 50 requests/s; stay well under it.
REQUEST_INTERVAL_S = 0.1


def fetch_features(base_url: str, collection: str, bbox: str, limit: int, max_pages: int | None) -> dict:
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
        response = session.get(url, params=params, timeout=60)
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


def main() -> None:
    """Run the OGC fetcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--bbox", required=True, help="CRS84 bbox as minx,miny,maxx,maxy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fetch_features(args.base_url, args.collection, args.bbox, args.limit, args.max_pages), indent=2),
    )


if __name__ == "__main__":
    main()
