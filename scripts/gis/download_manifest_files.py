"""Download files listed in a JSON manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from scripts.gis.pdok_http import retrying_session


def _download(url: str, output: Path, expected_size: int | None) -> None:
    # Skip only when we can confirm the existing file is complete; without an expected size we
    # cannot tell a full file from one truncated by an aborted run, so re-download to be safe.
    if output.exists() and expected_size is not None and output.stat().st_size == expected_size:
        return

    with retrying_session().get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if expected_size is not None and output.stat().st_size != expected_size:
        msg = f"Downloaded size mismatch for {url}: expected {expected_size}, got {output.stat().st_size}"
        raise RuntimeError(msg)


def main() -> None:
    """Download manifest tiles and write a local-path manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-manifest", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    local_tiles = []
    for tile in manifest["tiles"]:
        url = tile["download_link"]
        filename = Path(urlparse(url).path).name
        output = output_dir / filename
        _download(url, output, tile.get("download_size_bytes"))
        local_tile = dict(tile)
        local_tile["local_path"] = str(output)
        local_tiles.append(local_tile)

    local_manifest = dict(manifest)
    local_manifest["tiles"] = local_tiles
    local_manifest_path = Path(args.local_manifest)
    local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    local_manifest_path.write_text(json.dumps(local_manifest, indent=2))


if __name__ == "__main__":
    main()
