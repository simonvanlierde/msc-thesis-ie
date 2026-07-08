"""Download the RIVM urban-heat-island raster and clip it to the city bbox.

RIVM publishes the national UHI layer (Atlas Natuurlijk Kapitaal) only as a ~1.95 GB
ZIP holding a single national GeoTIFF -- there is no WCS to clip server-side. This
downloads the ZIP once (cached), then windowed-reads just the city bbox straight out
of the archive via GDAL's ``/vsizip/`` (no multi-GB extraction), writing a small
clipped GeoTIFF in the raster's native CRS so it lines up with the RD New census
blocks the model's zonal statistics run against.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import rasterio
import requests
from pdok_http import retrying_session
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

# fetch_city_boundary writes the bbox in CRS84 (lon/lat); the national raster is in
# RD New. The bbox is reprojected to the raster CRS before windowing.
BBOX_CRS = "OGC:CRS84"
BBOX_VALUES = 4  # minx, miny, maxx, maxy


def _download(url: str, dest: Path) -> None:
    """Download the national archive, skipping a cached copy that is already complete."""
    expected: int | None = None
    try:  # HEAD is a cheap size probe; if it fails, fall through to a full download.
        head = requests.head(url, timeout=60, allow_redirects=True)
        head.raise_for_status()
        expected = int(head.headers.get("content-length", 0)) or None
    except requests.RequestException:
        expected = None

    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with retrying_session().get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if expected is not None and dest.stat().st_size != expected:
        msg = f"UHI download size mismatch for {url}: expected {expected}, got {dest.stat().st_size}"
        raise RuntimeError(msg)


def _tif_member(zip_path: Path) -> str:
    """Return the single GeoTIFF member inside the archive."""
    with zipfile.ZipFile(zip_path) as archive:
        tifs = [name for name in archive.namelist() if name.lower().endswith(".tif")]
    if not tifs:
        msg = f"No .tif found in {zip_path}."
        raise RuntimeError(msg)
    return tifs[0]


def _clip(zip_path: Path, member: str, bbox_crs84: tuple[float, float, float, float], output: Path) -> None:
    """Windowed-read the city bbox out of the zipped national GeoTIFF and write it."""
    minx, miny, maxx, maxy = bbox_crs84
    with rasterio.open(f"/vsizip/{zip_path}/{member}") as src:
        left, bottom, right, top = transform_bounds(BBOX_CRS, src.crs, minx, miny, maxx, maxy)
        window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=window)
        profile = src.profile
        profile.update(height=data.shape[1], width=data.shape[2], transform=src.window_transform(window))

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data)


def main() -> None:
    """Download and clip the national UHI raster to the requested bbox."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL of the national UHI ZIP.")
    parser.add_argument("--bbox", required=True, help="CRS84 minx,miny,maxx,maxy extent.")
    parser.add_argument("--cache-dir", required=True, help="Directory the national ZIP is cached in.")
    parser.add_argument("--output", required=True, help="Path for the clipped GeoTIFF.")
    args = parser.parse_args()

    bbox = tuple(float(value) for value in args.bbox.split(","))
    if len(bbox) != BBOX_VALUES:
        msg = f"--bbox must be 'minx,miny,maxx,maxy', got {args.bbox!r}"
        raise SystemExit(msg)

    zip_path = Path(args.cache_dir) / Path(args.url).name
    _download(args.url, zip_path)
    _clip(zip_path, _tif_member(zip_path), bbox, Path(args.output))


if __name__ == "__main__":
    main()
