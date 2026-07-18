"""Fetch the Habib et al. (2025) 5 m UHImax raster for The Hague from 4TU.

4TU serves the whole 99-municipality dataset as a single ~4.5 GB archive
(``UHI_NL.zip``) rather than per-municipality downloads. This resolves that
archive via the 4TU files API, caches it under ``data/raw/habib_uhi/``, extracts
only The Hague's UHImax member, reprojects to EPSG:28992 if needed, and writes
the model input raster. License: CC BY-SA 4.0 (attribution required in the
paper's data statement).
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

RD_NEW = "EPSG:28992"
RD_NEW_EPSG = 28992


def download_file(files_api: str, file_name: str, dest: Path) -> None:
    """Resolve file_name via the 4TU files API and download it to dest."""
    with urllib.request.urlopen(files_api) as resp:  # noqa: S310 -- fixed https endpoint from config
        files = json.load(resp)
    matches = [f for f in files if f["name"] == file_name]
    if not matches:
        msg = f"{file_name!r} not in 4TU file list: {[f['name'] for f in files]}"
        raise FileNotFoundError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(matches[0]["download_url"]) as resp, dest.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)


def extract_member(archive_path: Path, member: str, dest: Path) -> None:
    """Extract a single member (The Hague's UHImax GeoTIFF) from the national zip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf, zf.open(member) as src, dest.open("wb") as fh:
        shutil.copyfileobj(src, fh)


def ensure_rd_new(src_path: Path, out_path: Path) -> None:
    """Write out_path in EPSG:28992, reprojecting only if the source differs."""
    with rasterio.open(src_path) as src:
        if src.crs is not None and src.crs.to_epsg() == RD_NEW_EPSG:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, out_path)
            return
        transform, width, height = calculate_default_transform(src.crs, RD_NEW, src.width, src.height, *src.bounds)
        meta = src.meta | {"crs": RD_NEW, "transform": transform, "width": width, "height": height}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                resampling=Resampling.bilinear,
            )


def main() -> None:
    """Download, cache and normalize the Habib UHImax raster."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-api", required=True)
    parser.add_argument("--archive-file-name", required=True, help="4TU top-level file name (the national zip).")
    parser.add_argument("--zip-member", required=True, help="Path inside the archive to The Hague's UHImax GeoTIFF.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/habib_uhi"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive = args.cache_dir / args.archive_file_name
    if not archive.exists():
        download_file(args.files_api, args.archive_file_name, archive)

    extracted = args.cache_dir / Path(args.zip_member).name
    if not extracted.exists():
        extract_member(archive, args.zip_member, extracted)

    ensure_rd_new(extracted, args.output)


if __name__ == "__main__":
    main()
