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
import shutil
import zipfile
from pathlib import Path

import rasterio
import requests
from rasterio.crs import CRS  # ty: ignore[unresolved-import]  # NOTE: compiled module without stubs since rasterio 1.5
from rasterio.warp import Resampling, calculate_default_transform, reproject

from scripts.gis.pdok_http import retrying_session

RD_NEW_EPSG = 28992
RD_NEW = CRS.from_epsg(RD_NEW_EPSG)


def download_file(files_api: str, file_name: str, dest: Path) -> None:
    """Resolve file_name via the 4TU files API and download it to dest.

    Skips re-download only when a cached copy's size already matches the
    server-reported size, mirrors fetch_uhi_raster.py's ``_download``. A truncated
    prior download is otherwise indistinguishable from a complete one, which would
    make the Snakefile's ``retries: NETWORK_RETRIES`` re-invoke and immediately hit
    a ``BadZipFile`` instead of healing.
    """
    session = retrying_session()
    response = session.get(files_api, timeout=60)
    response.raise_for_status()
    files = response.json()
    matches = [f for f in files if f["name"] == file_name]
    if not matches:
        msg = f"{file_name!r} not in 4TU file list: {[f['name'] for f in files]}"
        raise FileNotFoundError(msg)
    url = matches[0]["download_url"]

    expected: int | None = None
    try:  # HEAD is a cheap size probe; if it fails, fall through to a full download.
        head = session.head(url, timeout=60, allow_redirects=True)
        head.raise_for_status()
        expected = int(head.headers.get("content-length", 0)) or None
    except requests.RequestException:
        expected = None

    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with session.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    if expected is not None and tmp.stat().st_size != expected:
        size = tmp.stat().st_size
        tmp.unlink()
        msg = f"download size mismatch for {file_name!r}: expected {expected}, got {size}"
        raise RuntimeError(msg)

    tmp.rename(dest)


def extract_member(archive_path: Path, member: str, dest: Path) -> None:
    """Extract a single member (The Hague's UHImax GeoTIFF) from the national zip.

    Skips re-extraction only when a cached dest already matches the member's
    uncompressed size, so a partial extract from an interrupted run is redone
    rather than treated as valid.
    """
    with zipfile.ZipFile(archive_path) as zf:
        info = zf.getinfo(member)
        if dest.exists() and dest.stat().st_size == info.file_size:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, dest.open("wb") as fh:
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
    download_file(args.files_api, args.archive_file_name, archive)

    extracted = args.cache_dir / Path(args.zip_member).name
    extract_member(archive, args.zip_member, extracted)

    ensure_rd_new(extracted, args.output)


if __name__ == "__main__":
    main()
