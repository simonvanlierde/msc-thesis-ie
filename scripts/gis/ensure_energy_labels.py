"""Download (or validate) the EP-Online energy-label export.

The EP-Online v5 API hands out a signed, short-lived download URL for the full
public label file. This script resolves that URL with the ``EP_ONLINE_API_KEY``
credential, streams the ZIP, and extracts the CSV to the path the GIS stage
expects. If the file is already present and looks valid, it is left untouched so
the ~1.5 GB export is not re-downloaded on every run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

DOWNLOAD_INFO_URL = "https://public.ep-online.nl/api/v5/Mutatiebestand/DownloadInfo"
_MAX_PREAMBLE_SCAN = 20  # preamble is 2 rows in practice; scan a few extra for drift


def _load_api_key() -> str:
    """Read EP_ONLINE_API_KEY from the environment, falling back to a local .env file."""
    key = os.environ.get("EP_ONLINE_API_KEY")
    if key:
        return key.strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "EP_ONLINE_API_KEY":
                return value.strip().strip('"').strip("'")
    msg = (
        "EP_ONLINE_API_KEY is not set. Add it to the environment or a .env file. "
        "Request a key at https://www.ep-online.nl/PublicData."
    )
    raise SystemExit(
        msg,
    )


def _looks_valid(labels: Path) -> bool:
    """Check the export header (past the key;value preamble) for the columns the model needs."""
    if not labels.exists() or labels.stat().st_size == 0:
        return False
    with labels.open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            lowered = line.lower()
            if "bagverblijfsobjectid" in lowered:
                return "energieklasse" in lowered
            if index > _MAX_PREAMBLE_SCAN:
                break
    return False


def _download_export(labels: Path) -> None:
    """Resolve the signed download URL, stream the ZIP, and extract the CSV member."""
    key = _load_api_key()
    info = requests.get(
        DOWNLOAD_INFO_URL,
        params={"fileType": "csv"},
        headers={"accept": "text/plain", "Authorization": key},
        timeout=120,
    )
    info.raise_for_status()
    download_url = info.json()["downloadUrl"]

    labels.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "ep_online.zip"
        with requests.get(download_url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)

        with zipfile.ZipFile(zip_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                msg = f"No CSV found in the EP-Online download. Members: {archive.namelist()}"
                raise SystemExit(msg)
            with archive.open(csv_members[0]) as source, labels.open("wb") as target:
                shutil.copyfileobj(source, target)


def main() -> None:
    """Ensure the EP-Online energy-label export exists and is usable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    labels = Path(args.labels)

    if _looks_valid(labels):
        return
    _download_export(labels)
    if not _looks_valid(labels):
        msg = f"Downloaded EP-Online export at {labels} is missing the expected columns."
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
