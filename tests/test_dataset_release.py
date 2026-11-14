"""The release allowlist excludes restricted data even if the source gains columns."""

import hashlib
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest

from scripts.build_dataset_release import FIELDS, build_release, main


def test_release_excludes_private_columns_and_verifies_archive(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source.gpkg"
    buildings = gpd.read_file(Path(__file__).parent / "fixtures/bag_2000.gpkg", rows=3)
    buildings["energy_label"] = "do not distribute"
    buildings["energy_label_int"] = 8
    buildings["activity_uuid"] = "private identifier"
    buildings["impact_score"] = 42.0
    buildings.to_file(source, layer="BAG_buildings_full", driver="GPKG")
    output = tmp_path / "release"
    archive = build_release(source, output)
    restored = gpd.read_parquet(output / "buildings.parquet")
    assert list(restored.columns) == [*FIELDS, "geometry"]
    assert len(restored) == 3
    assert json.loads((output / "inventory.json").read_text())["rows"] == 3
    with zipfile.ZipFile(archive) as handle:
        for line in handle.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split(" *")
            assert hashlib.sha256(handle.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        build_release(source, output)


def test_release_rejects_missing_columns_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "incomplete.gpkg"
    buildings = gpd.read_file(Path(__file__).parent / "fixtures/bag_2000.gpkg", rows=1)
    buildings.drop(columns="UHI_max_C").to_file(source, layer="BAG_buildings_full", driver="GPKG")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="UHI_max_C"):
        build_release(source, output)
    assert not output.exists()


def test_release_preserves_dotted_name_and_existing_archive(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source.gpkg"
    buildings = gpd.read_file(Path(__file__).parent / "fixtures/bag_2000.gpkg", rows=1)
    buildings.to_file(source, layer="BAG_buildings_full", driver="GPKG")
    output = tmp_path / "dataset-v0.1"
    archive = tmp_path / "dataset-v0.1.zip"
    archive.write_bytes(b"existing deposit")
    with pytest.raises(FileExistsError):
        build_release(source, output)
    assert archive.read_bytes() == b"existing deposit"
    assert not output.exists()
    archive.unlink()
    assert build_release(source, output) == archive


def test_release_records_invalid_geometry_exclusions(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source.gpkg"
    buildings = gpd.read_file(Path(__file__).parent / "fixtures/bag_2000.gpkg", rows=2)
    buildings.loc[0, "geometry"] = None
    buildings.to_file(source, layer="BAG_buildings_full", driver="GPKG")
    output = tmp_path / "release"
    build_release(source, output)
    inventory = json.loads((output / "inventory.json").read_text())
    assert inventory["source_rows"] == 2
    assert inventory["rows"] == 1
    assert inventory["excluded_invalid_geometry"] == 1
    review = (tmp_path / "release-review/excluded-records.csv").read_text()
    assert buildings.loc[0, "id_BAG"] in review
    assert "invalid or empty geometry" in review


def test_cli_missing_source_reports_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "release"
    monkeypatch.setattr(
        "sys.argv",
        ["build_dataset_release.py", "--source", str(tmp_path / "missing.gpkg"), "--output", str(output)],
    )
    assert main() == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert not output.exists()
