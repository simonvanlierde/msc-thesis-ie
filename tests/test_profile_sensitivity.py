import json
import subprocess
import sys
from typing import TYPE_CHECKING

import geopandas as gpd
import pytest

from scripts import profile_sensitivity

if TYPE_CHECKING:
    from pathlib import Path


def test_empty_workload_is_rejected_before_writing(tmp_path: Path) -> None:
    profile = tmp_path / "invalid.prof"
    result = subprocess.run(  # noqa: S603 - Fixed interpreter and module; paths come from pytest.
        [
            sys.executable,
            "-m",
            "scripts.profile_sensitivity",
            "--source",
            "missing.gpkg",
            "--profile",
            str(profile),
            "sensitivity",
            "--steps",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--steps" in result.stderr
    assert not result.stdout
    assert not profile.exists()


def test_missing_source_uses_io_exit_code(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603 - Fixed interpreter and module; paths come from pytest.
        [
            sys.executable,
            "-m",
            "scripts.profile_sensitivity",
            "--source",
            str(tmp_path / "missing.gpkg"),
            "--profile",
            str(tmp_path / "missing.prof"),
            "sensitivity",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 3
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("variable", ["T_thresh_C", "SEER"])
def test_real_sensitivity_profile_is_offline_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variable: str,
) -> None:
    buildings = gpd.read_file("tests/fixtures/bag_2000.gpkg", layer="buildings").head(4)
    buildings["energy_label"] = "A"
    buildings["energy_label_int"] = 1
    source = tmp_path / "stock.gpkg"
    buildings.to_file(source, layer="BAG_buildings_full", driver="GPKG")
    profile = tmp_path / "case.prof"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--source",
            str(source),
            "--profile",
            str(profile),
            "sensitivity",
            "--sample",
            "2",
            "--steps",
            "2",
            "--variable",
            variable,
        ],
    )
    assert profile_sensitivity.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["workload"]["buildings"] == 2
    assert report["workload"]["weather_hours"] == 8760
    assert report["workload"]["steps"] == 2
    assert report["workload"]["variable"] == variable
    assert report["peak_rss_mib"] > 0
    assert report["wall_seconds"] > 0
    assert profile.stat().st_size > 0
    # A second run must preserve the original profile instead of truncating it.
    original = profile.read_bytes()
    assert profile_sensitivity.main() == 3
    assert profile.read_bytes() == original


@pytest.mark.parametrize("name", ["release", "release.v0.1"])
def test_profile_cannot_overwrite_measured_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    output = tmp_path / name
    archive = tmp_path / f"{name}.zip"
    buildings = gpd.read_file("tests/fixtures/bag_2000.gpkg", layer="buildings").head(2)
    source = tmp_path / "stock.gpkg"
    buildings.to_file(source, layer="BAG_buildings_full", driver="GPKG")
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile", "--source", str(source), "--profile", str(archive), "package", "--output", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        profile_sensitivity.main()
    assert error.value.code == 2
    assert not output.exists()
    assert not archive.exists()
