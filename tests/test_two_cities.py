"""Resolve the production workflow twice without downloading or publishing data."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]


def _run(directory: Path, snakefile: str, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "snakemake", "--snakefile", snakefile, "--cores", "2", *arguments],
        cwd=directory,
        env={**os.environ, "XDG_CACHE_HOME": str(directory / ".cache")},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def _summary(directory: Path, snakefile: str) -> dict[str, str]:
    return {
        columns[0]: columns[2]
        for line in _run(directory, snakefile, "--summary").splitlines()
        if len(columns := line.split("\t")) == 6 and columns[0] != "output_file"
    }


@pytest.fixture
def workflow(tmp_path: Path) -> Path:
    for name in ("Snakefile", "uv.lock", "config", "workflow", "cdm", "scripts", "docs", "data/input/parameters"):
        source = ROOT / name
        destination = tmp_path / name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
        elif source.exists():
            shutil.copy2(source, destination)
    return tmp_path


def test_two_city_outputs_are_isolated(workflow: Path) -> None:
    products = _summary(workflow, "workflow/two_cities.smk")
    for city in ("oegstgeest", "voorschoten"):
        city_products = {path for path, rule in products.items() if rule.startswith(f"{city}_")}
        assert city_products
        assert all(
            path.startswith((f"results/two-cities/{city}/", f"data/raw/two-cities/{city}/")) for path in city_products
        ), city_products
        assert f"results/two-cities/{city}/CDM_results_SQ_full.csv" in city_products
        assert products[f"data/raw/two-cities/{city}/uhi/habib.tif"] == "extract_habib_member"
    assert not any("TheHague" in path or path.startswith("dashboard/") for path in products)
    assert list(products.values()).count("provide_ep_online_energy_labels") == 1
    assert products["data/raw/habib_uhi/UHI_NL.zip"] == "fetch_habib_archive"
    assert list(products.values()).count("fetch_habib_archive") == 1


def test_default_hague_outputs_remain_unchanged(workflow: Path) -> None:
    products = _summary(workflow, "Snakefile")
    assert "results/CDM_results_SQ_full.csv" in products
    assert "data/input/geodata/UHImax_habib_TheHague_5m.tif" in products
    assert not any(path.startswith("results/two-cities/") for path in products)


def test_each_city_extracts_its_own_habib_member(workflow: Path) -> None:
    archive = workflow / "data/raw/habib_uhi/UHI_NL.zip"
    archive.parent.mkdir(parents=True)
    targets = []
    with zipfile.ZipFile(archive, "w") as zipped:
        for slug, value in (("oegstgeest", 2), ("voorschoten", 4)):
            source = workflow / f"{slug}.tif"
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="float32",
                crs="EPSG:28992",
                transform=from_origin(90000, 460000, 5, 5),
            ) as raster:
                raster.write(np.full((2, 2), value, dtype="float32"), 1)
            profile = yaml.safe_load((workflow / f"config/cities/{slug}.yaml").read_text())
            expected_member = f"UHI_NL/0_primary_dataset/UHImax/UHImax_{slug.capitalize()}.tif"
            assert profile["habib_uhi"]["uhimax_zip_member"] == expected_member
            zipped.write(source, expected_member)
            targets.append(f"data/raw/two-cities/{slug}/uhi/habib.tif")
    _run(workflow, "workflow/two_cities.smk", *targets, "--allowed-rules", "extract_habib_member")
    for target, value in zip(targets, (2, 4), strict=True):
        with rasterio.open(workflow / target) as raster:
            np.testing.assert_array_equal(raster.read(1), np.full((2, 2), value))
            assert raster.crs.to_epsg() == 28992
    assert archive.exists()


def test_aggregate_uses_each_profile_subset_and_results_root(workflow: Path) -> None:
    profile = workflow / "config/cities/oegstgeest.yaml"
    settings = yaml.safe_load(profile.read_text())
    settings["results_dir"] = "results/alternate-oegstgeest"
    settings["building_subset"] = "sample"
    profile.write_text(yaml.safe_dump(settings))
    products = _summary(workflow, "workflow/two_cities.smk")
    assert "results/alternate-oegstgeest/CDM_results_SQ_sample.csv" in products
    assert "results/two-cities/voorschoten/CDM_results_SQ_full.csv" in products
    assert not any(path.startswith("results/two-cities/oegstgeest/") for path in products)
