"""Exercise production DAG invalidation with metadata-only, offline products."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_JOBS = {
    "prepare_bag_geodata",
    "add_uhi_to_buildings",
    "thermodynamic_model",
    "lca",
    "scenario_overview_figure",
    "all",
}


def _snakemake(directory: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "snakemake", "--configfile", "config/smoke.yaml", "--cores", "1", *args],
        cwd=directory,
        env={**os.environ, "XDG_CACHE_HOME": str(directory / ".cache")},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


@pytest.fixture
def workflow(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    target = getattr(request, "param", "all")
    for name in (
        "Snakefile",
        "uv.lock",
        "config",
        "cdm",
        "scripts",
        "docs",
        "data/input/parameters",
        "main.ipynb",
        "gis.ipynb",
        "dashboard/scripts",
    ):
        source = ROOT / name
        destination = tmp_path / name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, destination)
    # Only output paths come from --summary: no shell command or remote data is executed.
    summary = _snakemake(tmp_path, target, "--summary")
    for line in summary.splitlines():
        columns = line.split("\t")
        if len(columns) != 6 or columns[0] == "output_file":
            continue
        path = tmp_path / columns[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name in {"height_tiles", "main", "gis"}:
            path.mkdir(exist_ok=True)
        else:
            path.touch()
    _snakemake(tmp_path, target, "--touch", "--forceall")
    assert "Nothing to be done" in _snakemake(tmp_path, target, "--dry-run")
    return tmp_path


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (
            "scripts/gis/fetch_city_boundary.py",
            MODEL_JOBS
            | {
                "fetch_city_boundary",
                "fetch_bag_residences",
                "discover_pdok_3d_height_tiles",
                "download_pdok_3d_height_tiles",
            },
        ),
        ("cdm/thermodynamic.py", {"thermodynamic_model", "lca", "scenario_overview_figure", "all"}),
        ("cdm/geometric.py", MODEL_JOBS),
        (
            "data/input/parameters/raw_weather_data_2018_2022_HvH.csv",
            {"fetch_weather", "thermodynamic_model", "lca", "scenario_overview_figure", "all"},
        ),
        (
            "uv.lock",
            MODEL_JOBS
            | {
                "fetch_city_boundary",
                "fetch_bag_residences",
                "discover_pdok_3d_height_tiles",
                "download_pdok_3d_height_tiles",
                "fetch_uhi_habib",
                "provide_ep_online_energy_labels",
                "fetch_weather",
            },
        ),
        (
            "config/smoke.yaml",
            {"add_uhi_to_buildings", "thermodynamic_model", "lca", "scenario_overview_figure", "all"},
        ),
        (
            "data/input/parameters/presence_load_factors.csv",
            {"thermodynamic_model", "lca", "scenario_overview_figure", "all"},
        ),
    ],
)
def test_reruns_after_dependency_change(workflow: Path, changed: str, expected: set[str]) -> None:
    path = workflow / changed
    with path.open("a") as stream:
        stream.write("\nuhi_buffer_m: 45\n" if changed.endswith(".yaml") else "\n# changed dependency\n")
    output = _snakemake(workflow, "--dry-run")
    jobs = set(re.findall(r"^(?:local)?rule (\w+):", output, re.MULTILINE))
    assert jobs == expected, output


@pytest.mark.parametrize("workflow", ["run_main_notebook", "dashboard_data"], indirect=True)
@pytest.mark.parametrize("parameter", ["presence_load_factors.csv", "multidirectional_solar_radiation_fractions.csv"])
def test_optional_targets_track_load_inputs(workflow: Path, parameter: str, request: pytest.FixtureRequest) -> None:
    target = request.node.callspec.params["workflow"]
    with (workflow / "data/input/parameters" / parameter).open("a") as stream:
        stream.write("\n# changed dependency\n")
    output = _snakemake(workflow, target, "--dry-run")
    assert re.search(rf"^(?:local)?rule {target}:", output, re.MULTILINE), output


@pytest.mark.parametrize("workflow", ["run_main_notebook", "run_gis_notebook"], indirect=True)
def test_notebooks_receive_selected_results_root(workflow: Path, request: pytest.FixtureRequest) -> None:
    target = request.node.callspec.params["workflow"]
    output = _snakemake(workflow, target, "--dry-run", "--force", "--printshellcmds")
    assert "RESULTS_DIR=results/smoke" in output
    if target == "run_main_notebook":
        assert "WEATHER_CSV=results/smoke/weather/knmi_330_2020_2020.csv" in output
