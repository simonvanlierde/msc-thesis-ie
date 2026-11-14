"""Check notebook paths without running the full scientific calculation."""

import ast
import json
import os
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from cdm.figures import get_indices_from_dates

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("notebook", ["main.ipynb", "gis.ipynb"])
def test_notebook_paths_use_selected_results_root(
    notebook: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "custom-results"))
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    trees = [ast.parse("".join(cell["source"])) for cell in cells if cell["cell_type"] == "code"]
    namespace: dict[str, object] = {
        "os": os,
        "Path": Path,
        "SCENARIO": "SQ",
        "SCENARIO_FUTURE": "2050_M",
        "BUILDING_SUBSET_NAME": "sample",
        "start_date": "2020-08-01",
        "end_date": "2020-08-07",
        "PARAMETERS_GLOBAL": {"weather_station": 330, "weather_data_start_year": 2020, "weather_data_end_year": 2020},
    }
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RESULTS_DIR" for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), notebook, "exec"), namespace)  # noqa: S102
    paths = []
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Constant, ast.JoinedStr)):
                continue
            # Only complete path expressions, not literal pieces inside f-strings.
            if isinstance(node, ast.Constant) and not isinstance(node.value, str):
                continue
            text = ast.unparse(node)
            if "results/" not in text and not (isinstance(node, ast.JoinedStr) and "RESULTS_DIR" in text):
                continue
            if isinstance(node, ast.Constant) and any(
                node is child
                for parent in ast.walk(tree)
                if isinstance(parent, ast.JoinedStr)
                for child in parent.values  # noqa: PD011 — ast.JoinedStr, not a pandas object
            ):
                continue
            value = eval(compile(ast.Expression(body=node), notebook, "eval"), namespace)  # noqa: S307
            paths.append(Path(value))
    assert paths
    assert all(path.is_relative_to(tmp_path / "custom-results") for path in paths), paths


def test_main_notebook_reads_declared_weather_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weather = tmp_path / "selected-weather.csv"
    monkeypatch.setenv("WEATHER_CSV", str(weather))
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    namespace: dict[str, object] = {
        "os": os,
        "Path": Path,
        "RESULTS_DIR": "results",
        "PARAMETERS_GLOBAL": {"weather_station": 330, "weather_data_start_year": 2018, "weather_data_end_year": 2022},
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "WEATHER_CSV_PATH" for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "main.ipynb", "exec"), namespace)  # noqa: S102
    assert namespace["WEATHER_CSV_PATH"] == weather


@pytest.mark.parametrize("year", [2020, 2022])
def test_main_notebook_rejects_weather_without_thesis_plot_dates(
    year: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weather = tmp_path / "weather.csv"
    pd.DataFrame({"date": pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")}).to_csv(weather, index=False)
    monkeypatch.setenv("WEATHER_CSV", str(weather))
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "raw_weather_data =" in "".join(cell["source"])
    )
    statements: list[ast.stmt] = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.If)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in {"WEATHER_CSV_PATH", "raw_weather_data"} for t in node.targets)
        )
    ]
    namespace: dict[str, object] = {
        "os": os,
        "pd": pd,
        "Path": Path,
        "RESULTS_DIR": "results",
        "PARAMETERS_GLOBAL": {"weather_station": 330, "weather_data_start_year": 2018, "weather_data_end_year": 2022},
    }
    expectation = pytest.raises(ValueError, match=r"requires.*2022") if year == 2020 else nullcontext()
    with expectation:
        exec(compile(ast.Module(body=statements, type_ignores=[]), "main.ipynb", "exec"), namespace)  # noqa: S102
        weather_data = namespace["raw_weather_data"]
        assert isinstance(weather_data, pd.DataFrame)
        get_indices_from_dates(weather_data, "2022-07-18", "2022-07-20")
