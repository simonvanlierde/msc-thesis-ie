"""Benchmarks for the building geometry functions (cdm/geometric.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdm.geometric import calc_window_and_wall_areas, calc_window_and_wall_areas_vectorised

if TYPE_CHECKING:
    import pandas as pd
    from pytest_codspeed import BenchmarkFixture


def test_calc_window_and_wall_areas_vectorised(benchmark: BenchmarkFixture, building_stock: pd.DataFrame) -> None:
    window_areas, window_total, wall_total = benchmark(calc_window_and_wall_areas_vectorised, building_stock)
    assert len(window_areas) == len(window_total) == len(wall_total) == len(building_stock)


def test_calc_window_and_wall_areas_per_row(benchmark: BenchmarkFixture, buildings: pd.DataFrame) -> None:
    rows = [row for _, row in buildings.iterrows()]

    def run() -> list:
        return [calc_window_and_wall_areas(row) for row in rows]

    result = benchmark(run)
    assert len(result) == len(buildings)
