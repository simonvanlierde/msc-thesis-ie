"""Unit tests for the representative building subsample."""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subsample_buildings import STRATIFY_COLUMNS, draw_representative_sample


def _make_stock(n_common: int, n_rare: int) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """One large stratum and one single-building stratum, sharing an index."""
    types = [1] * n_common + [8]
    classes = [3] * n_common + [1]
    total = n_common + n_rare
    buildings = gpd.GeoDataFrame(
        {"payload": range(total)},
        geometry=[Point(i, i) for i in range(total)],
        crs="EPSG:28992",
    )
    strata = buildings.assign(building_type_int=types, energy_class_int=classes)
    return buildings, strata


def test_sample_keeps_input_schema_and_every_stratum() -> None:
    buildings, strata = _make_stock(n_common=500, n_rare=1)

    sample = draw_representative_sample(buildings, strata, sample_size=50, seed=0)

    # Schema is exactly the input's: the classification columns used for grouping never leak out.
    assert list(sample.columns) == list(buildings.columns)
    assert not set(STRATIFY_COLUMNS) & set(sample.columns)
    # The lone rare building survives despite rounding to ~0 of its proportional share.
    assert (sample["payload"] == 500).sum() == 1


def test_sample_is_deterministic_given_the_seed() -> None:
    buildings, strata = _make_stock(n_common=500, n_rare=1)

    first = draw_representative_sample(buildings, strata, sample_size=50, seed=7)
    second = draw_representative_sample(buildings, strata, sample_size=50, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_sample_is_proportional_to_the_stock() -> None:
    # Two equal strata -> a balanced sample should split roughly evenly.
    buildings = gpd.GeoDataFrame(
        {"payload": range(400)},
        geometry=[Point(i, i) for i in range(400)],
        crs="EPSG:28992",
    )
    strata = buildings.assign(
        building_type_int=[1] * 200 + [2] * 200,
        energy_class_int=[3] * 400,
    )

    sample = draw_representative_sample(buildings, strata, sample_size=40, seed=0)

    # 200/400 * 40 = 20 per stratum; recover the stratum via payload ranges.
    assert (sample["payload"] < 200).sum() == 20
    assert (sample["payload"] >= 200).sum() == 20
