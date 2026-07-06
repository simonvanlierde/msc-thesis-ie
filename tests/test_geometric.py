"""Unit tests for geometric helper functions."""

import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from cdm.geometric import (
    azimuth_line,
    azimuth_rectangle,
    calc_facade_area_per_orientation,
    calc_window_and_wall_areas,
    determine_orientation_class,
    dist,
)


def test_dist_pythagorean_triple() -> None:
    assert dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_dist_is_symmetric() -> None:
    a, b = (1.0, 2.0), (4.0, 6.0)
    assert dist(a, b) == pytest.approx(dist(b, a))


def test_azimuth_line_due_north_is_zero() -> None:
    # Point b directly north of a (same x, larger y) -> azimuth 0 degrees.
    assert azimuth_line((0.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_azimuth_line_due_east_is_ninety() -> None:
    assert azimuth_line((0.0, 0.0), (1.0, 0.0)) == pytest.approx(90.0)


def test_azimuth_line_is_folded_into_zero_to_180() -> None:
    # A line pointing south-west must be reported on the same 0-180 axis as north-east.
    angle = azimuth_line((0.0, 0.0), (-1.0, -1.0))
    assert 0.0 <= angle <= 180.0


@pytest.mark.parametrize(
    ("azimuth", "expected_class"),
    [(0.0, 1), (22.5, 1), (45.0, 2), (90.0, 3), (135.0, 4), (180.0, 1)],
)
def test_determine_orientation_class(azimuth: float, expected_class: int) -> None:
    assert determine_orientation_class(azimuth) == expected_class


@pytest.mark.parametrize("azimuth", [-1.0, 180.1, 360.0])
def test_determine_orientation_class_rejects_out_of_range(azimuth: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 180"):
        determine_orientation_class(azimuth)


def test_calc_facade_area_per_orientation_scales_with_height() -> None:
    building = pd.Series({"MBR_width_m": 10.0, "MBR_length_m": 20.0, "height_m": 3.0})
    facade = calc_facade_area_per_orientation(building, orientation_class_int=1)
    # Class 1 distributes (width, 0, length, 0, width, 0, length, 0) * height.
    expected = 3.0 * np.array([10.0, 0, 20.0, 0, 10.0, 0, 20.0, 0])
    assert np.allclose(facade, expected)
    # Total facade area equals perimeter * height = 2*(w+l)*h.
    assert facade.sum() == pytest.approx(2 * (10.0 + 20.0) * 3.0)


@pytest.mark.parametrize("bad_class", [0, 5, -1])
def test_calc_facade_area_per_orientation_rejects_bad_class(bad_class: int) -> None:
    building = pd.Series({"MBR_width_m": 10.0, "MBR_length_m": 20.0, "height_m": 3.0})
    with pytest.raises(ValueError, match="between 1 and 4"):
        calc_facade_area_per_orientation(building, orientation_class_int=bad_class)


def test_dist_matches_math_hypot() -> None:
    assert dist((2.0, -1.0), (-2.0, 2.0)) == pytest.approx(math.hypot(-4.0, 3.0))


def test_azimuth_rectangle_for_axis_aligned_rectangle() -> None:
    # A 20 (east-west) by 10 (north-south) axis-aligned rectangle.
    rectangle = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)])
    azimuth, width, length = azimuth_rectangle(rectangle)

    # The long axis runs due east, the short side is 10 m and the long side is 20 m.
    assert azimuth == pytest.approx(90.0)
    assert width == pytest.approx(10.0)
    assert length == pytest.approx(20.0)


def test_calc_window_and_wall_areas_splits_facade_by_factor() -> None:
    building = pd.Series(
        {
            "MBR_azimuth": 30.0,  # -> orientation class 2 (NE-SW)
            "MBR_width_m": 10.0,
            "MBR_length_m": 20.0,
            "height_m": 15.0,
            "f_wall": 0.7,
            "f_window": 0.3,
        },
    )
    window_per_orientation, window_total, wall_total = calc_window_and_wall_areas(building)

    total_facade = 2 * (10.0 + 20.0) * 15.0  # perimeter * height
    assert window_total == pytest.approx(total_facade * 0.3)
    assert wall_total == pytest.approx(total_facade * 0.7)
    assert window_per_orientation.sum() == pytest.approx(window_total)
