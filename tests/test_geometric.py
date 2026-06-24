"""Unit tests for geometric helper functions."""

import math

import numpy as np
import pandas as pd
import pytest

from functions.geometric import (
    azimuth_line,
    calc_facade_area_per_orientation,
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
