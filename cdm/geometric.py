"""Geometric calculation functions used in the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

### Import packages
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from geopandas.geoseries import GeoSeries

MAX_AZIMUTH_DEGREES = 180  # Azimuths are measured in range 0-180 degrees

### Geometry functions


def azimuth_line(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the azimuth between two 2D points a and b."""
    angle = np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1]))
    if angle < 0:
        angle += 180
    return angle


def azimuth_rectangle(rectangle: GeoSeries) -> tuple[float, float, float]:
    """Calculate the azimuth, width, and length of a rotated rectangle.

    Args:
        rectangle (GeoSeries): A GeoSeries object representing a rotated rectangle.

    Returns:
        tuple[float, float, float]: The azimuth of the rotated rectangle in degrees, and its width and length in meters.
    """
    bbox = list(rectangle.exterior.coords)  # Get the coordinates of the rotated rectangle
    axis1 = math.dist(bbox[0], bbox[3])  # Calculate the length of the first axis
    axis2 = math.dist(bbox[0], bbox[1])  # Calculate the length of the second axis

    if axis1 <= axis2:  # Determine the azimuth based on the longest axis
        az = azimuth_line(bbox[0], bbox[1])
        width, length = axis1, axis2
    else:
        az = azimuth_line(bbox[0], bbox[3])
        width, length = axis2, axis1

    return (az, width, length)


def determine_orientation_class(azimuth: float) -> int:
    """Determine the orientation class of a building based on its azimuth.

    Args:
        azimuth (float): The azimuth of the building in degrees.

    Raises:
        ValueError: If the azimuth is not in range 0-180 degrees.

    Returns:
        int: The orientation class of the building, represented as integer (1: N-S, 2: NE-SW, 3: E-W, 4: NW-SE)
    """
    if not 0 <= azimuth <= MAX_AZIMUTH_DEGREES:
        msg = "Azimuth should be between 0 and 180 degrees"
        raise ValueError(msg)

    orientation_class_ranges = [22.5, 67.5, 112.5, 157.5]

    for i, upper_limit in enumerate(orientation_class_ranges):
        if azimuth <= upper_limit:
            return i + 1  # Orientation classes are 1-based

    return 1  # Default to orientation class 1 for values larger than 157.5


def calc_facade_area_per_orientation(building: pd.Series, orientation_class_int: int) -> np.ndarray:
    """Calculate the facade area per compass direction in m2.

    Args:
        building (pd.Series): The building row for which the facade area is calculated.
        orientation_class_int (int): The orientation class of the building, represented as integer (1: N-S, 2: NE-SW, 3: E-W, 4: NW-SE)

    Raises:
        ValueError: If the orientation class integer is not in range 1-4.

    Returns:
        np.ndarray: The facade area per compass direction in m2.
    """
    if orientation_class_int not in range(1, 5):
        msg = "orientation_class_int should be between 1 and 4"
        raise ValueError(msg)

    # Load the building attributes
    width = building["MBR_width_m"]  # The width (short side) of the building MBR in m
    length = building["MBR_length_m"]  # The length (long side) of the building MBR in m
    height = building["height_m"]  # The height of the building in m

    orientation_to_facade_lengths = {
        1: (width, 0, length, 0, width, 0, length, 0),
        2: (0, width, 0, length, 0, width, 0, length),
        3: (length, 0, width, 0, length, 0, width, 0),
        4: (0, length, 0, width, 0, length, 0, width),
    }

    return height * np.array(orientation_to_facade_lengths[orientation_class_int])


# Vectorised form of determine_orientation_class -> calc_facade_area_per_orientation, for the whole
# stock at once. Each of the 8 compass walls carries either the MBR width (W), the MBR length (L),
# or nothing, depending on the orientation class (rows 0-3 == classes 1-4). These two masks encode
# exactly the orientation_to_facade_lengths table above.
_ORIENTATION_CLASS_BOUNDS = (22.5, 67.5, 112.5, 157.5)
_WIDTH_PLACEMENT = np.array(
    [
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
    ],
    dtype=float,
)
_LENGTH_PLACEMENT = np.array(
    [
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
    ],
    dtype=float,
)


def calc_window_and_wall_areas_vectorised(buildings: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Whole-stock form of calc_window_and_wall_areas (one pass, no per-row Python).

    Returns the per-orientation window areas (one length-8 array per building), and the total
    window and wall areas as arrays aligned to ``buildings``.
    """
    azimuth = buildings["MBR_azimuth"].to_numpy()
    if ((azimuth < 0) | (azimuth > MAX_AZIMUTH_DEGREES)).any():
        msg = "Azimuth should be between 0 and 180 degrees"
        raise ValueError(msg)

    # Bucket each azimuth into orientation class 1-4; >157.5 deg wraps back to class 1 (N-S), matching
    # determine_orientation_class. `% 4` maps digitize's 0-4 onto the four 0-based class rows.
    class_index = np.digitize(azimuth, _ORIENTATION_CLASS_BOUNDS, right=True) % 4

    width = buildings["MBR_width_m"].to_numpy()
    length = buildings["MBR_length_m"].to_numpy()
    height = buildings["height_m"].to_numpy()

    facade_area = height[:, None] * (
        width[:, None] * _WIDTH_PLACEMENT[class_index] + length[:, None] * _LENGTH_PLACEMENT[class_index]
    )
    window_area_per_orientation = facade_area * buildings["f_window"].to_numpy()[:, None]
    wall_area_total = facade_area.sum(axis=1) * buildings["f_wall"].to_numpy()
    window_area_total = window_area_per_orientation.sum(axis=1)

    return list(window_area_per_orientation), window_area_total, wall_area_total


def calc_window_and_wall_areas(building: pd.Series) -> tuple[np.ndarray, float, float]:
    """Calculate the window and wall areas of a building in m2.

    Args:
        building (pd.Series): The building row for which the window and wall areas are calculated.

    Returns:
        tuple[np.ndarray, float, float]: The window area per compass direction in m2, and the total window and wall areas in m2.
    """
    orientation_class = determine_orientation_class(
        building["MBR_azimuth"],
    )  # Determine the orientation class of the building
    facade_area_per_orientation_m2 = calc_facade_area_per_orientation(
        building,
        orientation_class,
    )  # Calculate the facade area per compass direction in m2
    facade_area_total_m2 = facade_area_per_orientation_m2.sum()  # Calculate the total facade area in m2
    wall_area_total_m2 = facade_area_total_m2 * building["f_wall"]  # Calculate the total wall area in m2
    window_area_per_orientation_m2 = (
        facade_area_per_orientation_m2 * building["f_window"]
    )  # Calculate the window area per compass direction in m2
    window_area_total_m2 = window_area_per_orientation_m2.sum()  # Calculate the total window area in m2

    return window_area_per_orientation_m2, window_area_total_m2, wall_area_total_m2
