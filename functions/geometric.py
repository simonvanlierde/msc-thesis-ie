"""Geometric calculation functions used in the cooling demand model.

@author: Simon van Lierde
"""

### Import packages


import math

import numpy as np
import pandas as pd
from geopandas.geoseries import GeoSeries

### Geometry functions


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the distance between two 2D points a and b."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


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
        tuple[float, float, float]: The azimuth of th rotated rectangle in degrees, and the width of the rotated rectangle in meters.
    """
    """Calculate the azimuth of a rotated rectangle in degrees."""
    bbox = list(rectangle.exterior.coords)  # Get the coordinates of the rotated rectangle
    axis1 = dist(bbox[0], bbox[3])  # Calculate the length of the first axis
    axis2 = dist(bbox[0], bbox[1])  # Calculate the length of the second axis

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
    if not 0 <= azimuth <= 180:
        raise ValueError("Azimuth should be between 0 and 180 degrees")

    orientation_class_ranges = [22.5, 67.5, 112.5, 157.5]

    for i, upper_limit in enumerate(orientation_class_ranges):
        if azimuth <= upper_limit:
            return i + 1  # Orientation classes are 1-based

    return 1  # Default to orientation class 1 for values larger than 157.5


def calc_facade_area_per_orientation(building: pd.Series, orientation_class_int: int) -> np.array:
    """Calculate the facade area per compass direction in m2.

    Args:
        building (pd.Series): The building row for which the facade area is calculated.
        orientation_class_int (int): The orientation class of the building, represented as integer (1: N-S, 2: NE-SW, 3: E-W, 4: NW-SE)

    Raises:
        ValueError: If the orientation class integer is not in range 1-4.

    Returns:
        np.array: The facade area per compass direction in m2.
    """
    if orientation_class_int not in range(1, 5):
        raise ValueError("orientation_class_int should be between 1 and 4")

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


def calc_window_and_wall_areas(building: pd.Series) -> tuple[int, np.array, float, float]:
    """Calculate the window and wall areas of a building in m2.

    Args:
        building (pd.Series): The building row for which the window and wall areas are calculated.

    Returns:
        tuple[int, np.array, float, float]: The orientation class of the building, the window area per compass direction in m2, and the total window and wall areas in m2
    """
    orientation_class = determine_orientation_class(building["MBR_azimuth"])  # Determine the orientation class of the building
    facade_area_per_orientation_m2 = calc_facade_area_per_orientation(
        building,
        orientation_class,
    )  # Calculate the facade area per compass direction in m2
    facade_area_total_m2 = facade_area_per_orientation_m2.sum()  # Calculate the total facade area in m2
    wall_area_total_m2 = facade_area_total_m2 * building["f_wall"]  # Calculate the total wall area in m2
    window_area_per_orientation_m2 = facade_area_per_orientation_m2 * building["f_window"]  # Calculate the window area per compass direction in m2
    window_area_total_m2 = window_area_per_orientation_m2.sum()  # Calculate the total window area in m2

    return window_area_per_orientation_m2, window_area_total_m2, wall_area_total_m2
