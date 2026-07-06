"""Shared constants for the cooling demand model.

@author: Simon van Lierde
"""

# Hours in a (non-leap) year. The model drops Feb 29, so every modelled year is exactly this length.
HOURS_PER_YEAR = 8760

# The eight window/facade compass directions, in the order used for both window areas and solar radiation.
SOLAR_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# Scenario identifiers, in reporting order.
SCENARIOS = ("SQ", "2030", "2050_L", "2050_M", "2050_H")

# Default percentile at which the peak cooling power demand is capped.
DEFAULT_CAP_PERCENTILE = 98
