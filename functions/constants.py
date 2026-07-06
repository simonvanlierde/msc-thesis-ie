"""Shared constants for the cooling demand model.

@author: Simon van Lierde
"""

import os

# Hours in a (non-leap) year. The model drops Feb 29, so every modelled year is exactly this length.
HOURS_PER_YEAR = 8760

# Base directory for output figures, overridable so the pipeline can redirect them (e.g. under results/).
IMAGE_OUTPUT_DIR = os.environ.get("CDM_IMAGE_OUTPUT_DIR", "data/output/images")

# The eight window/facade compass directions, in the order used for both window areas and solar radiation.
SOLAR_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# Scenario identifiers, in reporting order.
SCENARIOS = ("SQ", "2030", "2050_L", "2050_M", "2050_H")

# Default percentile at which the peak cooling power demand is capped.
DEFAULT_CAP_PERCENTILE = 98
