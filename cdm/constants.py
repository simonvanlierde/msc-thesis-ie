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

# Global parameters the model reads. read_global_parameters validates that the loaded
# configuration provides all of these, so a missing or renamed key fails at load with a
# clear message instead of a cryptic KeyError deep inside the model. Keep in sync when a
# new global_parameters["..."] access is added.
REQUIRED_GLOBAL_PARAMETERS = frozenset(
    {
        "T_sub_C",
        "T_thresh_C",
        "UHI_effect_day_C",
        "UHI_effect_night_C",
        "adp_intensity_cooling_equipment_kgSbeq_kg",
        "air_density",
        "air_heat_capacity",
        "alfa_i",
        "alfa_o",
        "building_stock_growth_office_old",
        "building_stock_growth_residential_new",
        "building_type_age_cutoff_yr",
        "building_type_height_cutoff_m",
        "carbon_intensity_EoL_kgCO2eq_kg",
        "carbon_intensity_electric_grid_kgCO2eq_kWh",
        "carbon_intensity_production_kgCO2eq_kg",
        "csi_intensity_cooling_equipment_kgSbeq_kg",
        "delta_P_solar_RoY",
        "delta_P_solar_summer",
        "delta_T_autumn_C",
        "delta_T_spring_C",
        "delta_T_summer_C",
        "delta_T_winter_C",
        "gwp_refrigerant_kgCO2eq_kg",
        "int_heat_gain_light_W_m2",
        "int_heat_gain_pp_W",
        "peak_cooling_percentile_cap",
        "people_density_office",
        "people_per_hh",
        "ventilation_efficiency",
        "weather_data_end_year",
        "weather_data_start_year",
        "weather_station",
    },
)
