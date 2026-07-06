"""Aggregation and building-stock scaling of the cooling demand model results.

@author: Simon van Lierde
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def scale_results_with_building_stock(
    buildings_agg: pd.DataFrame,
    global_parameters: dict[str, float],
    columns_to_scale: list[str],
) -> pd.DataFrame:
    """Scales the aggregated results of the cooling demand model with the building stock growth rates.

    Args:
        buildings_agg (pd.DataFrame): DataFrame containing the aggregated results of the cooling demand model for each building type and energy class.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        columns_to_scale (list[str]): The list of columns to scale.

    Returns:
        pd.DataFrame: The DataFrame containing the scaled aggregated results of the cooling demand model for each building type and energy class.
    """  # Unload the global parameters
    bs_scale_factor_residential = (
        1 + global_parameters["building_stock_growth_residential_new"]
    )  # Building stock growth rate of new residential buildings (type 1 and 3)
    bs_scale_factor_office = (
        1 + global_parameters["building_stock_growth_office_old"]
    )  # Building stock growth rate of old office buildings (type 6 and 8)

    # Multiply relevant columns with the residential building stock scale factor
    buildings_agg.loc[buildings_agg["building_type_int"].isin([1, 3]), columns_to_scale] *= bs_scale_factor_residential

    # Multiply relevant columns with the office building stock scale factor
    buildings_agg.loc[buildings_agg["building_type_int"].isin([6, 8]), columns_to_scale] *= bs_scale_factor_office

    return buildings_agg


def aggregate_results(
    buildings: pd.DataFrame,
    global_parameters: dict[str, float],
    groupby_columns: Sequence[str] = ("building_type_int", "building_type", "energy_class", "energy_class_int"),
    scale_with_building_stock: bool = True,
    include_time_series: bool = False,
) -> pd.DataFrame:
    """Aggregates the results of the cooling demand model to the building type and energy class level.

    Args:
        buildings (pd.DataFrame): The DataFrame containing the results of the cooling demand model for each building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        groupby_columns (tuple, optional): The tuple of columns to group by. Defaults to ("building_type_int", "building_type", "energy_class").
        scale_with_building_stock (bool, optional): Whether to scale the aggregated results with the building stock growth rates. Defaults to True.
        include_time_series (bool, optional): Whether to include the time series data in the aggregated results. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame containing the aggregated results of the cooling demand model for each building type and energy class.
    """

    # Create a aggregation function that calculates the weighted mean using total floor area as weights
    def weighted_mean(series: pd.Series) -> float:
        return np.average(series, weights=buildings.loc[series.index, "floor_area_total_m2"])

    # Construct the aggregations dictionary
    aggregations = {
        "id_BAG": "count",
        "construction_year": weighted_mean,
        "floor_area_ground_m2": "sum",
        "height_m": weighted_mean,
        "volume_m3": "sum",
        "energy_label_int": weighted_mean,
        "floor_area_total_m2": "sum",
        "number_of_residences": "sum",
        "population": "sum",
        "E_cooling_kWh": "sum",
        "E_cooling_capped_at_98th_percentile_kWh": "sum",
        "E_cooling_capped_at_98th_percentile_Wh_m2": weighted_mean,
        "P_cooling_peak_kW": "sum",
        "P_cooling_peak_98th_percentile_kW": "sum",
        "P_cooling_peak_98th_percentile_W_m2": weighted_mean,
        "electricity_use_kWh": "sum",
        "GHG_emissions_electricity_kgCO2eq": "sum",
        "GHG_emissions_refrigerant_leaks_kgCO2eq": "sum",
        "GHG_emissions_production_phase_kgCO2eq": "sum",
        "GHG_emissions_EoL_phase_kgCO2eq": "sum",
        "mass_cooling_equipment_kg": "sum",
        "ADP_kgSbeq": "sum",
        "CSI_kgSieq": "sum",
        "GHG_emissions_total_kgCO2eq": "sum",
        "electricity_use_intensity_kWh_m2": weighted_mean,
        "material_use_intensity_kg_m2": weighted_mean,
        "GHG_emissions_intensity_kgCO2eq_m2": weighted_mean,
        "avg_SEER_inv": weighted_mean,
        "total_MPR": weighted_mean,
        "avg_lifetime_cooling_technology_yr": weighted_mean,
    }

    if include_time_series:

        def sum_arrays(series_with_arrays: pd.Series) -> np.ndarray:
            return np.sum(np.vstack(list(series_with_arrays)), axis=0)

        # Add the time series aggregations to the aggregations dictionary
        time_series_aggregations = dict.fromkeys(buildings.filter(like="Q_").columns, sum_arrays)
        aggregations.update(time_series_aggregations)

    # Group by building type and energy class and aggregate the results
    buildings_agg = buildings.groupby(list(groupby_columns)).agg(aggregations)

    # Reset index
    buildings_agg = buildings_agg.reset_index()

    # Fetch columns that are summed, as these need to be scaled with the building stock growth rates
    sum_columns = [
        key
        for key, value in aggregations.items()
        if str(value).startswith("sum") or str(value).startswith("<function aggregate_results.<locals>.sum_arrays")
    ]

    if scale_with_building_stock:
        # Scale the aggregated results with the building stock growth rates
        buildings_agg = scale_results_with_building_stock(buildings_agg, global_parameters, sum_columns)

    return buildings_agg
