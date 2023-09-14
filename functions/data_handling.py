"""Data handling functions used in the cooling demand model.

@author: Simon van Lierde
"""

# Import packages
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from functions.geometric import calc_window_and_wall_areas

# Data reading functions


def read_buildings(buildings_path: Path, layer_name: str = "BAG_buildings") -> gpd.GeoDataFrame:
    """Reads the buildings for which the cooling demand is calculated from a GeoPackage file.

    Args:
        buildings_path (Path): The path to the GeoPackage file containing the buildings for which the cooling demand is calculated.
        layer_name (str, optional): The name of the layer in the GeoPackage file containing the buildings for which the cooling demand is calculated. Defaults to "BAG_buildings".

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame containing the buildings for which the cooling demand is calculated.
    """
    return (
        gpd.read_file(filename=buildings_path, layer=layer_name)  # Read the building data
        .loc[lambda df: df["end_use"].isin(["woonfunctie", "kantoorfunctie", "residential", "office"])]  # Filter by end use
        .replace({"woonfunctie": "residential", "kantoorfunctie": "office"})  # Translate end use from Dutch to English
        .loc[lambda df: df["status"] == "Pand in gebruik"]  # Keep only in-use buildings, i.e. drop those not built yet or already demolished
        .dropna(subset=["energy_label"])  # Drop buildings without energy label as we cannot determine the cooling demand without it
    )


def read_global_parameters(global_parameters_path: Path) -> dict[str, float]:
    """Reads the global parameters for the cooling demand model from a csv file.

    Args:
        global_parameters_path (Path): The path to the csv file containing the global parameters for the cooling demand model.

    Returns:
        dict[str, float]: The dictionary containing the global parameters for the cooling demand model.
    """
    global_parameters = {}  # Dictionary containing the global parameters for the cooling demand model
    with open(global_parameters_path) as csv_file:
        reader = csv.DictReader(csv_file)
        global_parameters = {
            row["parameter"]: json.loads(row["value"]) if row["parameter"] == "energy_class_ranges" else float(row["value"]) for row in reader
        }  # Convert the string value to a list of lists for the energy label class ranges and to a float for the other parameters

    return global_parameters


def read_parameter_specific_data(parameters_path: Path) -> list[dict[str, float]]:
    """Reads the building, energy label or cooling technology parameters for the cooling demand model from a csv file.

    Args:
        parameters_path (Path): The path to the csv file containing the parameters for the cooling demand model.

    Returns:
        list[dict[str, float]]: The list of dictionaries containing the parameters for the cooling demand model.
    """
    with open(parameters_path) as csv_file:
        reader = csv.DictReader(csv_file)
        parameter_specific_data = [
            {
                key: int(value)
                if i == 0  # Convert the value of the first parameter (which contains the building type, energy class or cooling technology in integer representation) to integer
                else str(value)
                if i == 1  # Convert the value of the second parameter (which contains the building type, energy class or cooling technology name) to string
                else json.loads(value)
                if key.startswith("energy_labels_included_")  #  Convert energy label lists to a list
                else float(value)  # Convert all other parameter values to float
                for i, (key, value) in enumerate(row.items())
            }
            for row in reader
        ]  # Add the parameter and value to the dictionary for each row

    return parameter_specific_data


# NOTE: The following function is not used in the current version of the model, but is kept for archival sake
def read_time_series_from_csv(time_series_path: Path, global_parameters: dict[str, float]) -> dict[str, np.ndarray]:
    """Reads the time series data from a csv file and converts it to a dictionary of numpy arrays.

    Args:
        time_series_path (Path): The path to the csv file containing the time series data.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        dict[str, np.ndarray]: The dictionary containing the time series data.
    """
    # Read the time series data
    time_series_df = pd.read_csv(time_series_path)

    # Convert each column of the time series DataFrame to a numpy array and store them in a dictionary
    time_series = {col: time_series_df[col].to_numpy() for col in time_series_df.columns}

    # Add a key to the time series dictionary with the difference between the inside cooling threshold temperature and the outside air temperature in °C
    time_series["T_outdoor_minus_indoor_C"] = time_series["T_outdoor_C"] - global_parameters["T_thresh_C"]

    return time_series


# Building type parameter assignment functions


def calculate_building_population(building: pd.Series, global_parameters: dict[str, float]) -> float:
    """Calculate the building population based on the building row and global parameters.

    Args:
        building (pd.Series): The building row for which the population is calculated.
        global_parameters (dict): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        float: The calculated building population.
    """
    if building["end_use"] == "office":
        population = building["floor_area_total_m2"] * global_parameters["people_density_office"]
    else:
        population = building["number_of_residences"] * global_parameters["people_per_hh"]

    return population


def determine_building_type(
    building: pd.Series,
    global_parameters: dict[str, float],
) -> int:
    """Determines the building type of a building based on the building parameters.

    Args:
        building (pd.Series): The building row for which the building type is determined.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        int: The building type in integer representation
    """
    # Unpack building parameters
    end_use = building["end_use"]
    height_m = building["height_m"]
    construction_year = building["construction_year"]

    #  Unpack global parameters
    building_type_height_cutoff_m = global_parameters["building_type_height_cutoff_m"]
    building_type_age_cutoff_yr = global_parameters["building_type_age_cutoff_yr"]

    # Determine the end use binary, giving 0 for residential and 1 for office
    end_use_binary = 0 if end_use == "residential" else 1

    # Determine the low-rise binary, giving 0 for high-rise and 1 for low-rise
    lowrise_binary = 0 if height_m > building_type_height_cutoff_m else 1

    # Determine the age binary, giving 0 for new  and 1 for old
    age_binary = 0 if construction_year > building_type_age_cutoff_yr else 1

    # Determine the building type based on the three binary columns
    building_type_binary = str(end_use_binary) + str(lowrise_binary) + str(age_binary)

    # Convert building_type_binary to decimal integer between 1 and 8
    return int(building_type_binary, 2) + 1


def assign_building_type_parameters(
    building: pd.Series,
    building_parameters: list[dict[str, float]],
) -> pd.Series:
    """Assigns the building type-specific parameters to a building based on the building type.

    Args:
        building (pd.Series): The building row for which the building-specific parameters are assigned.
        building_parameters (list[dict[str, float]]): The list containing the building parameter dictionaries for the cooling demand model.

    Returns:
        pd.Series: The series containing the building type-specific parameters corresponding to the building type.
    """
    building_type_int = building["building_type_int"]
    parameters = building_parameters[building_type_int - 1]
    return pd.Series(parameters)


def add_building_type_data_to_buildings(
    df_buildings: pd.DataFrame,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
) -> pd.DataFrame:
    """Adds the energy class data to the buildings DataFrame.

    Args:
        df_buildings (pd.DataFrame): The DataFrame containing the buildings for which the energy class data is added.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        building_type_parameters (list[dict[str, float]]): The list containing the building type dependant parameter dictionaries for the cooling demand model.

    Returns:
        pd.DataFrame: The DataFrame containing the buildings with the energy class dependent data added.
    """
    # Determine the building type (in integer representation)
    df_buildings["building_type_int"] = df_buildings.apply(determine_building_type, args=(global_parameters,), axis=1)

    # Determine the building-type specific parameters based on the building type
    df_building_type_parameters = df_buildings.apply(assign_building_type_parameters, args=(building_type_parameters,), axis=1).drop(
        columns=["building_type_int"],
    )
    df_buildings[df_building_type_parameters.columns] = df_building_type_parameters

    return df_buildings


# Energy class parameter assignment functions


def determine_energy_label_to_class_mappings(
    energy_class_parameters: dict[str, float],
) -> dict[int, int]:
    """Determines the mapping from energy label to energy class.

    Args:
        energy_class_parameters (dict[str, float]): The dictionary containing the energy label class data

    Returns:
        dict[int, int]: The dictionaries containing the mappings from energy label to energy class, for both residential and office buildings.
    """
    # Convert the energy class parameters to a DataFrame
    df_energy_class_parameters = pd.DataFrame(energy_class_parameters)

    # Construct the energy label to energy class mappings
    residential_mapping = {label: row["energy_class_int"] for _, row in df_energy_class_parameters.iterrows() for label in row["energy_labels_included_residential"]}
    office_mapping = {label: row["energy_class_int"] for _, row in df_energy_class_parameters.iterrows() for label in row["energy_labels_included_office"]}

    return residential_mapping, office_mapping


def assign_energy_class_parameters(
    building: pd.Series,
    energy_class_parameters: list[dict[str, float]],
) -> pd.Series:
    """Assigns the energy class-specific parameters to a building based on the energy class.

    Args:
        building (pd.Series): The building row for which the energy class-specific parameters are assigned.
        energy_class_parameters (list[dict[str, float]]): The list containing the energy class dependent parameter dictionaries for the cooling demand model.

    Returns:
        pd.Series: The series containing the energy class-specific parameters corresponding to the building type.
    """
    energy_class = building["energy_class_int"]
    parameters = energy_class_parameters[energy_class - 1]
    return pd.Series(parameters)


def add_energy_class_data_to_buildings(
    df_buildings: pd.DataFrame,
    energy_class_parameters: list[dict[str, float]],
) -> pd.DataFrame:
    """Adds the energy class data to the buildings DataFrame.

    Args:
        df_buildings (pd.DataFrame): The DataFrame containing the buildings for which the energy class data is added.
        energy_class_parameters (list[dict[str, float]]): The list containing the energy class parameter dictionaries for the cooling demand model.

    Returns:
        pd.DataFrame: The DataFrame containing the buildings with the energy class dependent data added.
    """
    # Create energy label to energy class mappings for residential and office buildings
    energy_label_to_class_mapping_residential, energy_label_to_class_mapping_office = determine_energy_label_to_class_mappings(
        energy_class_parameters,
    )

    # Determine energy label class based on the energy label, using the right mapping for residential and office buildings
    df_buildings["energy_class_int"] = df_buildings.apply(
        lambda row: energy_label_to_class_mapping_residential[int(row["energy_label_int"])]
        if row["end_use"] == "residential"
        else energy_label_to_class_mapping_office[int(row["energy_label_int"])],
        axis=1,
    )

    # Determine the energy class-specific parameters based on the energy class
    df_energy_class_parameters = df_buildings.apply(assign_energy_class_parameters, args=(energy_class_parameters,), axis=1).drop(
        columns=["energy_class_int"],
    )
    df_buildings[df_energy_class_parameters.columns] = df_energy_class_parameters

    return df_buildings


# Derivative parameter assignment functions


def add_derived_parameters_to_buildings(
    df_buildings: pd.DataFrame,
    global_parameters: dict[str, float],
) -> pd.DataFrame:
    """Adds the parameters derived from the primary parameters to the buildings DataFrame.

    Args:
        df_buildings (pd.DataFrame): The DataFrame containing the buildings for which the derived parameters are added.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        pd.DataFrame: The DataFrame containing the buildings with the derived parameters added.
    """
    # Determine the building population
    df_buildings["population"] = df_buildings.apply(calculate_building_population, args=(global_parameters,), axis=1)

    # Determine the wall and window areas
    (
        df_buildings["window_area_per_orientation_m2"],
        df_buildings["window_area_total_m2"],
        df_buildings["wall_area_total_m2"],
    ) = zip(*df_buildings.apply(calc_window_and_wall_areas, axis=1), strict=True)

    # Determine the total market penetration rate of cooling equipment
    df_buildings["total_MPR"] = df_buildings[[col for col in df_buildings.columns if col.startswith("cooling_technology_share")]].sum(axis=1)

    return df_buildings


# Cooling technology parameter assignment functions


def add_cooling_technology_data_to_buildings(buildings: pd.DataFrame, cooling_technologies: pd.DataFrame) -> pd.DataFrame:
    """Assigns the cooling technology-specific parameters to the buildings DataFrame based on the cooling technology mix of each building.

    Args:
        buildings (pd.DataFrame): The buildings DataFrame for which the cooling technology-specific parameters are assigned.
        cooling_technologies (pd.DataFrame): The DataFrame containing the cooling technology-specific parameters for the cooling demand model.

    Returns:
        pd.DataFrame: Updated buildings DataFrame with weighted average cooling technology-dependent parameters.
    """
    # Reshape the cooling technologies DataFrame and drop unnecessary columns
    cooling_tech_reshaped = cooling_technologies.set_index("cooling_technology").drop(
        columns=["cooling_technology_int", "SEER", "refrigerant_density_kg_kW"],
    )

    # Compute weighted averages for all cooling technology dependent parameters
    weighted_parameters_df = pd.DataFrame(index=buildings.index)  # Initialize DataFrame to store the weighted parameters
    for share_column in buildings.filter(like="cooling_technology_share").columns:  # Loop over all cooling technology share columns
        tech_name = share_column.split("cooling_technology_share_")[-1]  # Extract the cooling technology name from the share column name
        for param, value in cooling_tech_reshaped.loc[tech_name].items():  # Loop over all cooling technology dependent parameters
            column_name = f"avg_{param}"  # Create the column name for the weighted parameter
            weighted_parameters_df[column_name] = (
                weighted_parameters_df.get(column_name, 0) + buildings[share_column] * value
            )  # Compute the weighted parameter and add it to the DataFrame

    # Merge the computed weighted parameters with the original buildings DataFrame
    buildings_with_cooling_tech_data = pd.concat([buildings, weighted_parameters_df], axis=1)

    # Rename the cooling technology share columns to reflect the cooling technology name
    buildings_with_cooling_tech_data = buildings_with_cooling_tech_data.rename(columns={"avg_average_lifetime_yr": "avg_lifetime_cooling_technology_yr"})

    return buildings_with_cooling_tech_data


# Aggregation functions


def add_parameters_to_buildings(
    df_buildings: pd.DataFrame,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
    energy_class_parameters: list[dict[str, float]],
) -> pd.DataFrame:
    """Assigns the building type-, energy class-, and cooling technology-specific parameters to a building based on the building type, and energy class.

    Args:
        df_buildings (pd.DataFrame): The DataFrame containing the buildings for which the building-specific parameters are assigned.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        building_type_parameters (list[dict[str, float]]): The list containing the building type parameter dictionaries for the cooling demand model.
        energy_class_parameters (list[dict[str, float]]): The list containing the energy class parameter dictionaries for the cooling demand model.

    Returns:
        pd.DataFrame: The DataFrame containing the buildings with the building-specific parameters assigned.
    """
    # Determine the building type and add the building type-specific data
    df_buildings = add_building_type_data_to_buildings(df_buildings, global_parameters, building_type_parameters)

    # Determine the energy class and add the class-specific data
    df_buildings = add_energy_class_data_to_buildings(df_buildings, energy_class_parameters)

    # Add the derived parameters
    df_buildings = add_derived_parameters_to_buildings(df_buildings, global_parameters)

    return df_buildings


def scale_results_with_building_stock(buildings_agg: pd.DataFrame, global_parameters: dict[str, float], columns_to_scale: list[str]) -> pd.DataFrame:
    """Scales the aggregated results of the cooling demand model with the building stock growth rates.

    Args:
        buildings_agg (pd.DataFrame): DataFrame containing the aggregated results of the cooling demand model for each building type and energy class.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        columns_to_scale (list[str]): The list of columns to scale.

    Returns:
        pd.DataFrame: The DataFrame containing the scaled aggregated results of the cooling demand model for each building type and energy class.
    """  # Unload the global parameters
    bs_scale_factor_residential = 1 + global_parameters["building_stock_growth_residential_new"]  # Building stock growth rate of new residential buildings (type 1 and 3)
    bs_scale_factor_office = 1 + global_parameters["building_stock_growth_office_old"]  # Building stock growth rate of old office buildings (type 6 and 8)

    # Multiply relevant columns with the residential building stock scale factor
    buildings_agg.loc[buildings_agg["building_type_int"].isin([1, 3]), columns_to_scale] *= bs_scale_factor_residential

    # Multiply relevant columns with the office building stock scale factor
    buildings_agg.loc[buildings_agg["building_type_int"].isin([6, 8]), columns_to_scale] *= bs_scale_factor_office

    return buildings_agg


def aggregate_results(
    buildings: pd.DataFrame,
    global_parameters: pd.DataFrame,
    groupby_columns: tuple = ("building_type_int", "building_type", "energy_class", "energy_class_int"),
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

        def sum_arrays(series_with_arrays: pd.Series) -> np.array:
            return np.sum(np.vstack(series_with_arrays), axis=0)

        # Add the time series aggregations to the aggregations dictionary
        time_series_aggregations = {time_series_colum: sum_arrays for time_series_colum in buildings.filter(like="Q_").columns}
        aggregations.update(time_series_aggregations)

    # Group by building type and energy class and aggregate the results
    buildings_agg = buildings.groupby(list(groupby_columns)).agg(aggregations)

    # Reset index
    buildings_agg = buildings_agg.reset_index()

    # Fetch columns that are summed, as these need to be scaled with the building stock growth rates
    sum_columns = [key for key, value in aggregations.items() if str(value).startswith("sum") or str(value).startswith("<function aggregate_results.<locals>.sum_arrays")]

    if scale_with_building_stock:
        # Scale the aggregated results with the building stock growth rates
        buildings_agg = scale_results_with_building_stock(buildings_agg, global_parameters, sum_columns)

    return buildings_agg
