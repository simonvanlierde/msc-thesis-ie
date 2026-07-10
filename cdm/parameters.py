"""Building parameter assignment for the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cdm.geometric import calc_window_and_wall_areas


def calculate_building_population(buildings: pd.DataFrame, global_parameters: dict[str, float]) -> np.ndarray:
    """Calculate the building population based on the buildings and global parameters.

    Args:
        buildings (pd.DataFrame): The buildings for which the population is calculated.
        global_parameters (dict): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The calculated building population.
    """
    return np.where(
        buildings["end_use"] == "office",
        buildings["floor_area_total_m2"] * global_parameters["people_density_office"],
        buildings["number_of_residences"] * global_parameters["people_per_hh"],
    )


def determine_building_type(
    buildings: pd.DataFrame,
    global_parameters: dict[str, float],
) -> pd.Series:
    """Determines the building type of each building based on the building parameters.

    The type is a three-bit code (end use, low-rise, age) read as a decimal integer between 1 and 8.

    Args:
        buildings (pd.DataFrame): The buildings for which the building type is determined.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        pd.Series: The building type in integer representation
    """
    #  Unpack global parameters
    building_type_height_cutoff_m = global_parameters["building_type_height_cutoff_m"]
    building_type_age_cutoff_yr = global_parameters["building_type_age_cutoff_yr"]

    # Determine the end use binary, giving 0 for residential and 1 for office
    end_use_binary = buildings["end_use"] != "residential"

    # Determine the low-rise binary, giving 0 for high-rise and 1 for low-rise
    lowrise_binary = buildings["height_m"] <= building_type_height_cutoff_m

    # Determine the age binary, giving 0 for new and 1 for old
    age_binary = buildings["construction_year"] <= building_type_age_cutoff_yr

    # Read the three binary columns as a decimal integer between 1 and 8
    return 4 * end_use_binary + 2 * lowrise_binary + age_binary + 1


def assign_parameters_by_class(
    class_ints: pd.Series,
    parameters: list[dict[str, float]],
    class_column: str,
) -> pd.DataFrame:
    """Looks up the class-specific parameters for each building.

    Args:
        class_ints (pd.Series): The class (building type or energy class) of each building, in integer representation.
        parameters (list[dict[str, float]]): The list containing one parameter dictionary per class.
        class_column (str): The name of the key holding the class integer in each parameter dictionary.

    Returns:
        pd.DataFrame: The class-specific parameters for each building, indexed like ``class_ints``.
    """
    parameter_table = pd.DataFrame(parameters).set_index(class_column)
    return parameter_table.reindex(class_ints.to_numpy()).set_axis(class_ints.index)


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
    df_buildings["building_type_int"] = determine_building_type(df_buildings, global_parameters)

    # Determine the building-type specific parameters based on the building type
    df_building_type_parameters = assign_parameters_by_class(
        df_buildings["building_type_int"],
        building_type_parameters,
        "building_type_int",
    )
    df_buildings[df_building_type_parameters.columns] = df_building_type_parameters

    return df_buildings


# Energy class parameter assignment functions


def determine_energy_label_to_class_mappings(
    energy_class_parameters: list[dict],
) -> tuple[dict[int, int], dict[int, int]]:
    """Determines the mapping from energy label to energy class.

    Args:
        energy_class_parameters (list[dict]): The list of dictionaries containing the energy label class data. Each record mixes value types (int ids, float parameters and list-valued ``energy_labels_included_*`` columns).

    Returns:
        tuple[dict[int, int], dict[int, int]]: The dictionaries mapping energy label to energy class, for residential and office buildings respectively.
    """
    # Convert the energy class parameters to a DataFrame
    df_energy_class_parameters = pd.DataFrame(energy_class_parameters)

    # Construct the energy label to energy class mappings
    residential_mapping = {
        label: row["energy_class_int"]
        for _, row in df_energy_class_parameters.iterrows()
        for label in row["energy_labels_included_residential"]
    }
    office_mapping = {
        label: row["energy_class_int"]
        for _, row in df_energy_class_parameters.iterrows()
        for label in row["energy_labels_included_office"]
    }

    return residential_mapping, office_mapping


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
    energy_label_to_class_mapping_residential, energy_label_to_class_mapping_office = (
        determine_energy_label_to_class_mappings(
            energy_class_parameters,
        )
    )

    # Determine energy label class based on the energy label, using the right mapping for residential and office buildings
    energy_label_int = df_buildings["energy_label_int"]
    if energy_label_int.isna().any():
        # A null derived label cannot be cast to int (astype would crash) nor classified.
        msg = f"{int(energy_label_int.isna().sum())} building(s) have a null energy_label_int and cannot be assigned an energy class."
        raise ValueError(msg)
    energy_labels = energy_label_int.astype(int)
    energy_class_int = np.where(
        df_buildings["end_use"] == "residential",
        energy_labels.map(energy_label_to_class_mapping_residential),
        energy_labels.map(energy_label_to_class_mapping_office),
    )
    # A label absent from every energy_labels_included_* list maps to NaN, which would then be
    # silently dropped from every groupby-sum downstream (understating the stock totals). Fail loud.
    if pd.isna(energy_class_int).any():
        unmapped = sorted({int(label) for label in energy_labels[pd.isna(energy_class_int)]})
        msg = (
            f"energy label int(s) {unmapped} are not covered by any energy_labels_included_* list in "
            "the energy-class parameters; their buildings would be dropped from the results."
        )
        raise ValueError(msg)
    df_buildings["energy_class_int"] = energy_class_int

    # Determine the energy class-specific parameters based on the energy class
    df_energy_class_parameters = assign_parameters_by_class(
        df_buildings["energy_class_int"],
        energy_class_parameters,
        "energy_class_int",
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
    df_buildings["population"] = calculate_building_population(df_buildings, global_parameters)

    # Determine the wall and window areas
    (
        df_buildings["window_area_per_orientation_m2"],
        df_buildings["window_area_total_m2"],
        df_buildings["wall_area_total_m2"],
    ) = zip(*df_buildings.apply(calc_window_and_wall_areas, axis=1), strict=True)

    # Determine the total market penetration rate of cooling equipment
    df_buildings["total_MPR"] = df_buildings[
        [col for col in df_buildings.columns if col.startswith("cooling_technology_share")]
    ].sum(axis=1)

    return df_buildings


# Cooling technology parameter assignment functions


def add_cooling_technology_data_to_buildings(
    buildings: pd.DataFrame,
    cooling_technologies: pd.DataFrame,
) -> pd.DataFrame:
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
    weighted_parameters_df = pd.DataFrame(
        index=buildings.index,
    )  # Initialize DataFrame to store the weighted parameters
    for share_column in buildings.filter(
        like="cooling_technology_share",
    ).columns:  # Loop over all cooling technology share columns
        tech_name = share_column.split("cooling_technology_share_")[
            -1
        ]  # Extract the cooling technology name from the share column name
        for param, value in cooling_tech_reshaped.loc[
            tech_name
        ].items():  # Loop over all cooling technology dependent parameters
            column_name = f"avg_{param}"  # Create the column name for the weighted parameter
            weighted_parameters_df[column_name] = (
                weighted_parameters_df.get(column_name, 0) + buildings[share_column] * value
            )  # Compute the weighted parameter and add it to the DataFrame

    # Merge the computed weighted parameters with the original buildings DataFrame
    buildings_with_cooling_tech_data = pd.concat([buildings, weighted_parameters_df], axis=1)

    # Rename the cooling technology share columns to reflect the cooling technology name
    return buildings_with_cooling_tech_data.rename(
        columns={"avg_average_lifetime_yr": "avg_lifetime_cooling_technology_yr"},
    )


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
    return add_derived_parameters_to_buildings(df_buildings, global_parameters)
