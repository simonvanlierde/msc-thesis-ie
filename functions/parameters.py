"""Building parameter assignment for the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

import pandas as pd

from functions.geometric import calc_window_and_wall_areas


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
    df_building_type_parameters = df_buildings.apply(
        assign_building_type_parameters,
        args=(building_type_parameters,),
        axis=1,
    ).drop(
        columns=["building_type_int"],
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
    energy_label_to_class_mapping_residential, energy_label_to_class_mapping_office = (
        determine_energy_label_to_class_mappings(
            energy_class_parameters,
        )
    )

    # Determine energy label class based on the energy label, using the right mapping for residential and office buildings
    df_buildings["energy_class_int"] = df_buildings.apply(
        lambda row: (
            energy_label_to_class_mapping_residential[int(row["energy_label_int"])]
            if row["end_use"] == "residential"
            else energy_label_to_class_mapping_office[int(row["energy_label_int"])]
        ),
        axis=1,
    )

    # Determine the energy class-specific parameters based on the energy class
    df_energy_class_parameters = df_buildings.apply(
        assign_energy_class_parameters,
        args=(energy_class_parameters,),
        axis=1,
    ).drop(
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
