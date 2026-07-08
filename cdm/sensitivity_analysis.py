"""Functions used in the sensitivity analysis of the cooling demand model.

@author: Simon van Lierde
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from cdm.environmental import (
    calculate_environmental_impacts_from_cooling_demand,
    calculate_environmental_parameters_for_cooling_technologies,
)
from cdm.parameters import (
    add_cooling_technology_data_to_buildings,
    add_parameters_to_buildings,
    assign_parameters_by_class,
)
from cdm.thermodynamic import calc_cooling_demand_metrics_for_df
from cdm.time_series import create_time_series


def _record_impact_intensities(
    results: pd.DataFrame,
    index_value: float,
    impact_summary: dict[str, dict[str, float]],
) -> None:
    """Write the five per-floor-area impact intensities from an impact summary into one results row."""
    intensities = impact_summary["impacts_per_floor_area"]
    results.loc[index_value, "Cooling energy demand (kWh/m2)"] = intensities["E_cooling_capped_at_percentile_kWh_m2"]
    results.loc[index_value, "Peak cooling power demand (W/m2)"] = intensities["P_cooling_peak_percentile_W_m2"]
    results.loc[index_value, "Electricity use (kWh/m2)"] = intensities["electricity_use_intensity_kWh_m2"]
    results.loc[index_value, "GHG emissions (kg CO2eq/m2)"] = intensities["GHG_emissions_intensity_kgCO2eq_m2"]
    results.loc[index_value, "Material demand (kg/m2)"] = intensities["material_use_intensity_kg_m2"]


def run_cooling_demand_stage(
    buildings: pd.DataFrame,
    raw_weather_data: pd.DataFrame,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
    energy_class_parameters: list[dict[str, float]],
    multi_directional_solar_radiation_fractions_path: str,
    presence_load_factors_path: str,
) -> pd.DataFrame:
    """Runs the expensive half of the model: building parameters, time series and hourly cooling demand.

    The cooling demand depends on the building geometry, the energy class parameters, the non-cooling-technology
    building type parameters, and the global parameters -- but not on the cooling technology mix or on any
    cooling technology parameter. A sensitivity analysis that varies only those can therefore run this stage
    once and feed the result to ``run_impact_stage`` on every step.

    Args:
        buildings (pd.DataFrame): The DataFrame containing the buildings for which the cooling demand is calculated.
        raw_weather_data (pd.DataFrame): The DataFrame containing the raw weather data.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        building_type_parameters (list[dict[str, float]]): The list containing the building type parameter dictionaries for the cooling demand model.
        energy_class_parameters (list[dict[str, float]]): The list containing the energy class parameter dictionaries for the cooling demand model.
        multi_directional_solar_radiation_fractions_path (str): The path to the csv file containing the multi-directional solar radiation fractions.
        presence_load_factors_path (str): The path to the csv file containing the presence load factors.

    Returns:
        pd.DataFrame: The buildings with their parameters and hourly cooling demand metrics added.
    """
    # Assign the building parameters
    buildings = add_parameters_to_buildings(
        buildings,
        global_parameters,
        building_type_parameters,
        energy_class_parameters,
    )

    # Create time series for weather series and other hourly data
    time_series = create_time_series(
        global_parameters,
        raw_weather_data,
        multi_directional_solar_radiation_fractions_path,
        presence_load_factors_path,
    )

    # Calculate the cooling demand
    buildings = calc_cooling_demand_metrics_for_df(buildings, time_series, global_parameters, include_time_series=False)

    # Drop duplicate columns from the DataFrame
    return buildings.loc[:, ~buildings.columns.duplicated()]


def run_impact_stage(
    buildings_with_cooling_demand: pd.DataFrame,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
    cooling_technology_parameters: list[dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Runs the cheap half of the model: the cooling technology mix and the environmental impacts it drives.

    Args:
        buildings_with_cooling_demand (pd.DataFrame): The buildings as returned by ``run_cooling_demand_stage``.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        building_type_parameters (list[dict[str, float]]): The list containing the building type parameter dictionaries, whose cooling technology shares are re-read here.
        cooling_technology_parameters (list[dict[str, float]]): The list containing the cooling technology parameter dictionaries for the cooling demand model.

    Returns:
        tuple[pd.DataFrame, dict[str, dict[str, float]]]: The buildings with the environmental impacts added, and a dictionary containing the summary of environmental impacts.
    """
    buildings = buildings_with_cooling_demand.copy()  # The caller reuses its cooling demand across SA steps

    # Re-read the cooling technology shares from the building type parameters, which an SA may have varied
    cooling_technology_shares = assign_parameters_by_class(
        buildings["building_type_int"],
        building_type_parameters,
        "building_type_int",
    ).filter(like="cooling_technology_share")
    buildings[cooling_technology_shares.columns] = cooling_technology_shares
    buildings["total_MPR"] = cooling_technology_shares.sum(axis=1)

    # Build the cooling technologies DataFrame
    cooling_technologies = calculate_environmental_parameters_for_cooling_technologies(
        cooling_technology_parameters,
        global_parameters,
    )

    # Assign the cooling technology parameters to the buildings
    buildings = add_cooling_technology_data_to_buildings(buildings, cooling_technologies)

    # Calculate the environmental impacts
    return calculate_environmental_impacts_from_cooling_demand(buildings, global_parameters)


def run_CDM_model_for_SA(
    buildings: pd.DataFrame,
    raw_weather_data: pd.DataFrame,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
    energy_class_parameters: list[dict[str, float]],
    cooling_technology_parameters: list[dict[str, float]],
    multi_directional_solar_radiation_fractions_path: str,
    presence_load_factors_path: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """run_CDM_model_for_SA is a wrapper function that runs the cooling demand model for a sensitivity analysis.

    Args:
        buildings (pd.DataFrame): The DataFrame containing the buildings for which the cooling demand is calculated.
        raw_weather_data (pd.DataFrame): The DataFrame containing the raw weather data.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        building_type_parameters (list[dict[str, float]]): The list containing the building type parameter dictionaries for the cooling demand model.
        energy_class_parameters (list[dict[str, float]]): The list containing the energy class parameter dictionaries for the cooling demand model.
        cooling_technology_parameters (list[dict[str, float]]): The list containing the cooling technology parameter dictionaries for the cooling demand model.
        multi_directional_solar_radiation_fractions_path (str): The path to the csv file containing the multi-directional solar radiation fractions.
        presence_load_factors_path (str): The path to the csv file containing the presence load factors.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: The DataFrame containing the buildings with the cooling demand metrics and environmental impacts added, and a dictionary containing the summary of environmental impacts.
    """
    buildings_with_cooling_demand = run_cooling_demand_stage(
        buildings,
        raw_weather_data,
        global_parameters,
        building_type_parameters,
        energy_class_parameters,
        multi_directional_solar_radiation_fractions_path,
        presence_load_factors_path,
    )

    return run_impact_stage(
        buildings_with_cooling_demand,
        global_parameters,
        building_type_parameters,
        cooling_technology_parameters,
    )


def _cooling_demand_from_static_parameters(
    static_parameters: dict,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
) -> pd.DataFrame:
    """Runs the cooling demand stage once, for the SA drivers whose swept variable leaves it invariant."""
    return run_cooling_demand_stage(
        static_parameters["buildings"],
        static_parameters["raw_weather_data"],
        global_parameters,
        building_type_parameters,
        static_parameters["energy_class_parameters"],
        static_parameters["multi_directional_solar_radiation_fractions_path"],
        static_parameters["presence_load_factors_path"],
    )


def run_SA_for_variable_in_global_parameters(
    variable_name: str,
    variable_start: float,
    variable_end: float,
    global_parameters: dict[str, float],
    static_parameters: dict,
    calculation_steps: int = 50,
) -> pd.DataFrame:
    """run_SA_for_variable_in_global_parameters is a wrapper function that runs a sensitivity analysis on a variable in the global parameters.

    Args:
        variable_name (str): The name of the variable in the global parameters for which the sensitivity analysis is run.
        variable_start (float): The start of the value range for which the sensitivity analysis is run.
        variable_end (float): The end of the value range for which the sensitivity analysis is run.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        static_parameters (dict): A dictionary containing the other parameters for the cooling demand model that will not be altered throughout the sensitivity analysis.
        calculation_steps (int, optional): The number of times the cooling demand model is run for different values of the variable. Defaults to 50.

    Returns:
        pd.DataFrame: The DataFrame containing the results of the sensitivity analysis.

    """
    # Define the variable range
    variable_range = np.linspace(variable_start, variable_end, calculation_steps)

    # Define the DataFrame to which the results should be added
    df_variable_results = pd.DataFrame(index=variable_range)

    # Loop over the variable range
    for variable in tqdm(variable_range):
        # Create a copy of the global_parameters in which the variable is changed to the current variable
        global_parameters_with_variable = global_parameters.copy()
        global_parameters_with_variable[variable_name] = variable

        # Run the cooling demand model with the current variable
        _, impact_summary = run_CDM_model_for_SA(
            static_parameters["buildings"],
            static_parameters["raw_weather_data"],
            global_parameters_with_variable,
            static_parameters["building_type_parameters"],
            static_parameters["energy_class_parameters"],
            static_parameters["cooling_technology_parameters"],
            static_parameters["multi_directional_solar_radiation_fractions_path"],
            static_parameters["presence_load_factors_path"],
        )

        # Add the results to the DataFrame
        _record_impact_intensities(df_variable_results, variable, impact_summary)

    return df_variable_results


def run_SA_for_variable_in_cooling_technology_parameters(
    variable_name: str,
    multiplier_start: float,
    multiplier_end: float,
    cooling_technology_parameters: dict[str, float],
    static_parameters: dict,
    calculation_steps: int = 50,
) -> pd.DataFrame:
    """A wrapper function that runs a sensitivity analysis on a variable in the cooling technology parameters. It does this by multiplying all values of a variable within the cooling technology table with a range of multipliers.

    Args:
        variable_name (str): The name of the variable in the global parameters for which the sensitivity analysis is run.
        multiplier_start (float): The start of the range of multiplier for the variable for which the sensitivity analysis is run.
        multiplier_end (float): The end of the range of multiplier for the variable for which the sensitivity analysis is run.
        cooling_technology_parameters (dict[str, float]): The dictionary containing the cooling technology parameters for the cooling demand model.
        static_parameters (dict): A dictionary containing the other parameters for the cooling demand model that will not be altered throughout the sensitivity analysis.
        calculation_steps (int, optional): The number of times the cooling demand model is run for different values of the variable. Defaults to 50.

    Returns:
        pd.DataFrame: The DataFrame containing the results of the sensitivity analysis.

    """
    # Read the cooling technology parameters as DataFrame
    df_cooling_technology_parameters = pd.DataFrame(cooling_technology_parameters)

    # Define the multiplication range
    multiplication_range = np.linspace(multiplier_start, multiplier_end, calculation_steps)

    # Figure out the average value of the variable in the cooling technology parameters
    variable_average = df_cooling_technology_parameters[variable_name].mean()

    # Create a variable_range by multiplying the average value of the variable with the multiplication_range
    variable_average_range = variable_average * multiplication_range

    # Define the DataFrame to which the results should be added
    df_variable_results = pd.DataFrame(index=variable_average_range)

    # The cooling technology parameters do not enter the cooling demand, so it is calculated once up front
    buildings_with_cooling_demand = _cooling_demand_from_static_parameters(
        static_parameters,
        static_parameters["global_parameters"],
        static_parameters["building_type_parameters"],
    )

    # Loop over the variable range
    for variable in tqdm(variable_average_range):
        # Determine the multiplier for the variable
        multiplier = variable / variable_average

        # Create a copy of the global_parameters in which the variable is changed to the current variable
        df_cooling_technology_parameters_with_multiplier = df_cooling_technology_parameters.copy()
        df_cooling_technology_parameters_with_multiplier[variable_name] *= multiplier

        # Recalculate only the environmental impacts with the current variable
        _, impact_summary = run_impact_stage(
            buildings_with_cooling_demand,
            static_parameters["global_parameters"],
            static_parameters["building_type_parameters"],
            df_cooling_technology_parameters_with_multiplier.to_dict(orient="records"),
        )

        # Add the results to the DataFrame
        _record_impact_intensities(df_variable_results, variable, impact_summary)

    return df_variable_results


def run_SA_for_cooling_technology_mix(
    cooling_tech_one: str,
    cooling_tech_two: str,
    building_type_parameters: list[dict[str, float]],
    static_parameters: dict,
    calculation_steps: int = 50,
) -> pd.DataFrame:
    """A wrapper function that runs a sensitivity analysis on the cooling technology mix. It simulates a cooling technology mix in which the share of one cooling technology is increased, while the share of another cooling technology is decreased.

    Args:
        cooling_tech_one (str): The name of the first cooling technology. Can be one of "ASHP", "GSHP", "WSHP", "chiller", "AC_split", or "AC_mobile".
        cooling_tech_two (str): The name of the second cooling technology. Can be one of "ASHP", "GSHP", "WSHP", "chiller", "AC_split", or "AC_mobile".
        building_type_parameters (list[dict[str, float]]): The dictionary containing the building type parameters for the cooling demand model.
        static_parameters (dict): A dictionary containing the other parameters for the cooling demand model that will not be altered throughout the sensitivity analysis.
        calculation_steps (int, optional): The number of times the cooling demand model is run for different values of the cooling technology mix. Defaults to 50.

    Returns:
        pd.DataFrame: The DataFrame containing the results of the sensitivity analysis.

    """
    # Define the variable range
    mix_range = np.linspace(0, 100, calculation_steps)

    # Define the DataFrame to which the results should be added
    df_variable_results = pd.DataFrame(index=mix_range)

    # Read the building type parameters as DataFrame
    df_building_type_parameters = pd.DataFrame(building_type_parameters)

    # Set all cooling technology shares to zero
    df_building_type_parameters = df_building_type_parameters.assign(
        **{col: 0 for col in df_building_type_parameters.columns if col.startswith("cooling_technology_share")},
    )

    # The cooling technology mix does not enter the cooling demand, so it is calculated once up front
    buildings_with_cooling_demand = _cooling_demand_from_static_parameters(
        static_parameters,
        static_parameters["global_parameters"],
        df_building_type_parameters.to_dict(orient="records"),
    )

    # Loop over the variable range
    for mix_share in tqdm(mix_range):
        # Create a copy of the building_type_parameters in which the variable is changed to the current variable
        df_building_type_parameters_with_mix_share = df_building_type_parameters.copy()

        # Assign the mix shares to the cooling technology shares
        df_building_type_parameters_with_mix_share["cooling_technology_share_" + cooling_tech_one] = mix_share / 100
        df_building_type_parameters_with_mix_share["cooling_technology_share_" + cooling_tech_two] = (
            100 - mix_share
        ) / 100

        # Recalculate only the environmental impacts with the current mix
        _, impact_summary = run_impact_stage(
            buildings_with_cooling_demand,
            static_parameters["global_parameters"],
            df_building_type_parameters_with_mix_share.to_dict(orient="records"),
            static_parameters["cooling_technology_parameters"],
        )

        # Add the results to the DataFrame
        _record_impact_intensities(df_variable_results, mix_share, impact_summary)

    return df_variable_results


def run_SA_for_total_market_penetration(
    multiplier_start: float,
    multiplier_end: float,
    building_type_parameters: dict[str, float],
    building_type_prevalence: pd.Series,
    static_parameters: dict,
    calculation_steps: int = 50,
) -> pd.DataFrame:
    """A wrapper function that runs a sensitivity analysis on the total market penetration rate.

    Args:
        multiplier_start (float): The start of the range of multiplier for the total market penetration rate.
        multiplier_end (float): The end of the range of multiplier for the total market penetration rate.
        building_type_parameters (dict[str, float]): The dictionary containing the building type parameters for the cooling demand model.
        building_type_prevalence (pd.Series): The prevalence of the building types in the building stock, normalized to sum to 1.
        static_parameters (dict): A dictionary containing the other parameters for the cooling demand model that will not be altered throughout the sensitivity analysis.
        calculation_steps (int, optional): The number of times the cooling demand model is run for different values of the for the total market penetration rate. Defaults to 50.

    Returns:
        pd.DataFrame: The DataFrame containing the results of the sensitivity analysis.

    """
    # Read the cooling technology parameters as DataFrame
    df_building_type_parameters = pd.DataFrame(building_type_parameters)
    df_building_type_parameters = df_building_type_parameters.set_index("building_type")

    # Define the multiplication range
    multiplication_range = np.linspace(multiplier_start, multiplier_end, calculation_steps)

    # Define the DataFrame to which the results should be added
    df_results = pd.DataFrame(index=multiplication_range)

    # The market penetration rate only scales the cooling technology shares, which do not enter the
    # cooling demand, so it is calculated once up front
    buildings_with_cooling_demand = _cooling_demand_from_static_parameters(
        static_parameters,
        static_parameters["global_parameters"],
        df_building_type_parameters.to_dict(orient="records"),
    )

    # Loop over the variable range
    for multiplier in tqdm(multiplication_range):
        # Create a copy of the global_parameters in which the variable is changed to the current variable
        df_building_type_parameters_with_multiplier = df_building_type_parameters.copy()

        # Multiply each column that start with the name "cooling_technology_share" with the multiplier
        for col in df_building_type_parameters_with_multiplier.columns:
            if col.startswith("cooling_technology_share"):
                df_building_type_parameters_with_multiplier[col] *= multiplier

        # Calculate the weighted total market penetration rate
        df_building_type_parameters_with_multiplier["total_market_penetration_rate"] = (
            df_building_type_parameters_with_multiplier[
                [
                    col
                    for col in df_building_type_parameters_with_multiplier.columns
                    if col.startswith("cooling_technology_share")
                ]
            ].sum(axis=1)
        )

        # Figure out the weighted average value of the total market penetration
        total_market_penetration_with_multiplier = (
            df_building_type_parameters_with_multiplier["total_market_penetration_rate"]
            .mul(building_type_prevalence)
            .sum()
        )

        # Recalculate only the environmental impacts with the current market penetration rate
        _, impact_summary = run_impact_stage(
            buildings_with_cooling_demand,
            static_parameters["global_parameters"],
            df_building_type_parameters_with_multiplier.to_dict(orient="records"),
            static_parameters["cooling_technology_parameters"],
        )

        # Add the results to the DataFrame
        df_results.loc[multiplier, "Total MPR"] = total_market_penetration_with_multiplier
        _record_impact_intensities(df_results, multiplier, impact_summary)

    # Set the index of the results DataFrame to the measured total market penetration, in percent
    df_results = df_results.set_index(df_results["Total MPR"] * 100)
    return df_results.drop(columns="Total MPR")


def normalize_SA_results(
    SA_results: pd.DataFrame,
    ref_value_in_SA_results: float,
) -> pd.DataFrame:
    """Normalize the results of a sensitivity analysis to the value of the independent variable used in the reference scenario.

    Args:
        SA_results (pd.DataFrame): The DataFrame containing the results of the sensitivity analysis.
        ref_value_in_SA_results (float): The value of the independent variable in the sensitivity analysis, closest to the one used in the reference scenario.

    Returns:
        pd.DataFrame: The DataFrame containing the normalized results of the sensitivity analysis.
    """
    # Normalize all the columns in the DataFrame to the value closest to the reference value
    SA_results_normalized = SA_results / SA_results.loc[ref_value_in_SA_results]

    # Rename all the columns in SA_results_normalized to remove impact units from the labels
    SA_results_normalized.columns = [col.split("(")[0].strip() for col in SA_results_normalized.columns]

    return SA_results_normalized


def calculate_elasticity_for_SA_results(SA_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate the elasticity of the cooling demand and environmental impacts with respect to the independent variable.

    Args:
        SA_results (pd.DataFrame): The DataFrame containing the results of the sensitivity analysis.

    Returns:
        pd.DataFrame: The DataFrame containing the elasticity of the cooling demand and environmental impacts with respect to the independent variable.
    """
    # Determine the elasticity between the independent variable and the cooling demand and environmental impacts
    SA_results_elasticity = (
        SA_results.pct_change() / SA_results.index.to_series().pct_change().to_numpy()[:, np.newaxis]
    )

    # Rename all the columns in SA_results_elasticity to remove impact units from the labels
    SA_results_elasticity.columns = [col.split("(")[0].strip() for col in SA_results_elasticity.columns]

    return SA_results_elasticity
