"""Functions used in the sensitivity analysis of the cooling demand model.

@author: Simon van Lierde
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from functions.data_handling import add_cooling_technology_data_to_buildings, add_parameters_to_buildings
from functions.environmental import (
    calculate_environmental_impacts_from_cooling_demand,
    calculate_environmental_parameters_for_cooling_technologies,
)
from functions.thermodynamic import calc_cooling_demand_metrics_for_df
from functions.time_series import create_time_series


def run_CDM_model_for_SA(
    buildings: pd.DataFrame,
    raw_weather_data: pd.DataFrame,
    global_parameters: dict[str, float],
    building_type_parameters: list[dict[str, float]],
    energy_class_parameters: list[dict[str, float]],
    cooling_technology_parameters: list[dict[str, float]],
    multi_directional_solar_radiation_fractions_path: str,
    presence_load_factors_path: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
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
    # Assign the building parameters
    buildings = add_parameters_to_buildings(buildings, global_parameters, building_type_parameters, energy_class_parameters)

    # Build the cooling technologies DataFrame
    cooling_technologies = calculate_environmental_parameters_for_cooling_technologies(cooling_technology_parameters, global_parameters)

    # Assign the cooling technology parameters to the buildings
    buildings = add_cooling_technology_data_to_buildings(buildings, cooling_technologies)

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
    buildings = buildings.loc[:, ~buildings.columns.duplicated()]

    # Calculate the environmental impacts
    buildings, impact_summary = calculate_environmental_impacts_from_cooling_demand(buildings, global_parameters)

    return buildings, impact_summary


def run_CDM_model_for_SA_for_indirect_parameter(
    variable_name_to_report: str,
    aggregation_type: str,
    parameters: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """A wrapper function that runs the cooling demand model for a sensitivity analysis and reports back the value of an indirect parameter.

    Args:
        variable_name_to_report (str): The column name of the variable to report the value of.
        aggregation_type (str): The type of aggregation to use for the results. Either "sum" or "mean".
        parameters (dict): A dictionary containing the parameters for the cooling demand model.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: The DataFrame containing the buildings with the cooling demand metrics and environmental impacts added, and a dictionary containing the summary of environmental impacts.
    """
    # Unload the parameters
    buildings = parameters["buildings"]
    raw_weather_data = parameters["raw_weather_data"]
    global_parameters = parameters["global_parameters"]
    building_type_parameters = parameters["building_type_parameters"]
    energy_class_parameters = parameters["energy_class_parameters"]
    cooling_technology_parameters = parameters["cooling_technology_parameters"]
    multi_directional_solar_radiation_fractions_path = parameters["multi_directional_solar_radiation_fractions_path"]
    presence_load_factors_path = parameters["presence_load_factors_path"]

    # Assign the building parameters
    buildings = add_parameters_to_buildings(buildings, global_parameters, building_type_parameters, energy_class_parameters)

    # Build the cooling technologies DataFrame
    cooling_technologies = calculate_environmental_parameters_for_cooling_technologies(cooling_technology_parameters, global_parameters)

    # Assign the cooling technology parameters to the buildings
    buildings = add_cooling_technology_data_to_buildings(buildings, cooling_technologies)

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
    buildings = buildings.loc[:, ~buildings.columns.duplicated()]

    # Calculate the environmental impacts
    buildings, impact_summary = calculate_environmental_impacts_from_cooling_demand(buildings, global_parameters)

    # Calculate the total floor area of the buildings
    total_floor_area_m2 = buildings["floor_area_total_m2"].sum()

    # Calculate the weighted value of the variable
    if aggregation_type == "sum":
        value_of_variable = (buildings[variable_name_to_report] * buildings["floor_area_total_m2"]).sum() / total_floor_area_m2
    elif aggregation_type == "mean":
        value_of_variable = (buildings[variable_name_to_report] * buildings["floor_area_total_m2"]).mean() / total_floor_area_m2
    else:
        raise ValueError("The aggregation type must be either 'sum' or 'mean'.")

    return buildings, impact_summary, value_of_variable


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
        df_variable_results.loc[variable, "Cooling energy demand (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["E_cooling_capped_at_percentile_kWh_m2"]
        df_variable_results.loc[variable, "Peak cooling power demand (W/m2)"] = impact_summary["impacts_per_floor_area"]["P_cooling_peak_percentile_W_m2"]
        df_variable_results.loc[variable, "Electricity use (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["electricity_use_intensity_kWh_m2"]
        df_variable_results.loc[variable, "GHG emissions (kg CO2eq/m2)"] = impact_summary["impacts_per_floor_area"]["GHG_emissions_intensity_kgCO2eq_m2"]
        df_variable_results.loc[variable, "Material demand (kg/m2)"] = impact_summary["impacts_per_floor_area"]["material_use_intensity_kg_m2"]

    return df_variable_results


def run_SA_for_variable_in_building_type_parameters(
    variable_name: str,
    variable_start: float,
    variable_end: float,
    building_type_parameters: list[dict[str, float]],
    static_parameters: dict,
    calculation_steps: int = 50,
) -> pd.DataFrame:
    """A wrapper function that runs a sensitivity analysis on a variable in the building type parameters.

    Args:
        variable_name (str): The name of the variable in the building type parameters for which the sensitivity analysis is run.
        variable_start (float): The start of the value range for which the sensitivity analysis is run.
        variable_end (float): The end of the value range for which the sensitivity analysis is run.
        building_type_parameters (list[dict[str, float]]): The dictionary containing the building type parameters for the cooling demand model.
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
        # Create a copy of the building_type_parameters in which the variable is changed to the current variable
        building_type_parameters_with_variable = building_type_parameters.copy()
        building_type_parameters_with_variable[variable_name] = variable

        # Run the cooling demand model with the current variable
        _, impact_summary = run_CDM_model_for_SA(
            static_parameters["buildings"],
            static_parameters["raw_weather_data"],
            static_parameters["global_parameters"],
            building_type_parameters_with_variable,
            static_parameters["energy_class_parameters"],
            static_parameters["cooling_technology_parameters"],
            static_parameters["multi_directional_solar_radiation_fractions_path"],
            static_parameters["presence_load_factors_path"],
        )

        # Add the results to the DataFrame
        df_variable_results.loc[variable, "Cooling energy demand (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["E_cooling_capped_at_percentile_kWh_m2"]
        df_variable_results.loc[variable, "Peak cooling power demand (W/m2)"] = impact_summary["impacts_per_floor_area"]["P_cooling_peak_percentile_W_m2"]
        df_variable_results.loc[variable, "Electricity use (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["electricity_use_intensity_kWh_m2"]
        df_variable_results.loc[variable, "GHG emissions (kg CO2eq/m2)"] = impact_summary["impacts_per_floor_area"]["GHG_emissions_intensity_kgCO2eq_m2"]
        df_variable_results.loc[variable, "Material demand (kg/m2)"] = impact_summary["impacts_per_floor_area"]["material_use_intensity_kg_m2"]

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

    # Loop over the variable range
    for variable in tqdm(variable_average_range):
        # Determine the multiplier for the variable
        multiplier = variable / variable_average

        # Create a copy of the global_parameters in which the variable is changed to the current variable
        df_cooling_technology_parameters_with_multiplier = df_cooling_technology_parameters.copy()
        df_cooling_technology_parameters_with_multiplier[variable_name] *= multiplier

        # Run the cooling demand model with the current variable
        _, impact_summary = run_CDM_model_for_SA(
            static_parameters["buildings"],
            static_parameters["raw_weather_data"],
            static_parameters["global_parameters"],
            static_parameters["building_type_parameters"],
            static_parameters["energy_class_parameters"],
            df_cooling_technology_parameters_with_multiplier.to_dict(orient="records"),
            static_parameters["multi_directional_solar_radiation_fractions_path"],
            static_parameters["presence_load_factors_path"],
        )

        # Add the results to the DataFrame
        df_variable_results.loc[variable, "Cooling energy demand (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["E_cooling_capped_at_percentile_kWh_m2"]
        df_variable_results.loc[variable, "Peak cooling power demand (W/m2)"] = impact_summary["impacts_per_floor_area"]["P_cooling_peak_percentile_W_m2"]
        df_variable_results.loc[variable, "Electricity use (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["electricity_use_intensity_kWh_m2"]
        df_variable_results.loc[variable, "GHG emissions (kg CO2eq/m2)"] = impact_summary["impacts_per_floor_area"]["GHG_emissions_intensity_kgCO2eq_m2"]
        df_variable_results.loc[variable, "Material demand (kg/m2)"] = impact_summary["impacts_per_floor_area"]["material_use_intensity_kg_m2"]

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
    df_building_type_parameters = df_building_type_parameters.assign(**{col: 0 for col in df_building_type_parameters.columns if col.startswith("cooling_technology_share")})

    # Loop over the variable range
    for mix_share in tqdm(mix_range):
        # Create a copy of the building_type_parameters in which the variable is changed to the current variable
        df_building_type_parameters_with_mix_share = df_building_type_parameters.copy()

        # Assign the mix shares to the cooling technology shares
        df_building_type_parameters_with_mix_share["cooling_technology_share_" + cooling_tech_one] = mix_share / 100
        df_building_type_parameters_with_mix_share["cooling_technology_share_" + cooling_tech_two] = (100 - mix_share) / 100

        # Run the cooling demand model with the current variable
        _, impact_summary = run_CDM_model_for_SA(
            static_parameters["buildings"],
            static_parameters["raw_weather_data"],
            static_parameters["global_parameters"],
            df_building_type_parameters_with_mix_share.to_dict(orient="records"),
            static_parameters["energy_class_parameters"],
            static_parameters["cooling_technology_parameters"],
            static_parameters["multi_directional_solar_radiation_fractions_path"],
            static_parameters["presence_load_factors_path"],
        )

        # Add the results to the DataFrame
        df_variable_results.loc[mix_share, "Cooling energy demand (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["E_cooling_capped_at_percentile_kWh_m2"]
        df_variable_results.loc[mix_share, "Peak cooling power demand (W/m2)"] = impact_summary["impacts_per_floor_area"]["P_cooling_peak_percentile_W_m2"]
        df_variable_results.loc[mix_share, "Electricity use (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["electricity_use_intensity_kWh_m2"]
        df_variable_results.loc[mix_share, "GHG emissions (kg CO2eq/m2)"] = impact_summary["impacts_per_floor_area"]["GHG_emissions_intensity_kgCO2eq_m2"]
        df_variable_results.loc[mix_share, "Material demand (kg/m2)"] = impact_summary["impacts_per_floor_area"]["material_use_intensity_kg_m2"]

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

    # Loop over the variable range
    for multiplier in tqdm(multiplication_range):
        # Create a copy of the global_parameters in which the variable is changed to the current variable
        df_building_type_parameters_with_multiplier = df_building_type_parameters.copy()

        # Multiply each column that start with the name "cooling_technology_share" with the multiplier
        for col in df_building_type_parameters_with_multiplier.columns:
            if col.startswith("cooling_technology_share"):
                df_building_type_parameters_with_multiplier[col] *= multiplier

        # Calculate the weighted total market penetration rate
        df_building_type_parameters_with_multiplier["total_market_penetration_rate"] = df_building_type_parameters_with_multiplier[
            [col for col in df_building_type_parameters_with_multiplier.columns if col.startswith("cooling_technology_share")]
        ].sum(axis=1)

        # Figure out the weighted average value of the total market penetration
        total_market_penetration_with_multiplier = df_building_type_parameters_with_multiplier["total_market_penetration_rate"].mul(building_type_prevalence).sum()

        # Run the cooling demand model with the current variable
        _, impact_summary = run_CDM_model_for_SA(
            static_parameters["buildings"],
            static_parameters["raw_weather_data"],
            static_parameters["global_parameters"],
            df_building_type_parameters_with_multiplier.to_dict(orient="records"),
            static_parameters["energy_class_parameters"],
            static_parameters["cooling_technology_parameters"],
            static_parameters["multi_directional_solar_radiation_fractions_path"],
            static_parameters["presence_load_factors_path"],
        )

        # Add the results to the DataFrame
        df_results.loc[multiplier, "Total MPR"] = total_market_penetration_with_multiplier
        df_results.loc[multiplier, "Cooling energy demand (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["E_cooling_capped_at_percentile_kWh_m2"]
        df_results.loc[multiplier, "Peak cooling power demand (W/m2)"] = impact_summary["impacts_per_floor_area"]["P_cooling_peak_percentile_W_m2"]
        df_results.loc[multiplier, "Electricity use (kWh/m2)"] = impact_summary["impacts_per_floor_area"]["electricity_use_intensity_kWh_m2"]
        df_results.loc[multiplier, "GHG emissions (kg CO2eq/m2)"] = impact_summary["impacts_per_floor_area"]["GHG_emissions_intensity_kgCO2eq_m2"]
        df_results.loc[multiplier, "Material demand (kg/m2)"] = impact_summary["impacts_per_floor_area"]["material_use_intensity_kg_m2"]

    # Set the index of the results DataFrame to the measured total market penetration, in percent
    df_results = df_results.set_index(df_results["Total MPR"] * 100)
    df_results = df_results.drop(columns="Total MPR")

    return df_results


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
    SA_results_elasticity = SA_results.pct_change() / SA_results.index.to_series().pct_change().to_numpy()[:, np.newaxis]

    # Rename all the columns in SA_results_elasticity to remove impact units from the labels
    SA_results_elasticity.columns = [col.split("(")[0].strip() for col in SA_results_elasticity.columns]

    return SA_results_elasticity


def plot_SA_results_per_impact(
    SA_results: pd.DataFrame,
    reference_values: dict[str, float],
    variable_name_print: str,
    x_axis_label: str,
    include_scenario_lines_in_plots: bool = True,
    figure_size: tuple[float, float] = (6, 6),
) -> None:
    """Plot the results of a sensitivity analysis per impact.

    Args:
        SA_results (pd.DataFrame): The DataFrame containing the results of the sensitivity analysis.
        reference_values (dict[str, float]): The dictionary containing the values of the independent variable used across scenarios.
        variable_name_print (str): The name of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        x_axis_label (str): The label for the x-axis of the plot.
        include_scenario_lines_in_plots (bool): Whether or not to include the lines indicating the values of the independent variable in the different scenarios in the plots. Defaults to True.
        figure_size (tuple[float, float], optional): Figure width and height, in inches. Defaults to (6, 6).
    """
    SA_results.plot(subplots=True, figsize=figure_size, sharex=True)  # Initialize the plot

    plt.suptitle(f"Influence of {variable_name_print} on impacts")  # Add a title to the plot

    for ax in plt.gcf().axes:  # Add the line indicating the reference value in each subplot
        ax.axvline(x=reference_values["SQ"], color="dimgray", linestyle="dashed", alpha=0.7, label="Value in reference scenario")

        if include_scenario_lines_in_plots:
            ax.axvline(x=reference_values["2030"], color="gray", linestyle="dotted", alpha=0.7, label="Value in 2030 scenario")
            ax.axvline(x=reference_values["2050_L"], color="gold", linestyle="dotted", alpha=0.7, label="Value in 2050 L scenario")
            ax.axvline(x=reference_values["2050_M"], color="darkorange", linestyle="dotted", alpha=0.7, label="Value in 2050 M scenario")
            ax.axvline(x=reference_values["2050_H"], color="red", linestyle="dotted", alpha=0.7, label="Value in 2050 H scenario")

    plt.gcf().axes[0].legend()  # Add a legend for the reference value to the first subplot
    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.tight_layout()  # Make sure the subplots don't overlap

    plt.savefig(
        f"data/output/images/SA/SA_results_{variable_name_print.replace(' ', '_')}{'_with_scenario_lines' if include_scenario_lines_in_plots else ''}.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    plt.show()  # Show the plot to the console
    plt.close()


def plot_SA_results_normalized(
    SA_results_normalized: pd.DataFrame,
    reference_values: float,
    ref_value_in_SA_results: float,
    variable_name_print: str,
    variable_unit_print: str,
    x_axis_label: str,
    round_to: int = 0,
    include_scenario_lines_in_plots: bool = True,
    figure_size: tuple[float, float] = (6, 6),
) -> None:
    """Plot the normalized results of a sensitivity analysis.

    Args:
        SA_results_normalized (pd.DataFrame): The DataFrame containing the normalized results of the sensitivity analysis.
        reference_values (dict[str, float]): The dictionary containing the values of the independent variable used across scenarios.
        ref_value_in_SA_results: The value of the independent variable in the sensitivity analysis, closest to the one used in the reference scenario.
        variable_name_print (str): The name of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        variable_unit_print (str): The unit of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        x_axis_label (str):The label for the x-axis of the plot.
        round_to (int): The number of decimals to which printed values should be rounded. Defaults to 0.
        include_scenario_lines_in_plots (bool): Whether or not to include the lines indicating the values of the independent variable in the different scenarios in the plots. Defaults to True.
        figure_size (tuple[float, float], optional): Figure width and height, in inches. Defaults to (6, 6).
    """
    SA_results_normalized.plot(subplots=False, figsize=figure_size, sharex=True)  # Initialize the plot

    plt.title(
        f"Influence of {variable_name_print} on impacts,\nnormalized to the value at {round(reference_values['SQ'], round_to)} {variable_unit_print}",
    )  # Add a title to the plot
    plt.axvline(x=ref_value_in_SA_results, color="dimgray", linestyle="dashed", alpha=0.7, label="Value in reference scenario")

    if include_scenario_lines_in_plots:
        plt.axvline(x=reference_values["2030"], color="gray", linestyle="dotted", alpha=0.7, label="Value in 2030 scenario")
        plt.axvline(x=reference_values["2050_L"], color="gold", linestyle="dotted", alpha=0.7, label="Value in 2050 L scenario")
        plt.axvline(x=reference_values["2050_M"], color="darkorange", linestyle="dotted", alpha=0.7, label="Value in 2050 M scenario")
        plt.axvline(x=reference_values["2050_H"], color="red", linestyle="dotted", alpha=0.7, label="Value in 2050 H scenario")

    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.legend()  # Add a legend to the plot
    plt.tight_layout()  # Make sure the subplots don't overlap
    plt.savefig(
        f"data/output/images/SA/SA_results_{variable_name_print.replace(' ', '_')}_normalized{'_with_scenario_lines' if include_scenario_lines_in_plots else ''}.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    plt.show()  # Show the plot to the console
    plt.close()


def plot_SA_results_elasticities(
    SA_results_elasticities: pd.DataFrame,
    reference_values: float,
    variable_name_print: str,
    x_axis_label: str,
    include_scenario_lines_in_plots: bool = True,
    figure_size: tuple[float, float] = (6, 6),
) -> None:
    """Plot the normalized results of a sensitivity analysis.

    Args:
        SA_results_elasticities (pd.DataFrame): The DataFrame containing the elasticities between the independent variable and the cooling demand and environmental impacts.
        reference_values (dict[str, float]): The dictionary containing the values of the independent variable used across scenarios.
        variable_name_print (str): The name of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        variable_unit_print (str): The unit of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        x_axis_label (str): The label for the x-axis of the plot.
        include_scenario_lines_in_plots (bool): Whether or not to include the lines indicating the values of the independent variable in the different scenarios in the plots. Defaults to True.
        figure_size (tuple[float, float], optional): Figure width and height, in inches. Defaults to (6, 6).
    """
    SA_results_elasticities.plot(subplots=False, figsize=figure_size, sharex=True)  # Initialize the plot

    plt.title(f"Elasticity of impacts with respect to the {variable_name_print}")  # Add a title to the plot

    plt.axvline(x=reference_values["SQ"], color="dimgray", linestyle="dashed", alpha=0.7, label="Value in reference scenario")

    if include_scenario_lines_in_plots:
        plt.axvline(x=reference_values["2030"], color="gray", linestyle="dotted", alpha=0.7, label="Value in 2030 scenario")
        plt.axvline(x=reference_values["2050_L"], color="gold", linestyle="dotted", alpha=0.7, label="Value in 2050 L scenario")
        plt.axvline(x=reference_values["2050_M"], color="darkorange", linestyle="dotted", alpha=0.7, label="Value in 2050 M scenario")
        plt.axvline(x=reference_values["2050_H"], color="red", linestyle="dotted", alpha=0.7, label="Value in 2050 H scenario")

    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.legend()  # Add a legend to the plot
    plt.tight_layout()  # Make sure the subplots don't overlap

    plt.savefig(
        f"data/output/images/SA/SA_results_{variable_name_print.replace(' ', '_')}_elasticities{'_with_scenario_lines' if include_scenario_lines_in_plots else ''}.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    plt.show()  # Show the plot to the console
    plt.close()


def post_process_SA_results(
    SA_results: pd.DataFrame,
    reference_values: dict[str, float],
    variable_name_print: str,
    variable_unit_print: str,
    round_to: int = 0,
    include_scenario_lines_in_plots: bool = True,
    figure_size: tuple[float, float] = (6, 6),
    include_CDM_metrics_in_plots: bool = False,
) -> pd.DataFrame:
    """Normalizes the results of a sensitivity analysis, determines the elasticities and plots the outcomes.

    Args:
        SA_results (pd.DataFrame): The DataFrame containing the results of the sensitivity analysis.
        reference_values (dict[str, float]): The dictionary containing the values of the independent variable used across scenarios.
        variable_name (str): The name of the independent variable for which the sensitivity analysis is run, in the code.
        variable_name_print (str): The name of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        variable_unit_print (str): The unit of the independent variable for which the sensitivity analysis is run, as to be printed in the plot.
        round_to (int): The number of decimals to which printed values should be rounded. Defaults to 0.
        include_scenario_lines_in_plots (bool): Whether or not to include the lines indicating the values of the independent variable in the different scenarios in the plots. Defaults to True.
        figure_size (tuple[float, float], optional): Figure width and height, in inches. Defaults to (6, 6).
        include_CDM_metrics_in_plots (bool, optional): Whether or not to include the cooling demand metrics (cooling energy demand and peak cooling power demand) in the plots. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame containing the elasticity of the cooling demand and environmental impacts with respect to the independent variable, at the reference value.
    """
    # Find the variable value in the sensitivity analysis closest to the reference value used in the status quo scenario
    ref_value_in_SA_results = SA_results.index[(np.abs(SA_results.index - reference_values["SQ"])).argmin()]

    # Normalize the results to the value used for the independent variable in the reference scenario
    SA_results_normalized = normalize_SA_results(SA_results, ref_value_in_SA_results)

    # Calculate the elasticity of the cooling demand and environmental impacts with respect to the independent variable
    SA_results_elasticity_full = SA_results_elasticity = calculate_elasticity_for_SA_results(SA_results)

    # Define the x-axis label
    x_axis_label = variable_name_print[0].upper() + variable_name_print[1:] + f" ({variable_unit_print})"

    # If the CDM metrics are not to be included in the plots, we need only the last three columns of the DataFrame, containing the environmental impacts
    if not include_CDM_metrics_in_plots:
        SA_results = SA_results.iloc[:, -3:]
        SA_results_normalized = SA_results_normalized.iloc[:, -3:]
        SA_results_elasticity = SA_results_elasticity.iloc[:, -3:]

    # Plot the sensitivity analysis results per impact
    plot_SA_results_per_impact(SA_results, reference_values, variable_name_print, x_axis_label, include_scenario_lines_in_plots, figure_size)

    # Plot the normalized results
    plot_SA_results_normalized(
        SA_results_normalized,
        reference_values,
        ref_value_in_SA_results,
        variable_name_print,
        variable_unit_print,
        x_axis_label,
        round_to,
        include_scenario_lines_in_plots,
        figure_size,
    )

    # Plot the elasticity of the cooling demand and environmental impacts with respect to the independent variable
    plot_SA_results_elasticities(
        SA_results_elasticity,
        reference_values,
        variable_name_print,
        x_axis_label,
        include_scenario_lines_in_plots,
        figure_size,
    )

    # Return the elasticities at the value closest to the reference value
    print(
        f"The elasticities of the cooling demand and environmental impacts with respect to the {variable_name_print} at {round(reference_values['SQ'])} {variable_unit_print} are:",
    )
    return SA_results_elasticity_full.loc[ref_value_in_SA_results]


def post_process_SA_cooling_tech_mix(
    SA_results: pd.DataFrame,
    cooling_tech_one: str,
    cooling_tech_two: str,
    show_plots: bool = True,
    figure_size: tuple[float, float] = (6, 6),
) -> pd.DataFrame:
    """Post-process the results of a sensitivity analysis on the cooling technology mix.

    Args:
        SA_results (pd.DataFrame): The DataFrame containing the results of the sensitivity analysis.
        cooling_tech_one (str): The name of the first cooling technology. Can be one of "ASHP", "GSHP", "WSHP", "chiller", "AC_split", or "AC_mobile".
        cooling_tech_two (str): The name of the second cooling technology. Can be one of "ASHP", "GSHP", "WSHP", "chiller", "AC_split", or "AC_mobile".
        show_plots (bool, optional): Whether or not to show the plots to the console. Defaults to True.
        figure_size (tuple[float, float], optional): The size of the figures, in inches. Defaults to (6, 6).

    Returns:
        pd.DataFrame: The DataFrame containing the elasticity of the cooling demand and environmental impacts with respect to the mix split between the two cooling technologies.
    """
    # Calculate the elasticity of the environmental impacts with respect to the cooling technology mix
    SA_results_elasticity = SA_results.pct_change() / SA_results.index.to_series().pct_change().to_numpy()[:, np.newaxis]

    # Rename all the columns in SA_results_elasticity to remove impact units from the labels
    SA_results_elasticity.columns = [col.split("(")[0].strip() for col in SA_results_elasticity.columns]

    # Drop the cooling energy demand and peak cooling power demand columns, they are not influenced by the cooling technology mix
    SA_results = SA_results.iloc[:, -3:]
    SA_results_elasticity = SA_results_elasticity.iloc[:, -3:]

    # Define the x-axis label
    x_axis_label = f"Cooling technology mix: share of {cooling_tech_one} vs. {cooling_tech_two} (%)"

    # Plot the sensitivity analysis results per impact
    SA_results.plot(subplots=True, figsize=figure_size, sharex=True)  # Initialize the plot
    plt.suptitle("Influence of cooling technology mix on impacts")  # Add a title to the plot
    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.tight_layout()  # Make sure the subplots don't overlap
    plt.savefig(
        "data/output/images/SA/SA_results_cooling_technology_mix.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    if show_plots:
        plt.show()  # Show the plot to the console
    plt.close()

    # Plot the elasticity of the cooling demand and environmental impacts with respect to the independent variable
    SA_results_elasticity.plot(subplots=False, figsize=figure_size, sharex=True)  # Initialize the plot
    plt.title(f"Elasticity of impacts with respect to the cooling technology mix: share of {cooling_tech_one} vs. {cooling_tech_two}")  # Add a title to the plot
    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.legend()  # Add a legend to the plot
    plt.tight_layout()  # Make sure the subplots don't overlap
    plt.savefig(
        "data/output/images/SA/SA_results_cooling_technology_mix_elasticities.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    if show_plots:
        plt.show()  # Show the plot to the console
    plt.close()

    return SA_results_elasticity.mean().to_frame(name="mean")
