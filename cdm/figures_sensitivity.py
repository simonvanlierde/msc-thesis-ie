"""Plotting helpers for the sensitivity analysis of the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from cdm.sensitivity_analysis import calculate_elasticity_for_SA_results, normalize_SA_results

if TYPE_CHECKING:
    import pandas as pd

# Directory for sensitivity-analysis figures. Overridable via the SA_IMAGE_DIR
# environment variable so the Snakemake pipeline can write under results/ while
# the notebook keeps the original location.
SA_IMAGE_DIR = os.environ.get("SA_IMAGE_DIR", "data/output/images/SA")


def _draw_future_scenario_lines(ax: plt.Axes, reference_values: dict[str, float]) -> None:
    """Draw the four future-scenario reference lines (2030, 2050 L/M/H) on an axis."""
    ax.axvline(x=reference_values["2030"], color="gray", linestyle="dotted", alpha=0.7, label="Value in 2030 scenario")
    ax.axvline(
        x=reference_values["2050_L"],
        color="gold",
        linestyle="dotted",
        alpha=0.7,
        label="Value in 2050 L scenario",
    )
    ax.axvline(
        x=reference_values["2050_M"],
        color="darkorange",
        linestyle="dotted",
        alpha=0.7,
        label="Value in 2050 M scenario",
    )
    ax.axvline(
        x=reference_values["2050_H"],
        color="red",
        linestyle="dotted",
        alpha=0.7,
        label="Value in 2050 H scenario",
    )


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
        ax.axvline(
            x=reference_values["SQ"],
            color="dimgray",
            linestyle="dashed",
            alpha=0.7,
            label="Value in reference scenario",
        )

        if include_scenario_lines_in_plots:
            _draw_future_scenario_lines(ax, reference_values)

    plt.gcf().axes[0].legend()  # Add a legend for the reference value to the first subplot
    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.tight_layout()  # Make sure the subplots don't overlap

    plt.savefig(
        f"{SA_IMAGE_DIR}/SA_results_{variable_name_print.replace(' ', '_')}{'_with_scenario_lines' if include_scenario_lines_in_plots else ''}.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    plt.show()  # Show the plot to the console
    plt.close()


def plot_SA_results_normalized(
    SA_results_normalized: pd.DataFrame,
    reference_values: dict[str, float],
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
    plt.axvline(
        x=ref_value_in_SA_results,
        color="dimgray",
        linestyle="dashed",
        alpha=0.7,
        label="Value in reference scenario",
    )

    if include_scenario_lines_in_plots:
        _draw_future_scenario_lines(plt.gca(), reference_values)

    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.legend()  # Add a legend to the plot
    plt.tight_layout()  # Make sure the subplots don't overlap
    plt.savefig(
        f"{SA_IMAGE_DIR}/SA_results_{variable_name_print.replace(' ', '_')}_normalized{'_with_scenario_lines' if include_scenario_lines_in_plots else ''}.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    plt.show()  # Show the plot to the console
    plt.close()


def plot_SA_results_elasticities(
    SA_results_elasticities: pd.DataFrame,
    reference_values: dict[str, float],
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

    plt.axvline(
        x=reference_values["SQ"],
        color="dimgray",
        linestyle="dashed",
        alpha=0.7,
        label="Value in reference scenario",
    )

    if include_scenario_lines_in_plots:
        _draw_future_scenario_lines(plt.gca(), reference_values)

    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.legend()  # Add a legend to the plot
    plt.tight_layout()  # Make sure the subplots don't overlap

    plt.savefig(
        f"{SA_IMAGE_DIR}/SA_results_{variable_name_print.replace(' ', '_')}_elasticities{'_with_scenario_lines' if include_scenario_lines_in_plots else ''}.png",
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
    plot_SA_results_per_impact(
        SA_results,
        reference_values,
        variable_name_print,
        x_axis_label,
        include_scenario_lines_in_plots,
        figure_size,
    )

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

    print(  # noqa: T201 -- intentional user-facing console output
        f"The elasticities of the cooling demand and environmental impacts with respect to the {variable_name_print} at {round(reference_values['SQ'])} {variable_unit_print} are:",
    )

    # Return the elasticities at the value closest to the reference value
    return SA_results_elasticity_full.loc[ref_value_in_SA_results]


def post_process_SA_cooling_tech_mix(
    SA_results: pd.DataFrame,
    cooling_tech_one: str,
    cooling_tech_two: str,
    show_plots: bool = True,
    figure_size: tuple[float, float] = (6, 6),
    image_dir: str = SA_IMAGE_DIR,
) -> pd.DataFrame:
    """Post-process the results of a sensitivity analysis on the cooling technology mix.

    Args:
        SA_results (pd.DataFrame): The DataFrame containing the results of the sensitivity analysis.
        cooling_tech_one (str): The name of the first cooling technology. Can be one of "ASHP", "GSHP", "WSHP", "chiller", "AC_split", or "AC_mobile".
        cooling_tech_two (str): The name of the second cooling technology. Can be one of "ASHP", "GSHP", "WSHP", "chiller", "AC_split", or "AC_mobile".
        show_plots (bool, optional): Whether or not to show the plots to the console. Defaults to True.
        figure_size (tuple[float, float], optional): The size of the figures, in inches. Defaults to (6, 6).
        image_dir (str, optional): Directory the figures are written to. Defaults to SA_IMAGE_DIR, so the notebook keeps its behaviour; the Snakemake rule passes its declared output directory.

    Returns:
        pd.DataFrame: The DataFrame containing the elasticity of the cooling demand and environmental impacts with respect to the mix split between the two cooling technologies.
    """
    # Calculate the elasticity of the environmental impacts with respect to the cooling technology mix
    SA_results_elasticity = (
        SA_results.pct_change() / SA_results.index.to_series().pct_change().to_numpy()[:, np.newaxis]
    )

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
        f"{image_dir}/SA_results_cooling_technology_mix.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    if show_plots:
        plt.show()  # Show the plot to the console
    plt.close()

    # Plot the elasticity of the cooling demand and environmental impacts with respect to the independent variable
    SA_results_elasticity.plot(subplots=False, figsize=figure_size, sharex=True)  # Initialize the plot
    plt.title(
        f"Elasticity of impacts with respect to the cooling technology mix: share of {cooling_tech_one} vs. {cooling_tech_two}",
    )  # Add a title to the plot
    plt.xlabel(x_axis_label)  # Add the x-axis label
    plt.legend()  # Add a legend to the plot
    plt.tight_layout()  # Make sure the subplots don't overlap
    plt.savefig(
        f"{image_dir}/SA_results_cooling_technology_mix_elasticities.png",
        dpi=300,
        bbox_inches="tight",
    )  # Save the figure
    if show_plots:
        plt.show()  # Show the plot to the console
    plt.close()

    return SA_results_elasticity.mean().to_frame(name="mean")
