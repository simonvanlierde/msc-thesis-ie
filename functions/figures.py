"""This file contains functions for plotting figures for the results of the cooling demand model.

@author: Simon van Lierde
"""


from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_indices_from_dates(time_series: pd.DataFrame, start_date: str | datetime, end_date: str | datetime) -> tuple[int, int]:
    """Return the indices of the rows corresponding to the given start and end dates.

    Args:
        time_series (pd.DataFrame): DataFrame containing the 'date' column.
        start_date (str or datetime): Start date.
        end_date (str or datetime): End date.

    Returns:
        tuple[int, int]: (start_index, end_index)
    """
    # Convert start and end dates to datetime if they are strings
    start_index = time_series[time_series["date"] == pd.to_datetime(start_date)].index[0]
    end_index = time_series[time_series["date"] == pd.to_datetime(end_date)].index[0]

    return start_index, end_index


def plot_thermal_flows_for_building_types(
    building: pd.Series,
    time_series: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    subplot_ax: plt.Axes = None,
    unit_print: str = "W/m2",
    sort_hours: bool = False,
    save_figure: bool = False,
    scenario: str = "SQ",
    building_subset_name: str = "sample",
) -> None:
    """Plot the thermal flows for a building type between two dates.

    Args:
        building (pd.Series): The building to plot the thermal flows for. It should contain thermal flow time series as np.arrays.
        time_series (pd.DataFrame): The time series DataFrame used to generate the thermal flows. It should contain a "date" column.
        start_date (str | datetime): The start date of the time series to plot.
        end_date (str | datetime): The end date of the time series to plot.
        subplot_ax (plt.Axes, optional): The subplot to plot the thermal flows on. Defaults to None.
        unit_print (str, optional): The unit of the thermal flows to print. Defaults to "W/m2".
        sort_hours (bool, optional): Whether to sort the hours in descending order. Defaults to False.
        save_figure (bool, optional): Whether to save the plot to a file. Defaults to False.
        scenario (str, optional): The scenario used to generate the thermal flows, for filename purposes. Defaults to "SQ".
        building_subset_name (str, optional): The building subset used to generate the thermal flows, for filename purposes. Defaults to "sample".
    """
    # Fetch the floor area of the building
    floor_area = building["floor_area_total_m2"]

    # Fetch the building type as a readable string
    building_type = f"{building['building_type_int']!s} ({building['building_type']})"

    # Convert the datetimes to a readable format
    start_date_str = pd.to_datetime(start_date).strftime("%B %-d, %Y")
    end_date_str = pd.to_datetime(end_date).strftime("%B %-d, %Y")

    # Fetch the start and end index from the time series
    start_index, end_index = get_indices_from_dates(time_series, start_date, end_date)

    # Create a custom color map for the thermal flows
    tab10 = plt.cm.get_cmap("tab10", 10)
    first_5_colors = [tab10(i) for i in range(5)]
    colors_thermal_flows = [*first_5_colors, "blue"]

    # Construct a heat flow to label dictionary
    heat_flow_to_label = {
        "Q_transmission_Wh": "Transmission",
        "Q_infiltration_Wh": "Infiltration",
        "Q_ventilation_Wh": "Ventilation",
        "Q_solar_radiation_Wh": "Solar radiation",
        "Q_internal_heat_Wh": "Internal heat",
        "Q_cooling_demand_Wh": "Cooling demand",
    }

    if subplot_ax:
        for (heat_flow, label), color in zip(heat_flow_to_label.items(), colors_thermal_flows, strict=True):
            alpha = 0.7 if heat_flow == "Q_cooling_demand_Wh" else 0.5
            if sort_hours:  # Optionally sort the hours
                subplot_ax.plot(np.sort(building[heat_flow][start_index:end_index])[::-1] / floor_area, label=label, alpha=alpha, color=color)
            else:
                subplot_ax.plot(building[heat_flow][start_index:end_index] / floor_area, label=label, alpha=alpha, color=color)

        subplot_ax.set_title(f"Building type: {building_type}")

    else:
        for (heat_flow, label), color in zip(heat_flow_to_label.items(), colors_thermal_flows, strict=True):
            alpha = 0.7 if heat_flow == "Q_cooling_demand_Wh" else 0.5
            if sort_hours:  # Optionally sort the hours
                plt.plot(np.sort(building[heat_flow][start_index:end_index])[::-1] / floor_area, label=label, alpha=alpha, color=color)
            else:
                plt.plot(building[heat_flow][start_index:end_index] / floor_area, label=label, alpha=alpha, color=color)

        plt.title(f"Cooling demand components for building type {building_type}\n between {start_date_str} and {end_date_str}{', sorted' if sort_hours else ''}")
        plt.ylabel(f"Thermal flow ({unit_print})")
        plt.xlabel("Hour")
        plt.legend()
        plt.tight_layout()

        if save_figure:
            plt.savefig(
                f"data/output/images/thermal_flows/thermal_components_{'sorted' if sort_hours else ''}_building_type_{building['building_type_int']!s}_{start_date}_to_{end_date}_{scenario}_{building_subset_name}.png",
                dpi=300,
            )
        plt.show()


def plot_thermal_flows_for_energy_classes(
    building: pd.Series,
    time_series: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    subplot_ax: plt.Axes = None,
    unit_print: str = "W/m2",
    sort_hours: bool = False,
    save_figure: bool = False,
    scenario: str = "SQ",
    building_subset_name: str = "sample",
) -> None:
    """Plot the thermal flows for an energy class between two dates.

    Args:
        building (pd.Series): The building to plot the thermal flows for. It should contain thermal flow time series as np.arrays.
        time_series (pd.DataFrame): The time series DataFrame used to generate the thermal flows. It should contain a "date" column.
        start_date (str | datetime): The start date of the time series to plot.
        end_date (str | datetime): The end date of the time series to plot.
        subplot_ax (plt.Axes, optional): The subplot to plot the thermal flows on. Defaults to None.
        unit_print (str, optional): The unit of the thermal flows to print. Defaults to "W/m2".
        sort_hours (bool, optional): Whether to sort the hours in descending order. Defaults to False.
        save_figure (bool, optional): Whether to save the plot to a file. Defaults to False.
        scenario (str, optional): The scenario used to generate the thermal flows, for filename purposes. Defaults to "SQ".
        building_subset_name (str, optional): The building subset used to generate the thermal flows, for filename purposes. Defaults to "sample".
    """
    # Fetch the floor area of the building
    floor_area = building["floor_area_total_m2"]

    # Fetch the energy class as a readable string
    energy_class = building["energy_class"].replace("-", " - ")

    # Convert the datetimes to a readable format
    start_date_str = pd.to_datetime(start_date).strftime("%B %-d, %Y")
    end_date_str = pd.to_datetime(end_date).strftime("%B %-d, %Y")

    # Fetch the start and end index from the time series
    start_index, end_index = get_indices_from_dates(time_series, start_date, end_date)

    # Create a custom color map for the thermal flows
    tab10 = plt.cm.get_cmap("tab10", 10)
    first_5_colors = [tab10(i) for i in range(5)]
    colors_thermal_flows = [*first_5_colors, "blue"]

    # Construct a heat flow to label dictionary
    heat_flow_to_label = {
        "Q_transmission_Wh": "Transmission",
        "Q_infiltration_Wh": "Infiltration",
        "Q_ventilation_Wh": "Ventilation",
        "Q_solar_radiation_Wh": "Solar radiation",
        "Q_internal_heat_Wh": "Internal heat",
        "Q_cooling_demand_Wh": "Cooling demand",
    }

    if subplot_ax:
        for (heat_flow, label), color in zip(heat_flow_to_label.items(), colors_thermal_flows, strict=True):
            alpha = 0.7 if heat_flow == "Q_cooling_demand_Wh" else 0.5
            if sort_hours:  # Optionally sort the hours
                subplot_ax.plot(np.sort(building[heat_flow][start_index:end_index])[::-1] / floor_area, label=label, alpha=alpha, color=color)
            else:
                subplot_ax.plot(building[heat_flow][start_index:end_index] / floor_area, label=label, alpha=alpha, color=color)

        subplot_ax.set_title(f"Energy labels: {energy_class}")

    else:
        for (heat_flow, label), color in zip(heat_flow_to_label.items(), colors_thermal_flows, strict=True):
            alpha = 0.7 if heat_flow == "Q_cooling_demand_Wh" else 0.5
            if sort_hours:  # Optionally sort the hours
                plt.plot(np.sort(building[heat_flow][start_index:end_index])[::-1] / floor_area, label=label, alpha=alpha, color=color)
            else:
                plt.plot(building[heat_flow][start_index:end_index] / floor_area, label=label, alpha=alpha, color=color)

        plt.title(f"Cooling demand components for buildings with energy labels {energy_class}\n between {start_date_str} and {end_date_str}{', sorted' if sort_hours else ''}")
        plt.ylabel(f"Thermal flow ({unit_print})")
        plt.xlabel("Hour")
        plt.legend()
        plt.tight_layout()

        if save_figure:
            plt.savefig(
                f"data/output/images/thermal_flows/thermal_components_{'sorted' if sort_hours else ''}_energy_class_{building['energy_class_int']!s}_{start_date}_to_{end_date}_{scenario}_{building_subset_name}.png",
                dpi=300,
            )
        plt.show()


def plot_cooling_loads_for_building_types(
    building: pd.Series,
    years_int_time_series: int = 5,
    peak_power_percentile_cap: int = 98,
    subplot_ax: plt.Axes = None,
    save_figure: bool = False,
    scenario: str = "SQ",
    building_subset_name: str = "sample",
) -> None:
    """Plot the average annual cooling load distribution for a building type.

    Args:
        building (pd.Series): The building to plot the cooling load distribution for. It should contain thermal flow time series as np.arrays.
        years_int_time_series (int, optional): The number of years in the time series. Defaults to 5.
        peak_power_percentile_cap (int, optional): The percentile to cap the peak power at. Defaults to 98.
        subplot_ax (plt.Axes, optional): The subplot to plot the cooling load distribution on. Defaults to None.
        save_figure (bool, optional): Whether to save the plot to a file. Defaults to False.
        scenario (str, optional): The scenario used to generate the cooling load distribution, for filename purposes. Defaults to "SQ".
        building_subset_name (str, optional): The building subset used to generate the cooling load distribution, for filename purposes. Defaults to "sample".
    """
    # Fetch the floor area of the building
    floor_area = building["floor_area_total_m2"]

    # Fetch the building type as a readable string
    building_type = f"{building['building_type_int']!s} ({building['building_type']})"

    # Fetch the average cooling demand per hour per square meter of floor area in a year across the full set of years in the time series
    Q_cooling_demand_average = building["Q_cooling_demand_Wh"].reshape(years_int_time_series, 8760).mean(axis=0) / floor_area

    # Sort the average cooling demand in descending order
    Q_cooling_demand_sorted = np.sort(Q_cooling_demand_average)[::-1]

    # Figure out the nth percentile of the average cooling power demand
    peak_power_percentile = np.percentile(Q_cooling_demand_sorted, peak_power_percentile_cap)

    # Cap the sorted energy array at the power percentile cap
    Q_cooling_demand_capped = np.where(Q_cooling_demand_sorted > peak_power_percentile, peak_power_percentile, Q_cooling_demand_sorted)

    # Sort the capped cooling demand in descending order
    Q_cooling_demand_capped_sorted = np.sort(Q_cooling_demand_capped)[::-1]

    if subplot_ax:
        subplot_ax.plot(Q_cooling_demand_average, label="Cooling energy demand", alpha=0.5)
        subplot_ax.plot(Q_cooling_demand_sorted, label="Cooling energy demand, sorted", color="blue", linestyle="--", alpha=0.5)
        subplot_ax.hlines(
            peak_power_percentile,
            0,
            8760,
            label=f"{int(peak_power_percentile_cap)}th percentile of cooling demand",
            color="red",
            linestyle="--",
        )
        subplot_ax.plot(
            Q_cooling_demand_capped_sorted,
            label="Cooling energy demand, capped",
            color="blue",
        )
        subplot_ax.fill_between(range(8760), Q_cooling_demand_sorted, color="blue", alpha=0.2)
        subplot_ax.fill_between(range(8760), Q_cooling_demand_capped_sorted, color="blue", alpha=0.2)
        subplot_ax.set_title(f"Building type: {building_type}")

    else:
        plt.plot(Q_cooling_demand_average, label="Cooling energy demand", alpha=0.5)
        plt.plot(Q_cooling_demand_sorted, label="Cooling energy demand, sorted", color="blue", linestyle="--", alpha=0.5)
        plt.hlines(
            peak_power_percentile,
            0,
            8760,
            label=f"{int(peak_power_percentile_cap)}th percentile of cooling demand",
            color="red",
            linestyle="--",
        )
        plt.plot(
            Q_cooling_demand_capped_sorted,
            label="Cooling energy demand, capped",
            color="blue",
        )
        plt.fill_between(range(8760), Q_cooling_demand_sorted, color="blue", alpha=0.2)
        plt.fill_between(range(8760), Q_cooling_demand_capped_sorted, color="blue", alpha=0.2)
        plt.title(f"Annual cooling demand load for building type {building_type}, average over {years_int_time_series} years")
        plt.xlabel("Hour")
        plt.ylabel("Cooling energy demand (Wh/m2)")
        plt.legend()
        plt.tight_layout()

        if save_figure:
            plt.savefig(f"data/output/images/thermal_flows/cooling_load_building_type_{building['building_type_int']!s}_{scenario}_{building_subset_name}.png", dpi=300)
        plt.show()


def plot_cooling_loads_for_energy_classes(
    building: pd.Series,
    years_int_time_series: int = 5,
    peak_power_percentile_cap: int = 98,
    subplot_ax: plt.Axes = None,
    save_figure: bool = False,
    scenario: str = "SQ",
    building_subset_name: str = "sample",
) -> None:
    """Plot the average annual cooling load distribution for an energy class.

    Args:
        building (pd.Series): The building to plot the cooling load distribution for. It should contain thermal flow time series as np.arrays.
        years_int_time_series (int, optional): The number of years in the time series. Defaults to 5.
        peak_power_percentile_cap (int, optional): The percentile to cap the peak power at. Defaults to 98.
        subplot_ax (plt.Axes, optional): The subplot to plot the cooling load distribution on. Defaults to None.
        save_figure (bool, optional): Whether to save the plot to a file. Defaults to False.
        scenario (str, optional): The scenario used to generate the cooling load distribution, for filename purposes. Defaults to "SQ".
        building_subset_name (str, optional): The building subset used to generate the cooling load distribution, for filename purposes. Defaults to "sample".
    """
    # Fetch the floor area of the building
    floor_area = building["floor_area_total_m2"]

    # Fetch the energy class as a readable string
    energy_class = building["energy_class"].replace("-", " - ")

    # Fetch the average cooling demand per hour per square meter of floor area in a year across the full set of years in the time series
    Q_cooling_demand_average = building["Q_cooling_demand_Wh"].reshape(years_int_time_series, 8760).mean(axis=0) / floor_area

    # Sort the average cooling demand in descending order
    Q_cooling_demand_sorted = np.sort(Q_cooling_demand_average)[::-1]

    # Figure out the nth percentile of the average cooling power demand
    peak_power_percentile = np.percentile(Q_cooling_demand_sorted, peak_power_percentile_cap)

    # Cap the sorted energy array at the power percentile cap
    Q_cooling_demand_capped = np.where(Q_cooling_demand_sorted > peak_power_percentile, peak_power_percentile, Q_cooling_demand_sorted)

    # Sort the capped cooling demand in descending order
    Q_cooling_demand_capped_sorted = np.sort(Q_cooling_demand_capped)[::-1]

    if subplot_ax:
        subplot_ax.plot(Q_cooling_demand_average, label="Cooling energy demand", alpha=0.5)
        subplot_ax.plot(Q_cooling_demand_sorted, label="Cooling energy demand, sorted", color="blue", linestyle="--", alpha=0.5)
        subplot_ax.hlines(
            peak_power_percentile,
            0,
            8760,
            label=f"{int(peak_power_percentile_cap)}th percentile of cooling demand",
            color="red",
            linestyle="--",
        )
        subplot_ax.plot(
            Q_cooling_demand_capped_sorted,
            label="Cooling energy demand, capped",
            color="blue",
        )
        subplot_ax.fill_between(range(8760), Q_cooling_demand_sorted, color="blue", alpha=0.2)
        subplot_ax.fill_between(range(8760), Q_cooling_demand_capped_sorted, color="blue", alpha=0.2)
        subplot_ax.set_title(f"Energy class: {energy_class}")

    else:
        plt.plot(Q_cooling_demand_average, label="Cooling energy demand", alpha=0.5)
        plt.plot(Q_cooling_demand_sorted, label="Cooling energy demand, sorted", color="blue", linestyle="--", alpha=0.5)
        plt.hlines(
            peak_power_percentile,
            0,
            8760,
            label=f"{int(peak_power_percentile_cap)}th percentile of cooling demand",
            color="red",
            linestyle="--",
        )
        plt.plot(
            Q_cooling_demand_capped_sorted,
            label="Cooling energy demand, capped",
            color="blue",
        )
        plt.fill_between(range(8760), Q_cooling_demand_sorted, color="blue", alpha=0.2)
        plt.fill_between(range(8760), Q_cooling_demand_capped_sorted, color="blue", alpha=0.2)
        plt.title(f"Annual cooling demand load for energy class {energy_class}, average over {years_int_time_series} years")
        plt.xlabel("Hour")
        plt.ylabel("Cooling energy demand (Wh/m2)")
        plt.legend()
        plt.tight_layout()

        if save_figure:
            plt.savefig(f"data/output/images/thermal_flows/cooling_load_energy_class_{building['energy_class_int']!s}_{scenario}_{building_subset_name}.png", dpi=300)
        plt.show()
