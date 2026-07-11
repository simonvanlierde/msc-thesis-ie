"""This file contains functions for plotting figures for the results of the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cdm.constants import HOURS_PER_YEAR, IMAGE_OUTPUT_DIR

if TYPE_CHECKING:
    from datetime import datetime


def get_indices_from_dates(
    time_series: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
) -> tuple[int, int]:
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


# Thermal flow figures


def _plot_thermal_flows(
    building: pd.Series,
    time_series: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    subplot_ax: plt.Axes | None,
    unit_print: str,
    sort_hours: bool,
    save_figure: bool,
    scenario: str,
    building_subset_name: str,
    *,
    subplot_title: str,
    standalone_subject: str,
    filename_group: str,
    filename_id: str,
) -> None:
    """Plot the thermal flows for a building between two dates, shared by the building-type and energy-class variants."""
    floor_area = building["floor_area_total_m2"]

    # Convert the datetimes to a readable format
    start_date_str = pd.to_datetime(start_date).strftime("%B %-d, %Y")
    end_date_str = pd.to_datetime(end_date).strftime("%B %-d, %Y")

    # Fetch the start and end index from the time series
    start_index, end_index = get_indices_from_dates(time_series, start_date, end_date)

    # Create a custom color map for the thermal flows (first five tab10 colors, then blue for the cooling demand)
    colors_thermal_flows = [*[plt.get_cmap("tab10", 10)(i) for i in range(5)], "blue"]

    # Construct a heat flow to label dictionary
    heat_flow_to_label = {
        "Q_transmission_Wh": "Transmission",
        "Q_infiltration_Wh": "Infiltration",
        "Q_ventilation_Wh": "Ventilation",
        "Q_solar_radiation_Wh": "Solar radiation",
        "Q_internal_heat_Wh": "Internal heat",
        "Q_cooling_demand_Wh": "Cooling demand",
    }

    # Plot each heat flow on the given subplot, or on the current axis when plotting standalone
    ax = subplot_ax or plt.gca()
    for (heat_flow, label), color in zip(heat_flow_to_label.items(), colors_thermal_flows, strict=True):
        alpha = 0.7 if heat_flow == "Q_cooling_demand_Wh" else 0.5
        series = building[heat_flow][start_index:end_index] / floor_area
        if sort_hours:  # Optionally sort the hours
            series = np.sort(series)[::-1]
        ax.plot(series, label=label, alpha=alpha, color=color)

    if subplot_ax:
        subplot_ax.set_title(subplot_title)
        return

    plt.title(
        f"Cooling demand components for {standalone_subject}\n between {start_date_str} and {end_date_str}{', sorted' if sort_hours else ''}",
    )
    plt.ylabel(f"Thermal flow ({unit_print})")
    plt.xlabel("Hour")
    plt.legend()
    plt.tight_layout()

    if save_figure:
        plt.savefig(
            f"{IMAGE_OUTPUT_DIR}/thermal_flows/thermal_components_{'sorted' if sort_hours else ''}_{filename_group}_{filename_id}_{start_date}_to_{end_date}_{scenario}_{building_subset_name}.png",
            dpi=300,
        )
    plt.show()


def plot_thermal_flows_for_end_use(
    building: pd.Series,
    time_series: pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    subplot_ax: plt.Axes | None = None,
    unit_print: str = "W/m2",
    sort_hours: bool = False,
    save_figure: bool = False,
    scenario: str = "SQ",
    building_subset_name: str = "sample",
) -> None:
    """Plot the thermal flows for one end-use group (residential or office) between two dates."""
    end_use = str(building["end_use"]).capitalize()
    _plot_thermal_flows(
        building,
        time_series,
        start_date,
        end_date,
        subplot_ax,
        unit_print,
        sort_hours,
        save_figure,
        scenario,
        building_subset_name,
        subplot_title=end_use,
        standalone_subject=f"{end_use.lower()} buildings",
        filename_group="end_use",
        filename_id=str(building["end_use"]),
    )


def _plot_cooling_loads(
    building: pd.Series,
    years_int_time_series: int,
    peak_power_percentile_cap: int,
    subplot_ax: plt.Axes | None,
    save_figure: bool,
    scenario: str,
    building_subset_name: str,
    *,
    subplot_title: str,
    standalone_subject: str,
    filename_group: str,
    filename_id: str,
) -> None:
    """Plot the average annual cooling load distribution, shared by the building-type and energy-class variants."""
    floor_area = building["floor_area_total_m2"]

    # Fetch the average cooling demand per hour per m2 of floor area in a year, averaged over all years in the time series
    Q_cooling_demand_average = (
        building["Q_cooling_demand_Wh"].reshape(years_int_time_series, HOURS_PER_YEAR).mean(axis=0) / floor_area
    )

    # Sort the average cooling demand in descending order
    Q_cooling_demand_sorted = np.sort(Q_cooling_demand_average)[::-1]

    # Figure out the nth percentile of the average cooling power demand
    peak_power_percentile = np.percentile(Q_cooling_demand_sorted, peak_power_percentile_cap)

    # Cap the sorted energy array at the power percentile cap, then re-sort in descending order
    Q_cooling_demand_capped = np.where(
        Q_cooling_demand_sorted > peak_power_percentile,
        peak_power_percentile,
        Q_cooling_demand_sorted,
    )
    Q_cooling_demand_capped_sorted = np.sort(Q_cooling_demand_capped)[::-1]

    # Plot on the given subplot, or on the current axis when plotting standalone
    ax = subplot_ax or plt.gca()
    ax.plot(Q_cooling_demand_average, label="Cooling energy demand", alpha=0.5)
    ax.plot(Q_cooling_demand_sorted, label="Cooling energy demand, sorted", color="blue", linestyle="--", alpha=0.5)
    ax.hlines(
        peak_power_percentile,
        0,
        HOURS_PER_YEAR,
        label=f"{int(peak_power_percentile_cap)}th percentile of cooling demand",
        color="red",
        linestyle="--",
    )
    ax.plot(Q_cooling_demand_capped_sorted, label="Cooling energy demand, capped", color="blue")
    ax.fill_between(range(HOURS_PER_YEAR), Q_cooling_demand_sorted, color="blue", alpha=0.2)
    ax.fill_between(range(HOURS_PER_YEAR), Q_cooling_demand_capped_sorted, color="blue", alpha=0.2)

    if subplot_ax:
        subplot_ax.set_title(subplot_title)
        return

    plt.title(f"Annual cooling demand load for {standalone_subject}, average over {years_int_time_series} years")
    plt.xlabel("Hour")
    plt.ylabel("Cooling energy demand (Wh/m2)")
    plt.legend()
    plt.tight_layout()

    if save_figure:
        plt.savefig(
            f"{IMAGE_OUTPUT_DIR}/thermal_flows/cooling_load_{filename_group}_{filename_id}_{scenario}_{building_subset_name}.png",
            dpi=300,
        )
    plt.show()


def plot_cooling_loads_for_end_use(
    building: pd.Series,
    years_int_time_series: int = 5,
    peak_power_percentile_cap: int = 98,
    subplot_ax: plt.Axes | None = None,
    save_figure: bool = False,
    scenario: str = "SQ",
    building_subset_name: str = "sample",
) -> None:
    """Plot the average annual cooling load distribution for one end-use group (residential or office)."""
    end_use = str(building["end_use"]).capitalize()
    _plot_cooling_loads(
        building,
        years_int_time_series,
        peak_power_percentile_cap,
        subplot_ax,
        save_figure,
        scenario,
        building_subset_name,
        subplot_title=end_use,
        standalone_subject=f"{end_use.lower()} buildings",
        filename_group="end_use",
        filename_id=str(building["end_use"]),
    )


