"""Thermodynamic calculation functions used in the cooling demand model.

@author: Simon van Lierde
"""

from __future__ import annotations

# Import packages
from typing import TYPE_CHECKING

import numpy as np

from functions.constants import HOURS_PER_YEAR, SOLAR_DIRECTIONS

if TYPE_CHECKING:
    import pandas as pd

# Parameter derivation functions


def R_to_U(Rc: float, alfa_i: float = 7.5, alfa_o: float = 27.5) -> float:
    """R_to_U converts the thermal resistance of a building element to the transmittance of that element (U).

    Args:
        Rc (float): Thermal resistance of the building element in m2K/W
        alfa_i (float, optional): Combined heat transfer coefficient of convection and radiation on the inside in W/m2K. Defaults to 7.5.
        alfa_o (float, optional): Combined heat transfer coefficient of convection and radiation on the outside in W/m2K. Defaults to 27.5.

    Returns:
        float: Transmittance of the building element in W/m2K.
    """
    return 1 / (1 / alfa_i + Rc + 1 / alfa_o)  # Transmittance of the building element in W/m2K


# Thermal flow calculation functions


def calc_Q_transmission(
    building: pd.Series,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_transmission calculates the transmission heat flow Q of a building in Wh across a given time series.

    Args:
        building (pd.Series): The building row for which the transmission heat flows are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data for the building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The total transmission heat flow in Wh for each hour in the time series.
    """
    # Load building data from row
    window_area = building["window_area_total_m2"]  # The window area of the building in m2
    wall_area = building["wall_area_total_m2"]  # The wall area of the building in m2
    floor_area = building["floor_area_ground_m2"]  # The floor area of the building in m2
    Rc_wall = building["Rc_wall_m2K_W"]  # The thermal transmittance of the walls in W/m2K
    Rc_roof = building["Rc_roof_m2K_W"]  # The thermal transmittance of the roof in W/m2K
    Rc_floor = building["Rc_floor_m2K_W"]  # The thermal transmittance of the floor in W/m2K
    U_window = building["U_window_W_m2K"]  # The thermal transmittance of the windows in W/m2K

    # The roof area is assumed to be equal to the ground floor area
    roof_area = floor_area

    # Load temperature difference between in- and outdoor from time series
    delta_T_air = time_series["T_outdoor_minus_indoor_C"]

    # Load global parameters
    T_thresh_C = global_parameters["T_thresh_C"]  # The cooling threshold temperature in °C
    T_sub_C = global_parameters["T_sub_C"]  # The subsurface temperature in °C
    alfa_i = global_parameters[
        "alfa_i"
    ]  # Combined heat transfer coefficient of convection and radiation on the inside in W/m2K
    alfa_o = global_parameters[
        "alfa_o"
    ]  # Combined heat transfer coefficient of convection and radiation on the outside in W/m2K

    # Determine U values for the walls, floors, and roofs
    U_wall, U_roof, U_floor = (R_to_U(Rc, alfa_i, alfa_o) for Rc in [Rc_wall, Rc_roof, Rc_floor])

    # Calculate the transmission heat flow for each barrier type
    Q_transmission_window = U_window * window_area * delta_T_air
    Q_transmission_wall = U_wall * wall_area * delta_T_air
    Q_transmission_roof = U_roof * roof_area * delta_T_air
    Q_transmission_floor = U_floor * floor_area * (T_sub_C - T_thresh_C)

    # Calculate the total transmission heat flow in Wh
    return Q_transmission_window + Q_transmission_wall + Q_transmission_roof + Q_transmission_floor


def calc_Q_infiltration(
    building: pd.Series,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_infiltration calculates the infiltration heat flows of a building in Wh across a given time series.

    Args:
        building (pd.Series): The building row for which the heat flows are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data for the building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The  infiltration heat flows in Wh, for each hour in the time series.
    """
    # Load building data from row
    building_volume = building["volume_m3"]  # The volume of the building in m3
    infiltration_ACH = building["infiltration_ACH"]  # The air changes per hour due to infiltration

    # Load temperature difference between in- and outdoor from time series
    delta_T_air = time_series["T_outdoor_minus_indoor_C"]

    # Load global parameters
    air_density = global_parameters["air_density"]  # The density of air in kg/m3
    air_heat_capacity = global_parameters["air_heat_capacity"]  # The heat capacity of air in J/kgK

    # Determine the air mass flow rate due to infiltration in kg/s
    infiltration_mass_flow_rate = air_density * infiltration_ACH * building_volume / 3600

    # Calculate the heat flow due to infiltration in Wh
    return infiltration_mass_flow_rate * air_heat_capacity * delta_T_air


def calc_Q_ventilation(
    building: pd.Series,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
    calc_electricity_demand: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """calc_Q_ventilation calculates the ventilation heat flows and the electricity demand of ventilation systems of a building in Wh across a given time series.

    Args:
        building (pd.Series): The building row for which the heat flows are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data and presence load factors for the building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        calc_electricity_demand (bool, optional): Whether to calculate the electricity demand of the ventilation system. Defaults to False, as we currently don't take electricity from ventilation systems into account in the cooling demand model.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: The hourly ventilation heat flow and electricity demand time series in Wh, and the total annual average electricity demand of the ventilation system in kWh.
    """
    # Load building data from row
    building_end_use = building["end_use"]  # The building end use (residential, office, etc.)
    population = building["population"]  # The population of the building
    ventilation_rate_pp = building["ventilation_rate_pp_m3_h"]  # The ventilation rate per person in m3/h
    pressure_drop = building["pressure_drop_Pa"]  # The pressure drop of the ventilation system in Pa

    # Load temperature difference between in- and outdoor and people-presence load factors from time series
    delta_T_air = time_series["T_outdoor_minus_indoor_C"]
    people_presence = time_series[f"presence_people_{building_end_use}"]

    # Load global parameters
    air_density = global_parameters["air_density"]  # The density of air in kg/m3
    air_heat_capacity = global_parameters["air_heat_capacity"]  # The heat capacity of air in J/kgK
    ventilation_efficiency = global_parameters[
        "ventilation_efficiency"
    ]  # The assumed average electrical efficiency of ventilation systems

    # Determine the air mass flow rate due to ventilation in kg/s
    ventilation_rate = air_density * ventilation_rate_pp * population * people_presence / 3600

    # Determine the heat flow due to ventilation in Wh
    Q_ventilation = ventilation_rate * air_heat_capacity * delta_T_air

    if calc_electricity_demand:
        # Determine the electricity demand of the ventilation system in Wh
        Q_ventilation_electric = pressure_drop * ventilation_rate / ventilation_efficiency

        # Determine the amount of years in the time series
        years = len(Q_ventilation) / HOURS_PER_YEAR

        # Determine the total electricity demand per year in kWh
        E_ventilation_electric_kWh = np.sum(Q_ventilation_electric) / years / 1000
    else:
        Q_ventilation_electric = None
        E_ventilation_electric_kWh = None

    return Q_ventilation, Q_ventilation_electric, E_ventilation_electric_kWh


def calc_Q_solar_radiation(building: pd.Series, time_series: dict[str, np.ndarray]) -> np.ndarray:
    """calc_Q_solar_radiation calculates the heat gains due to solar radiation of a building in Wh across a given time series.

    Args:
        building (pd.Series): The building row for which the heat gains are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data for the building.

    Returns:
        np.ndarray: The total solar radiation heat inflows in Wh, for each hour in the time series.
    """
    # Load building data from row
    window_area_per_orientation = building[
        "window_area_per_orientation_m2"
    ]  # The window area per compass direction (N, NE, ...) of the building in m2
    g_window = building["g_window"]  # The solar transmittance factor of the windows (ranging from 0 to 1)

    # Solar radiation stacked per direction (N, NE, ..., NW), aligned to the window-area orientations.
    # Identical across buildings, so it is cached once per run by calc_cooling_demand_metrics_for_df;
    # fall back to building it here when called standalone.
    solar_radiation_per_direction = time_series.get("_solar_stack_W_m2")
    if solar_radiation_per_direction is None:
        solar_radiation_per_direction = np.array(
            [time_series[f"P_sol_{direction}_W_m2"] for direction in SOLAR_DIRECTIONS],
        )

    # Sum the solar heat inflow over the eight window orientations in Wh
    return g_window * (window_area_per_orientation[:, np.newaxis] * solar_radiation_per_direction).sum(axis=0)


def calc_Q_internal_heat(
    building: pd.Series,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_internal_heat calculates the internal heat gains in Wh across a given time series.

    Args:
        building (pd.Series): The building row for which the heat gains are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the presence load factors data for the building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The total internal heat gains in Wh, for each hour in the time series.
    """
    # Load building data from row
    population = building["population"]  # The population of the building
    building_end_use = building["end_use"]  # The building end use (residential, office, etc.)
    floor_area_total = building["floor_area_total_m2"]  # The total floor area of the building in m2
    int_heat_gain_appliances = building[
        "int_heat_gain_appliances_W_m2"
    ]  # The internal heat gain density of appliances in W/m2

    # Load presence load factors from time series, depending on building end use
    presence_people = time_series[f"presence_people_{building_end_use}"]
    presence_lighting = time_series[f"presence_lighting_{building_end_use}"]
    presence_appliances = time_series[f"presence_appliances_{building_end_use}"]

    # Load global parameters
    int_heat_gain_pp_W = global_parameters["int_heat_gain_pp_W"]  # The internal heat gain per person in W
    int_heat_gain_light_W_m2 = global_parameters[
        "int_heat_gain_light_W_m2"
    ]  # The internal heat gain density of lighting in W/m2

    # Calculate the internal heat gains from people, lighting and appliances in Wh
    Q_internal_heat_people = population * int_heat_gain_pp_W * presence_people
    Q_internal_heat_lighting = int_heat_gain_light_W_m2 * floor_area_total * presence_lighting
    Q_internal_heat_appliances = int_heat_gain_appliances * floor_area_total * presence_appliances

    # Calculate the total internal heat gains in Wh
    return Q_internal_heat_people + Q_internal_heat_lighting + Q_internal_heat_appliances


def calc_cooling_demand_from_thermal_flows(
    Q_transmission: np.ndarray,
    Q_infiltration: np.ndarray,
    Q_ventilation: np.ndarray,
    Q_solar_radiation: np.ndarray,
    Q_internal_heat: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """calc_cooling_demand calculates a number of metrics related to the cooling demand.

    Args:
        Q_transmission (np.ndarray): The time series of the transmission heat flows in Wh.
        Q_infiltration (np.ndarray): The time series of the infiltration heat flows in Wh.
        Q_ventilation (np.ndarray): The time series of the ventilation heat flows in Wh.
        Q_solar_radiation (np.ndarray): The time series of the solar radiation heat flows in Wh.
        Q_internal_heat (np.ndarray): The time series of the internal heat gains in Wh.

    Returns:
        Q_cooling (np.ndarray): The cooling demand per hour Q_cooling (Wh) across the time series for which heat flow data is available.
        E_cooling_avg_kWh (float): The average total cooling energy demand (kWh) per year.
        P_cooling_peak_avg_kW (float): The average peak cooling power demand (kW) per year.
    """
    # Calculate the net heat in- or outflow in Wh for each hour in the time series
    Q_net = Q_transmission + Q_infiltration + Q_ventilation + Q_solar_radiation + Q_internal_heat

    # Only if Q_net is positive, there is a cooling demand equal in size to Q_net. Otherwise, there is no cooling demand.
    Q_cooling_demand_Wh = np.maximum(Q_net, 0)

    # Calculate the average total cooling energy demand in kWh per year
    E_cooling_avg_kWh = np.sum(Q_cooling_demand_Wh) / 1000 / (len(Q_cooling_demand_Wh) / HOURS_PER_YEAR)

    # Determine the peak cooling power demand in kW for each year
    P_cooling_peaks_kW = Q_cooling_demand_Wh.reshape(-1, HOURS_PER_YEAR).max(axis=1) / 1000

    # Determine the average peak cooling power demand in kW per year
    P_cooling_peak_avg_kW = np.mean(P_cooling_peaks_kW)

    return Q_cooling_demand_Wh, E_cooling_avg_kWh, P_cooling_peak_avg_kW


# Cooling demand profile calculation functions


def calc_cooling_demand_percentile_per_year(
    Q_cooling_demand_Wh: np.ndarray,
    n_percentile: float = 98,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """calc_cooling_demand_percentile_per_year calculates cooling demand metrics for a given peak percentile n of the cooling demand distribution for a one-year period.

    Args:
        Q_cooling_demand_Wh (np.ndarray): The hourly cooling demand series (Wh) for a one-year period.
        n_percentile (float, optional): The percentile that the peak cooling power demand (kW) should be capped at. Defaults to 98.

    Returns:
        Q_cooling_sorted_Wh (np.ndarray): The hourly cooling demand series (Wh), sorted by size.
        P_cooling_peak_percentile_kW (float): The percentile of the peak cooling power demand (kW), i.e. the peak percentile of the cooling power demand.
        Q_cooling_capped_at_percentile_Wh (np.ndarray): The hourly cooling energy demand series Q_cooling (Wh), capped at the peak percentile of the cooling power demand.
        E_cooling_capped_at_percentile_kWh (float): The total cooling energy demand (kWh), when capped at the peak percentile of the cooling power demand.
    """
    # Sort the cooling demand series by size (starting with the largest values)
    Q_cooling_sorted_Wh = np.sort(Q_cooling_demand_Wh)[::-1]

    # Determine the percentile of the peak cooling power demand (kW), i.e. the peak percentile of the cooling power demand
    P_cooling_peak_percentile_kW = np.percentile(Q_cooling_sorted_Wh, n_percentile) / 1000

    # Cap the cooling demand series at the peak percentile of the cooling power demand
    Q_cooling_capped_at_percentile_Wh = np.minimum(Q_cooling_demand_Wh, P_cooling_peak_percentile_kW * 1000)

    # Calculate the total cooling energy demand (kWh), when capped at the peak percentile of the cooling power demand
    E_cooling_capped_at_percentile_kWh = np.sum(Q_cooling_capped_at_percentile_Wh) / 1000

    return (
        Q_cooling_sorted_Wh,
        P_cooling_peak_percentile_kW,
        Q_cooling_capped_at_percentile_Wh,
        E_cooling_capped_at_percentile_kWh,
    )


def calc_cooling_demand_percentile(
    Q_cooling_demand_Wh: np.ndarray,
    n_percentile: float = 98,
    include_time_series: bool = False,
) -> tuple[np.ndarray | None, float, np.ndarray | None, float]:
    """calc_cooling_demand_percentile calculates cooling demand metrics for a given peak percentile of the cooling demand distribution for a full time series.

    Args:
        Q_cooling_demand_Wh (np.ndarray): The hourly cooling demand time series (Wh), for any number of years.
        n_percentile (float, optional): The percentile that the peak cooling power demand (kW) should be capped at. Defaults to 98.
        include_time_series: (bool, optional): Whether to include the time series of the cooling demand metrics in the return values. Defaults to False.

    Returns:
        Q_cooling_sorted_Wh (np.ndarray): The hourly cooling demand series (Wh), sorted by size.
        P_cooling_peak_percentile_kW (float): The percentile of the peak cooling power demand (kW), i.e. the peak percentile of the cooling power demand.
        Q_cooling_capped_at_percentile_Wh (np.ndarray): The hourly cooling energy demand series Q_cooling (Wh), capped at the peak percentile of the cooling power demand.
        E_cooling_capped_at_percentile_kWh (float): The total cooling energy demand (kWh), when capped at the peak percentile of the cooling power demand.

    """
    # Split the hourly demand into full years (rows of HOURS_PER_YEAR hours)
    years = len(Q_cooling_demand_Wh) // HOURS_PER_YEAR
    Q_cooling_demand_years = Q_cooling_demand_Wh.reshape(years, HOURS_PER_YEAR)

    # Peak percentile of the cooling power demand per year (Wh), then averaged across years (kW)
    P_cooling_peak_percentile_per_year_Wh = np.percentile(Q_cooling_demand_years, n_percentile, axis=1)
    P_cooling_peak_percentile_kW = np.mean(P_cooling_peak_percentile_per_year_Wh) / 1000

    # Cap each year at its own peak percentile, then average the annual capped energy totals (kWh)
    Q_cooling_capped_years = np.minimum(Q_cooling_demand_years, P_cooling_peak_percentile_per_year_Wh[:, np.newaxis])
    E_cooling_capped_at_percentile_kWh = np.mean(Q_cooling_capped_years.sum(axis=1)) / 1000

    # The full hourly series are only sorted/materialized when explicitly requested (memory + speed)
    if include_time_series:
        Q_cooling_sorted_Wh = np.concatenate([np.sort(year)[::-1] for year in Q_cooling_demand_years])
        Q_cooling_capped_at_percentile_Wh = Q_cooling_capped_years.reshape(-1)
    else:
        Q_cooling_sorted_Wh = None
        Q_cooling_capped_at_percentile_Wh = None

    return (
        Q_cooling_sorted_Wh,
        P_cooling_peak_percentile_kW,
        Q_cooling_capped_at_percentile_Wh,
        E_cooling_capped_at_percentile_kWh,
    )


# Aggregate functions for cooling demand calculation


def calc_cooling_demand_for_building_row(
    building: pd.Series,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
    include_heat_flows: bool = False,
) -> tuple[np.ndarray, float, float, dict[str, np.ndarray] | None]:
    """calculate_cooling_demand_from_building_row calculates the cooling demand time series, total cooling energy demand and peak cooling power demand for a given building row.

    Args:
        building (pd.Series): The building row for which the cooling demand is calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data and presence load factors for the building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        include_heat_flows: (bool, optional): Whether to return the time series of the individual heat flows. Defaults to False.

    Returns:
        Q_cooling_demand_Wh (np.ndarray): The cooling demand per hour Q_cooling (Wh) across the time series for which heat flow data is available.
        E_cooling_avg_kWh (float): The average total cooling energy demand (kWh) per year.
        P_cooling_peak_avg_kW (float): The average peak cooling power demand (kW) per year.
    """
    # Determine the thermal flows in Wh
    Q_transmission_Wh = calc_Q_transmission(building, time_series, global_parameters)
    Q_infiltration_Wh = calc_Q_infiltration(building, time_series, global_parameters)
    Q_ventilation_Wh, _, _ = calc_Q_ventilation(
        building,
        time_series,
        global_parameters,
    )  # We currently don't calculate the electricity demand of the ventilation system
    Q_solar_radiation_Wh = calc_Q_solar_radiation(building, time_series)
    Q_internal_heat_Wh = calc_Q_internal_heat(building, time_series, global_parameters)

    # Determine the cooling demand metrics from the thermal flows
    (
        Q_cooling_demand_Wh,
        E_cooling_kWh,
        P_cooling_peak_kW,
    ) = calc_cooling_demand_from_thermal_flows(
        Q_transmission_Wh,
        Q_infiltration_Wh,
        Q_ventilation_Wh,
        Q_solar_radiation_Wh,
        Q_internal_heat_Wh,
    )
    if include_heat_flows:  # Pack the heat flow time series into a dictionary
        heat_flows = {
            "Q_transmission_Wh": Q_transmission_Wh,
            "Q_infiltration_Wh": Q_infiltration_Wh,
            "Q_ventilation_Wh": Q_ventilation_Wh,
            "Q_solar_radiation_Wh": Q_solar_radiation_Wh,
            "Q_internal_heat_Wh": Q_internal_heat_Wh,
        }
    else:
        heat_flows = None

    return Q_cooling_demand_Wh, E_cooling_kWh, P_cooling_peak_kW, heat_flows


def calc_cooling_demand_metrics_for_building_row(
    building: pd.Series,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
    include_time_series: bool = False,
) -> tuple:
    """calculate_cooling_demand_from_building_row calculates the cooling demand time series, total cooling energy demand and peak cooling power demand for a given building row.

    Args:
        building (pd.Series): The building row for which the cooling demand metrics area calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data and presence load factors for the building.
        global_parameters (Dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        include_time_series: (bool, optional): Whether to include the time series of the cooling demand metrics in the return values. Defaults to False.

    Returns:
        Q_cooling_demand_Wh (np.ndarray): The cooling demand per hour Q_cooling (Wh) across the time series for which heat flow data is available.
        E_cooling_avg_kWh (float): The average total cooling energy demand (kWh) per year.
        P_cooling_peak_avg_kW (float): The average peak cooling power demand (kW) per year.
    """
    # Load global parameters
    peak_cooling_percentile_cap = global_parameters[
        "peak_cooling_percentile_cap"
    ]  # The percentile that the peak cooling power demand (kW) should be capped at

    # Calculate the cooling demand for the building row
    (
        Q_cooling_demand_Wh,
        E_cooling_kWh,
        P_cooling_peak_kW,
        heat_flows,
    ) = calc_cooling_demand_for_building_row(
        building,
        time_series,
        global_parameters,
        include_time_series,
    )

    # Determine the cooling demand metrics for the peak percentile of the cooling power demand
    (
        Q_cooling_sorted_Wh,
        P_cooling_peak_percentile_kW,
        Q_cooling_capped_at_percentile_Wh,
        E_cooling_capped_at_percentile_kWh,
    ) = calc_cooling_demand_percentile(Q_cooling_demand_Wh, peak_cooling_percentile_cap, include_time_series)

    # Determine the E_cooling_capped_at_percentile_kWh_m2 and P_cooling_peak_percentile_kW_m2 per floor area
    E_cooling_capped_at_percentile_Wh_m2 = E_cooling_capped_at_percentile_kWh * 1000 / building["floor_area_total_m2"]
    P_cooling_peak_percentile_W_m2 = P_cooling_peak_percentile_kW * 1000 / building["floor_area_total_m2"]

    # # If include_time_series is false, set the time series to None for memory efficiency
    if not include_time_series:
        Q_cooling_demand_Wh = None
        Q_cooling_sorted_Wh = None
        Q_cooling_capped_at_percentile_Wh = None
        heat_flows = None

    return (
        E_cooling_kWh,
        E_cooling_capped_at_percentile_kWh,
        E_cooling_capped_at_percentile_Wh_m2,
        P_cooling_peak_kW,
        P_cooling_peak_percentile_kW,
        P_cooling_peak_percentile_W_m2,
        Q_cooling_demand_Wh,
        Q_cooling_sorted_Wh,
        Q_cooling_capped_at_percentile_Wh,
        heat_flows,
    )


def calc_cooling_demand_metrics_for_df(
    df_buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
    include_time_series: bool = False,
) -> pd.DataFrame:
    """calc_cooling_demand_metrics_for_df calculates the cooling demand metrics for each building row in a given DataFrame.

    Args:
        df_buildings (pd.DataFrame): The DataFrame containing the building rows for which the cooling demand metrics are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data and presence load factors for the building.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        include_time_series: (bool, optional): Whether to include the time series of the cooling demand metrics in the return values. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame containing the cooling demand metrics for each building row.
    """
    # Set the percentile that the peak cooling power demand (kW) should be capped at, for column naming purposes
    peak_cooling_percentile_cap = int(global_parameters["peak_cooling_percentile_cap"])

    # Cache the per-direction solar radiation stack once (N, NE, ..., NW; identical across buildings) so each row reuses it
    time_series["_solar_stack_W_m2"] = np.array(
        [time_series[f"P_sol_{direction}_W_m2"] for direction in SOLAR_DIRECTIONS],
    )

    apply_args = (time_series, global_parameters, include_time_series)

    def apply_calc_cooling_demand_metrics(row: pd.Series) -> tuple:
        return calc_cooling_demand_metrics_for_building_row(row, *apply_args)

    result_df = df_buildings.apply(apply_calc_cooling_demand_metrics, axis=1, result_type="expand")

    result_columns = [
        "E_cooling_kWh",
        f"E_cooling_capped_at_{peak_cooling_percentile_cap}th_percentile_kWh",
        f"E_cooling_capped_at_{peak_cooling_percentile_cap}th_percentile_Wh_m2",
        "P_cooling_peak_kW",
        f"P_cooling_peak_{peak_cooling_percentile_cap}th_percentile_kW",
        f"P_cooling_peak_{peak_cooling_percentile_cap}th_percentile_W_m2",
    ]

    if include_time_series:  # Add the time series columns to the result columns
        result_columns += [
            "Q_cooling_demand_Wh",
            "Q_cooling_sorted_Wh",
            f"Q_cooling_capped_at_{peak_cooling_percentile_cap}th_percentile_Wh",
        ]

        # Unpack the heat flows dictionary into separate columns
        for heat_flow in result_df.iloc[0][9]:  # The heat flows are stored in the 10th column of the result_df
            result_df[heat_flow] = result_df.apply(lambda row: row[9][heat_flow], axis=1)  # noqa: B023 (apply runs eagerly within the loop iteration)

        # Drop the heat flows dictionary column
        result_df = result_df.drop(result_df.columns[9], axis=1)

        # Add the heat flow columns to the result column names
        result_columns += list(result_df.columns[9:])

    df_buildings[result_columns] = result_df.dropna(axis=1, how="all")

    return df_buildings


# Environmental impact calculation functions
