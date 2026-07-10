"""Thermodynamic calculation functions used in the cooling demand model.

The heat-flow functions are vectorised over buildings: each takes a DataFrame of
buildings and returns a ``(n_buildings, n_hours)`` array of hourly heat flows in Wh.
Buildings are processed in chunks so those intermediates stay bounded in memory.

@author: Simon van Lierde
"""

from __future__ import annotations

# Import packages
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cdm.constants import HOURS_PER_YEAR, SOLAR_DIRECTIONS

if TYPE_CHECKING:
    from collections.abc import Iterator

# Buildings are processed in chunks of at most this many (building, hour) cells, so peak memory
# stays flat as the building stock grows. Each (n_buildings, n_hours) float32 intermediate costs
# 4 bytes per cell and roughly ten of them are live at once, so 4M cells is a few hundred MB.
# NOTE: a fixed budget, not a measured one; tune it if a machine's memory profile differs.
MAX_CHUNK_CELLS = 4_000_000

# The hourly (building, hour) blocks are held in float32: they dominate memory, and a Wh-scale
# quantity keeps ~7 significant digits, far more than the model's inputs justify. The dtype must
# be introduced where the arrays are *born* -- casting the finished flows costs more than the
# arithmetic it saves. Reductions along the hour axis still accumulate in float64 (see ACCUM_DTYPE),
# so summing 43,800 hours does not lose precision.
FLOW_DTYPE = np.float32
ACCUM_DTYPE = np.float64

# Parameter derivation functions


def R_to_U(Rc: float | np.ndarray, alfa_i: float = 7.5, alfa_o: float = 27.5) -> float | np.ndarray:
    """R_to_U converts the thermal resistance of a building element to the transmittance of that element (U).

    Args:
        Rc (float | np.ndarray): Thermal resistance of the building element(s) in m2K/W
        alfa_i (float, optional): Combined heat transfer coefficient of convection and radiation on the inside in W/m2K. Defaults to 7.5.
        alfa_o (float, optional): Combined heat transfer coefficient of convection and radiation on the outside in W/m2K. Defaults to 27.5.

    Returns:
        float | np.ndarray: Transmittance of the building element(s) in W/m2K.
    """
    return 1 / (1 / alfa_i + Rc + 1 / alfa_o)  # Transmittance of the building element in W/m2K


# Thermal flow calculation functions


def calc_Q_transmission(
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_transmission calculates the transmission heat flow Q of each building in Wh across a given time series.

    Args:
        buildings (pd.DataFrame): The buildings for which the transmission heat flows are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data for the buildings.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The total transmission heat flow in Wh, of shape (n_buildings, n_hours).
    """
    # Load building data
    window_area = buildings["window_area_total_m2"].to_numpy(dtype=FLOW_DTYPE)  # The window area of the buildings in m2
    wall_area = buildings["wall_area_total_m2"].to_numpy(dtype=FLOW_DTYPE)  # The wall area of the buildings in m2
    floor_area = buildings["floor_area_ground_m2"].to_numpy(dtype=FLOW_DTYPE)  # The floor area of the buildings in m2
    Rc_wall = buildings["Rc_wall_m2K_W"].to_numpy(dtype=FLOW_DTYPE)  # The thermal resistance of the walls in m2K/W
    Rc_roof = buildings["Rc_roof_m2K_W"].to_numpy(dtype=FLOW_DTYPE)  # The thermal resistance of the roof in m2K/W
    Rc_floor = buildings["Rc_floor_m2K_W"].to_numpy(dtype=FLOW_DTYPE)  # The thermal resistance of the floor in m2K/W
    U_window = buildings["U_window_W_m2K"].to_numpy(
        dtype=FLOW_DTYPE,
    )  # The thermal transmittance of the windows in W/m2K

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

    # The window, wall and roof heat flows all scale with the air temperature difference, so their
    # transmittance-area products (W/K) can be summed per building before the outer product with the
    # hourly temperature difference. The floor flow is constant, set by the subsurface gradient.
    UA_air_W_K = U_window * window_area + U_wall * wall_area + U_roof * roof_area
    Q_transmission_floor = U_floor * floor_area * (T_sub_C - T_thresh_C)

    # Calculate the total transmission heat flow in Wh
    return np.outer(UA_air_W_K, delta_T_air) + Q_transmission_floor[:, np.newaxis]


def calc_Q_infiltration(
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_infiltration calculates the infiltration heat flows of each building in Wh across a given time series.

    Args:
        buildings (pd.DataFrame): The buildings for which the heat flows are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data for the buildings.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The infiltration heat flows in Wh, of shape (n_buildings, n_hours).
    """
    # Load building data
    building_volume = buildings["volume_m3"].to_numpy(dtype=FLOW_DTYPE)  # The volume of the buildings in m3
    infiltration_ACH = buildings["infiltration_ACH"].to_numpy(
        dtype=FLOW_DTYPE,
    )  # The air changes per hour by infiltration

    # Load temperature difference between in- and outdoor from time series
    delta_T_air = time_series["T_outdoor_minus_indoor_C"]

    # Load global parameters
    air_density = global_parameters["air_density"]  # The density of air in kg/m3
    air_heat_capacity = global_parameters["air_heat_capacity"]  # The heat capacity of air in J/kgK

    # Determine the air mass flow rate due to infiltration in kg/s
    infiltration_mass_flow_rate = air_density * infiltration_ACH * building_volume / 3600

    # Calculate the heat flow due to infiltration in Wh
    return np.outer(infiltration_mass_flow_rate * air_heat_capacity, delta_T_air)


def calc_Q_ventilation(
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_ventilation calculates the ventilation heat flows of each building in Wh across a given time series.

    Args:
        buildings (pd.DataFrame): The buildings for which the heat flows are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data and presence load factors for the buildings.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The ventilation heat flows in Wh, of shape (n_buildings, n_hours).
    """
    # Load building data
    end_use = buildings["end_use"].to_numpy()  # The building end use (residential, office, etc.)
    population = buildings["population"].to_numpy(dtype=FLOW_DTYPE)  # The population of the buildings
    ventilation_rate_pp = buildings["ventilation_rate_pp_m3_h"].to_numpy(
        dtype=FLOW_DTYPE,
    )  # Ventilation rate pp in m3/h

    # Load temperature difference between in- and outdoor from time series
    delta_T_air = time_series["T_outdoor_minus_indoor_C"]

    # Load global parameters
    air_density = global_parameters["air_density"]  # The density of air in kg/m3
    air_heat_capacity = global_parameters["air_heat_capacity"]  # The heat capacity of air in J/kgK

    # The heat flow per building is a constant (its air mass flow rate at full occupancy, times the heat
    # capacity of air) multiplied by an hourly profile that depends only on the building's end use.
    ventilation_coefficient = air_density * ventilation_rate_pp * population * air_heat_capacity / 3600

    Q_ventilation = np.empty((len(buildings), len(delta_T_air)), dtype=FLOW_DTYPE)
    for use in np.unique(end_use):
        buildings_with_use = end_use == use
        hourly_profile = time_series[f"presence_people_{use}"] * delta_T_air
        Q_ventilation[buildings_with_use] = np.outer(ventilation_coefficient[buildings_with_use], hourly_profile)

    return Q_ventilation


def calc_Q_solar_radiation(buildings: pd.DataFrame, time_series: dict[str, np.ndarray]) -> np.ndarray:
    """calc_Q_solar_radiation calculates the heat gains due to solar radiation of each building in Wh across a given time series.

    Args:
        buildings (pd.DataFrame): The buildings for which the heat gains are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data for the buildings.

    Returns:
        np.ndarray: The total solar radiation heat inflows in Wh, of shape (n_buildings, n_hours).
    """
    # Load building data: the window area per compass direction (N, NE, ...) in m2 and the solar
    # transmittance factor of the windows (ranging from 0 to 1)
    window_area_per_orientation = np.stack(buildings["window_area_per_orientation_m2"].to_numpy()).astype(FLOW_DTYPE)
    g_window = buildings["g_window"].to_numpy(dtype=FLOW_DTYPE)

    # Solar radiation stacked per direction (N, NE, ..., NW), aligned to the window-area orientations
    solar_radiation_per_direction = np.array(
        [time_series[f"P_sol_{d}_W_m2"] for d in SOLAR_DIRECTIONS],
        dtype=FLOW_DTYPE,
    )

    # Sum the solar heat inflow over the eight window orientations in Wh
    return (g_window[:, np.newaxis] * window_area_per_orientation) @ solar_radiation_per_direction


def calc_Q_internal_heat(
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
) -> np.ndarray:
    """calc_Q_internal_heat calculates the internal heat gains of each building in Wh across a given time series.

    Args:
        buildings (pd.DataFrame): The buildings for which the heat gains are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the presence load factors data for the buildings.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.

    Returns:
        np.ndarray: The total internal heat gains in Wh, of shape (n_buildings, n_hours).
    """
    # Load building data
    end_use = buildings["end_use"].to_numpy()  # The building end use (residential, office, etc.)
    population = buildings["population"].to_numpy(dtype=FLOW_DTYPE)  # The population of the buildings
    floor_area_total = buildings["floor_area_total_m2"].to_numpy(dtype=FLOW_DTYPE)  # The total floor area in m2
    int_heat_gain_appliances = buildings["int_heat_gain_appliances_W_m2"].to_numpy(
        dtype=FLOW_DTYPE,
    )  # Appliances in W/m2

    # Load global parameters
    int_heat_gain_pp_W = global_parameters["int_heat_gain_pp_W"]  # The internal heat gain per person in W
    int_heat_gain_light_W_m2 = global_parameters[
        "int_heat_gain_light_W_m2"
    ]  # The internal heat gain density of lighting in W/m2

    # The heat gains from people, lighting and appliances are each a per-building constant (W) times an
    # hourly presence profile that depends only on the building's end use, so per end use the whole block
    # of buildings is one (n_buildings, 3) @ (3, n_hours) matrix product.
    heat_gains_W = np.column_stack(
        [
            population * int_heat_gain_pp_W,
            int_heat_gain_light_W_m2 * floor_area_total,
            int_heat_gain_appliances * floor_area_total,
        ],
    )

    Q_internal_heat = np.empty((len(buildings), len(time_series["T_outdoor_minus_indoor_C"])), dtype=FLOW_DTYPE)
    for use in np.unique(end_use):
        buildings_with_use = end_use == use
        presence_profiles = np.array(
            [time_series[f"presence_{load}_{use}"] for load in ("people", "lighting", "appliances")],
        )
        Q_internal_heat[buildings_with_use] = heat_gains_W[buildings_with_use] @ presence_profiles

    return Q_internal_heat


def calc_cooling_demand_from_thermal_flows(
    Q_transmission: np.ndarray,
    Q_infiltration: np.ndarray,
    Q_ventilation: np.ndarray,
    Q_solar_radiation: np.ndarray,
    Q_internal_heat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """calc_cooling_demand calculates a number of metrics related to the cooling demand.

    The heat flow arrays are hourly along their last axis; any leading axis (e.g. buildings) is preserved.

    Args:
        Q_transmission (np.ndarray): The time series of the transmission heat flows in Wh.
        Q_infiltration (np.ndarray): The time series of the infiltration heat flows in Wh.
        Q_ventilation (np.ndarray): The time series of the ventilation heat flows in Wh.
        Q_solar_radiation (np.ndarray): The time series of the solar radiation heat flows in Wh.
        Q_internal_heat (np.ndarray): The time series of the internal heat gains in Wh.

    Returns:
        Q_cooling (np.ndarray): The cooling demand per hour Q_cooling (Wh) across the time series for which heat flow data is available.
        E_cooling_avg_kWh (np.ndarray): The average total cooling energy demand (kWh) per year.
        P_cooling_peak_avg_kW (np.ndarray): The average peak cooling power demand (kW) per year.
    """
    # Calculate the net heat in- or outflow in Wh for each hour in the time series.
    # Accumulated in place: `a + b + c + d + e` would materialise a fresh (buildings, hours)
    # block per `+`. Left-to-right order is preserved, so the result is bitwise identical.
    Q_net = np.add(Q_transmission, Q_infiltration)
    np.add(Q_net, Q_ventilation, out=Q_net)
    np.add(Q_net, Q_solar_radiation, out=Q_net)
    np.add(Q_net, Q_internal_heat, out=Q_net)

    # Only if Q_net is positive, there is a cooling demand equal in size to Q_net. Otherwise, there is no cooling demand.
    Q_cooling_demand_Wh = np.maximum(Q_net, 0, out=Q_net)  # Q_net is a fresh array, so clipping it in place is safe

    # Split the hourly demand into full years (rows of HOURS_PER_YEAR hours)
    years = Q_cooling_demand_Wh.shape[-1] // HOURS_PER_YEAR
    Q_cooling_demand_years = Q_cooling_demand_Wh.reshape(*Q_cooling_demand_Wh.shape[:-1], years, HOURS_PER_YEAR)

    # Calculate the average total cooling energy demand in kWh per year. The hour-axis sum spans
    # 43,800 float32 terms, so it accumulates in float64 to keep the annual totals exact.
    E_cooling_avg_kWh = Q_cooling_demand_Wh.sum(axis=-1, dtype=ACCUM_DTYPE) / 1000 / years

    # Determine the peak cooling power demand in kW for each year, then average it across the years
    P_cooling_peak_avg_kW = Q_cooling_demand_years.max(axis=-1).mean(axis=-1, dtype=ACCUM_DTYPE) / 1000

    return Q_cooling_demand_Wh, E_cooling_avg_kWh, P_cooling_peak_avg_kW


# Cooling demand profile calculation functions


def calc_cooling_demand_percentile(
    Q_cooling_demand_Wh: np.ndarray,
    n_percentile: float = 98,
    include_time_series: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray]:
    """calc_cooling_demand_percentile calculates cooling demand metrics for a given peak percentile of the cooling demand distribution for a full time series.

    The demand array is hourly along its last axis; any leading axis (e.g. buildings) is preserved.

    Args:
        Q_cooling_demand_Wh (np.ndarray): The hourly cooling demand time series (Wh), for any number of years.
        n_percentile (float, optional): The percentile that the peak cooling power demand (kW) should be capped at. Defaults to 98.
        include_time_series: (bool, optional): Whether to include the time series of the cooling demand metrics in the return values. Defaults to False.

    Returns:
        Q_cooling_sorted_Wh (np.ndarray | None): The hourly cooling demand series (Wh), sorted by size within each year.
        P_cooling_peak_percentile_kW (np.ndarray): The percentile of the peak cooling power demand (kW), i.e. the peak percentile of the cooling power demand.
        Q_cooling_capped_at_percentile_Wh (np.ndarray | None): The hourly cooling energy demand series Q_cooling (Wh), capped at the peak percentile of the cooling power demand.
        E_cooling_capped_at_percentile_kWh (np.ndarray): The total cooling energy demand (kWh), when capped at the peak percentile of the cooling power demand.
    """
    # Split the hourly demand into full years (rows of HOURS_PER_YEAR hours)
    years = Q_cooling_demand_Wh.shape[-1] // HOURS_PER_YEAR
    Q_cooling_demand_years = Q_cooling_demand_Wh.reshape(*Q_cooling_demand_Wh.shape[:-1], years, HOURS_PER_YEAR)

    # Peak percentile of the cooling power demand per year (Wh), then averaged across years (kW).
    # When the hourly series are not returned, np.percentile is allowed to partition the block in place
    # instead of copying it (one full (n_buildings, n_hours) array per chunk). That permutes the hours
    # within each year, which the only later reader -- the capped sum along that same axis -- ignores.
    P_cooling_peak_percentile_per_year_Wh = np.percentile(
        Q_cooling_demand_years,
        n_percentile,
        axis=-1,
        overwrite_input=not include_time_series,
    )
    P_cooling_peak_percentile_kW = P_cooling_peak_percentile_per_year_Wh.mean(axis=-1, dtype=ACCUM_DTYPE) / 1000

    # Cap each year at its own peak percentile, then average the annual capped energy totals (kWh).
    # As above, the 8760-term hour-axis sum accumulates in float64.
    Q_cooling_capped_years = np.minimum(Q_cooling_demand_years, P_cooling_peak_percentile_per_year_Wh[..., np.newaxis])
    E_cooling_capped_at_percentile_kWh = Q_cooling_capped_years.sum(axis=-1, dtype=ACCUM_DTYPE).mean(axis=-1) / 1000

    # The full hourly series are only sorted/materialized when explicitly requested (memory + speed)
    if include_time_series:
        Q_cooling_sorted_Wh = np.sort(Q_cooling_demand_years, axis=-1)[..., ::-1].reshape(Q_cooling_demand_Wh.shape)
        Q_cooling_capped_at_percentile_Wh = Q_cooling_capped_years.reshape(Q_cooling_demand_Wh.shape)
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


def calc_cooling_demand_metrics_for_chunk(
    buildings: pd.DataFrame,
    time_series: dict[str, np.ndarray],
    global_parameters: dict[str, float],
    include_time_series: bool = False,
) -> pd.DataFrame:
    """Calculates the cooling demand metrics for a chunk of buildings that fits in memory at once.

    Args:
        buildings (pd.DataFrame): The buildings for which the cooling demand metrics are calculated.
        time_series (dict[str, np.ndarray]): The time series dictionary containing the weather data and presence load factors for the buildings.
        global_parameters (dict[str, float]): The dictionary containing the global parameters for the cooling demand model.
        include_time_series: (bool, optional): Whether to include the hourly time series (as per-building arrays) in the result. Defaults to False.

    Returns:
        pd.DataFrame: The cooling demand metrics, indexed like ``buildings``.
    """
    # The percentile that the peak cooling power demand (kW) should be capped at, also used for column naming
    cap = int(global_parameters["peak_cooling_percentile_cap"])

    # Determine the thermal flows in Wh, each of shape (n_buildings, n_hours)
    heat_flows = {
        "Q_transmission_Wh": calc_Q_transmission(buildings, time_series, global_parameters),
        "Q_infiltration_Wh": calc_Q_infiltration(buildings, time_series, global_parameters),
        "Q_ventilation_Wh": calc_Q_ventilation(buildings, time_series, global_parameters),
        "Q_solar_radiation_Wh": calc_Q_solar_radiation(buildings, time_series),
        "Q_internal_heat_Wh": calc_Q_internal_heat(buildings, time_series, global_parameters),
    }

    # Determine the cooling demand metrics from the thermal flows
    Q_cooling_demand_Wh, E_cooling_kWh, P_cooling_peak_kW = calc_cooling_demand_from_thermal_flows(*heat_flows.values())

    # Determine the cooling demand metrics for the peak percentile of the cooling power demand
    (
        Q_cooling_sorted_Wh,
        P_cooling_peak_percentile_kW,
        Q_cooling_capped_at_percentile_Wh,
        E_cooling_capped_at_percentile_kWh,
    ) = calc_cooling_demand_percentile(Q_cooling_demand_Wh, cap, include_time_series)

    # Determine the capped cooling energy and peak power demand per floor area
    floor_area_total_m2 = buildings["floor_area_total_m2"].to_numpy(dtype=FLOW_DTYPE)

    metrics = pd.DataFrame(
        {
            "E_cooling_kWh": E_cooling_kWh,
            f"E_cooling_capped_at_{cap}th_percentile_kWh": E_cooling_capped_at_percentile_kWh,
            f"E_cooling_capped_at_{cap}th_percentile_Wh_m2": E_cooling_capped_at_percentile_kWh
            * 1000
            / floor_area_total_m2,
            "P_cooling_peak_kW": P_cooling_peak_kW,
            f"P_cooling_peak_{cap}th_percentile_kW": P_cooling_peak_percentile_kW,
            f"P_cooling_peak_{cap}th_percentile_W_m2": P_cooling_peak_percentile_kW * 1000 / floor_area_total_m2,
        },
        index=buildings.index,
    )

    if include_time_series:
        # Store the hourly series as one array per building, in object columns
        hourly_series = {
            "Q_cooling_demand_Wh": Q_cooling_demand_Wh,
            "Q_cooling_sorted_Wh": Q_cooling_sorted_Wh,
            f"Q_cooling_capped_at_{cap}th_percentile_Wh": Q_cooling_capped_at_percentile_Wh,
            **heat_flows,
        }
        for name, hourly_values in hourly_series.items():
            metrics[name] = pd.Series(list(hourly_values), index=buildings.index, dtype=object)

    return metrics


def _chunks(buildings: pd.DataFrame, n_hours: int) -> Iterator[pd.DataFrame]:
    """Splits the buildings into chunks whose (building, hour) intermediates fit within MAX_CHUNK_CELLS."""
    chunk_size = max(1, MAX_CHUNK_CELLS // n_hours)
    for start in range(0, len(buildings), chunk_size):
        yield buildings.iloc[start : start + chunk_size]


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
    n_hours = len(time_series["T_outdoor_minus_indoor_C"])
    if n_hours % HOURS_PER_YEAR != 0:
        # The demand series is reshaped into whole years; a partial year (a weather CSV that still
        # holds a Feb 29, or a truncated series) otherwise fails the reshape with a cryptic
        # "cannot reshape" deep in the chunk loop. Fail here, at the boundary, with the real counts.
        msg = (
            f"weather series has {n_hours} hours, not a whole number of {HOURS_PER_YEAR}-hour years "
            f"({n_hours / HOURS_PER_YEAR:.4f}); drop leap days and supply full years."
        )
        raise ValueError(msg)

    # Cast the hourly series once, not per chunk: a float64 series here would promote every
    # (n_buildings, n_hours) block it touches straight back to float64.
    time_series = {name: np.asarray(series, dtype=FLOW_DTYPE) for name, series in time_series.items()}

    metrics = pd.concat(
        [
            calc_cooling_demand_metrics_for_chunk(chunk, time_series, global_parameters, include_time_series)
            for chunk in _chunks(df_buildings, n_hours)
        ],
    )

    df_buildings[metrics.columns] = metrics

    return df_buildings


# Environmental impact calculation functions
