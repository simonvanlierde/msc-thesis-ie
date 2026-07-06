"""Unit tests for thermodynamic helper functions."""

import numpy as np
import pandas as pd
import pytest

from cdm.thermodynamic import (
    R_to_U,
    calc_cooling_demand_for_building_row,
    calc_cooling_demand_from_thermal_flows,
    calc_cooling_demand_percentile,
    calc_cooling_demand_percentile_per_year,
    calc_Q_infiltration,
    calc_Q_internal_heat,
    calc_Q_solar_radiation,
    calc_Q_transmission,
    calc_Q_ventilation,
)


def test_R_to_U_against_formula() -> None:
    # U = 1 / (1/alfa_i + Rc + 1/alfa_o) # noqa: ERA001 # Formula, not code
    rc, alfa_i, alfa_o = 2.0, 7.5, 27.5
    expected = 1 / (1 / alfa_i + rc + 1 / alfa_o)
    assert R_to_U(rc, alfa_i, alfa_o) == pytest.approx(expected)


def test_R_to_U_uses_default_surface_coefficients() -> None:
    assert R_to_U(2.0) == pytest.approx(1 / (1 / 7.5 + 2.0 + 1 / 27.5))


def test_R_to_U_decreases_with_higher_resistance() -> None:
    # Better insulation (higher Rc) must yield a lower transmittance.
    assert R_to_U(5.0) < R_to_U(1.0)


def test_R_to_U_zero_resistance_is_surface_only() -> None:
    alfa_i, alfa_o = 7.5, 27.5
    assert R_to_U(0.0, alfa_i, alfa_o) == pytest.approx(1 / (1 / alfa_i + 1 / alfa_o))


def test_calc_Q_infiltration_scales_linearly_with_temperature_difference() -> None:
    building = pd.Series({"volume_m3": 300.0, "infiltration_ACH": 0.5})
    global_parameters = {"air_density": 1.2, "air_heat_capacity": 1005.0}

    delta_t = np.array([0.0, 1.0, 2.0, 4.0])
    time_series = {"T_outdoor_minus_indoor_C": delta_t}

    q = calc_Q_infiltration(building, time_series, global_parameters)

    mass_flow = 1.2 * 0.5 * 300.0 / 3600  # kg/s
    expected = mass_flow * 1005.0 * delta_t
    assert np.allclose(q, expected)
    # Zero temperature difference must give zero heat flow.
    assert q[0] == pytest.approx(0.0)


def test_calc_Q_transmission_matches_component_sum(
    building: pd.Series,
    time_series: dict,
    global_parameters: dict,
) -> None:
    delta_t = time_series["T_outdoor_minus_indoor_C"]
    u_wall, u_roof, u_floor = (R_to_U(rc) for rc in (2.0, 3.0, 2.5))

    # Window, wall and roof scale with the air temperature difference;
    # the floor with the (constant) subsurface gradient.
    expected = (1.8 * 50.0 + u_wall * 150.0 + u_roof * 100.0) * delta_t + u_floor * 100.0 * (12.0 - 25.0)

    assert np.allclose(calc_Q_transmission(building, time_series, global_parameters), expected)


def test_calc_Q_solar_radiation_sums_over_orientations(building: pd.Series, time_series: dict) -> None:
    window_area = building["window_area_per_orientation_m2"]
    radiation = np.array([time_series[f"P_sol_{d}_W_m2"] for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]])
    expected = (window_area[:, None] * 0.6 * radiation).sum(axis=0)

    assert np.allclose(calc_Q_solar_radiation(building, time_series), expected)


def test_calc_Q_internal_heat_combines_people_lighting_appliances(
    building: pd.Series,
    time_series: dict,
    global_parameters: dict,
) -> None:
    expected = (
        30.0 * 100.0 * time_series["presence_people_office"]
        + 5.0 * 300.0 * time_series["presence_lighting_office"]
        + 8.0 * 300.0 * time_series["presence_appliances_office"]
    )

    assert np.allclose(calc_Q_internal_heat(building, time_series, global_parameters), expected)


def test_calc_Q_ventilation_without_electricity_returns_none(
    building: pd.Series,
    time_series: dict,
    global_parameters: dict,
) -> None:
    q_vent, q_elec, e_elec = calc_Q_ventilation(building, time_series, global_parameters)

    rate = 1.2 * 30.0 * 30.0 * time_series["presence_people_office"] / 3600
    assert np.allclose(q_vent, rate * 1005.0 * time_series["T_outdoor_minus_indoor_C"])
    # Electricity demand is opt-in and skipped by default.
    assert q_elec is None
    assert e_elec is None


def test_calc_Q_ventilation_with_electricity_is_computed(
    building: pd.Series,
    time_series: dict,
    global_parameters: dict,
) -> None:
    _, q_elec, e_elec = calc_Q_ventilation(building, time_series, global_parameters, calc_electricity_demand=True)

    assert q_elec is not None
    assert e_elec is not None


def test_calc_cooling_demand_only_counts_positive_net_heat() -> None:
    # One hot hour (net +1000 Wh) and one cold hour (net -500 Wh) in an otherwise neutral year.
    q_internal = np.zeros(8760)
    q_internal[0] = 1000.0
    q_transmission = np.zeros(8760)
    q_transmission[1] = -500.0
    zeros = np.zeros(8760)

    q_cooling, e_cooling_kwh, p_peak_kw = calc_cooling_demand_from_thermal_flows(
        q_transmission,
        zeros,
        zeros,
        zeros,
        q_internal,
    )

    # Cooling demand exists only where the net heat flow is positive.
    assert q_cooling[0] == pytest.approx(1000.0)
    assert q_cooling[1] == pytest.approx(0.0)
    # 1000 Wh over a single one-year series is 1 kWh, peaking at 1 kW.
    assert e_cooling_kwh == pytest.approx(1.0)
    assert p_peak_kw == pytest.approx(1.0)


def test_calc_cooling_demand_averages_peaks_across_years() -> None:
    # Two years with peaks of 1000 Wh and 3000 Wh -> average peak of 2 kW.
    year_one = np.zeros(8760)
    year_one[0] = 1000.0
    year_two = np.zeros(8760)
    year_two[0] = 3000.0
    q_internal = np.concatenate([year_one, year_two])
    zeros = np.zeros(8760 * 2)

    _, _, p_peak_kw = calc_cooling_demand_from_thermal_flows(zeros, zeros, zeros, zeros, q_internal)

    assert p_peak_kw == pytest.approx(2.0)


def test_calc_cooling_demand_percentile_caps_peaks() -> None:
    # A flat year with two extreme spikes; capping at the 98th percentile removes the spikes.
    q = np.full(8760, 100.0)
    q[:2] = 50_000.0

    _, p_peak_percentile_kw, _, e_capped_kwh = calc_cooling_demand_percentile_per_year(q, n_percentile=98)

    uncapped_kwh = q.sum() / 1000
    # The 98th-percentile cap is far below the spike, so the capped energy is lower than the raw total.
    assert e_capped_kwh < uncapped_kwh
    assert p_peak_percentile_kw == pytest.approx(np.percentile(np.sort(q)[::-1], 98) / 1000)


def test_calc_cooling_demand_percentile_averages_over_years() -> None:
    # Two identical years -> the multi-year averages match the single-year result.
    one_year = np.full(8760, 100.0)
    one_year[:2] = 50_000.0
    two_years = np.concatenate([one_year, one_year])

    _, p_peak_two_years, _, e_two_years = calc_cooling_demand_percentile(two_years, n_percentile=98)
    _, p_peak_one_year, _, e_one_year = calc_cooling_demand_percentile_per_year(one_year, n_percentile=98)

    assert p_peak_two_years == pytest.approx(p_peak_one_year)
    assert e_two_years == pytest.approx(e_one_year)


def test_calc_cooling_demand_for_building_row_runs_end_to_end(
    building: pd.Series,
    time_series_full_year: dict,
    global_parameters: dict,
) -> None:
    q_cooling, e_cooling_kwh, p_peak_kw, heat_flows = calc_cooling_demand_for_building_row(
        building,
        time_series_full_year,
        global_parameters,
    )

    # A full year of hourly demand, non-negative everywhere, with positive aggregate metrics.
    assert q_cooling.shape == (8760,)
    assert (q_cooling >= 0).all()
    assert e_cooling_kwh > 0
    assert p_peak_kw > 0
    # Heat flows are only returned when explicitly requested.
    assert heat_flows is None
