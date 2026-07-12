"""Unit tests for thermodynamic helper functions.

The heat-flow functions are vectorised over buildings, so they take a DataFrame and
return a (n_buildings, n_hours) array; the fixtures below use a single-row stock.
"""

import numpy as np
import pandas as pd
import pytest

from cdm.thermodynamic import (
    FLOW_DTYPE,
    R_to_U,
    calc_cooling_demand_from_thermal_flows,
    calc_cooling_demand_metrics_for_df,
    calc_cooling_demand_percentile,
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


def test_R_to_U_is_elementwise_over_an_array() -> None:
    resistances = np.array([1.0, 5.0])
    assert np.allclose(R_to_U(resistances), [R_to_U(1.0), R_to_U(5.0)])


def test_calc_Q_infiltration_scales_linearly_with_temperature_difference() -> None:
    buildings = pd.DataFrame([{"volume_m3": 300.0, "infiltration_ACH": 0.5}])
    global_parameters = {"air_density": 1.2, "air_heat_capacity": 1005.0}

    delta_t = np.array([0.0, 1.0, 2.0, 4.0])
    time_series = {"T_outdoor_minus_indoor_C": delta_t}

    q = calc_Q_infiltration(buildings, time_series, global_parameters)

    mass_flow = 1.2 * 0.5 * 300.0 / 3600  # kg/s
    expected = mass_flow * 1005.0 * delta_t
    assert q.shape == (1, 4)
    assert np.allclose(q[0], expected)
    # Zero temperature difference must give zero heat flow.
    assert q[0, 0] == pytest.approx(0.0)


def test_calc_Q_transmission_matches_component_sum(
    buildings: pd.DataFrame,
    time_series: dict,
    global_parameters: dict,
) -> None:
    delta_t = time_series["T_outdoor_minus_indoor_C"]
    u_wall, u_roof, u_floor = (R_to_U(rc) for rc in (2.0, 3.0, 2.5))

    # Window, wall and roof scale with the air temperature difference;
    # the floor with the (constant) subsurface gradient.
    expected = (1.8 * 50.0 + u_wall * 150.0 + u_roof * 100.0) * delta_t + u_floor * 100.0 * (12.0 - 25.0)

    assert np.allclose(calc_Q_transmission(buildings, time_series, global_parameters)[0], expected)


def test_calc_Q_solar_radiation_sums_over_orientations(
    buildings: pd.DataFrame,
    building: pd.Series,
    time_series: dict,
) -> None:
    window_area = building["window_area_per_orientation_m2"]
    radiation = np.array([time_series[f"P_sol_{d}_W_m2"] for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]])
    expected = (window_area[:, None] * 0.6 * radiation).sum(axis=0)

    assert np.allclose(calc_Q_solar_radiation(buildings, time_series)[0], expected)


def test_calc_Q_internal_heat_combines_people_lighting_appliances(
    buildings: pd.DataFrame,
    time_series: dict,
    global_parameters: dict,
) -> None:
    expected = (
        30.0 * 100.0 * time_series["presence_people_office"]
        + 5.0 * 300.0 * time_series["presence_lighting_office"]
        + 8.0 * 300.0 * time_series["presence_appliances_office"]
    )

    assert np.allclose(calc_Q_internal_heat(buildings, time_series, global_parameters)[0], expected)


def test_calc_Q_ventilation_scales_with_occupancy(
    buildings: pd.DataFrame,
    time_series: dict,
    global_parameters: dict,
) -> None:
    q_vent = calc_Q_ventilation(buildings, time_series, global_parameters)

    rate = 1.2 * 30.0 * 30.0 * time_series["presence_people_office"] / 3600
    assert np.allclose(q_vent[0], rate * 1005.0 * time_series["T_outdoor_minus_indoor_C"])


def test_end_use_dependent_flows_use_each_buildings_own_presence_profile(
    building: pd.Series,
    time_series: dict,
    global_parameters: dict,
) -> None:
    # Two identical buildings apart from their end use must pick up different presence profiles.
    residential = building.to_dict() | {"end_use": "residential"}
    time_series = time_series | {
        "presence_people_residential": np.array([0.9, 0.1, 0.1, 0.9]),
        "presence_lighting_residential": np.array([0.8, 0.2, 0.2, 0.8]),
        "presence_appliances_residential": np.array([0.7, 0.3, 0.3, 0.7]),
    }
    buildings = pd.DataFrame([building.to_dict(), residential])

    q_vent = calc_Q_ventilation(buildings, time_series, global_parameters)
    q_internal = calc_Q_internal_heat(buildings, time_series, global_parameters)

    rate_residential = 1.2 * 30.0 * 30.0 * time_series["presence_people_residential"] / 3600
    assert np.allclose(q_vent[1], rate_residential * 1005.0 * time_series["T_outdoor_minus_indoor_C"])
    assert np.allclose(
        q_internal[1],
        30.0 * 100.0 * time_series["presence_people_residential"]
        + 5.0 * 300.0 * time_series["presence_lighting_residential"]
        + 8.0 * 300.0 * time_series["presence_appliances_residential"],
    )
    # The office row must be unaffected by the residential row sharing the same call.
    assert np.allclose(q_vent[0], calc_Q_ventilation(buildings.iloc[:1], time_series, global_parameters)[0])


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


def test_calc_cooling_demand_from_thermal_flows_is_vectorised_over_buildings() -> None:
    # Two buildings, the second with twice the internal gains of the first.
    q_internal = np.zeros((2, 8760))
    q_internal[0, 0] = 1000.0
    q_internal[1, 0] = 2000.0
    zeros = np.zeros((2, 8760))

    q_cooling, e_cooling_kwh, p_peak_kw = calc_cooling_demand_from_thermal_flows(zeros, zeros, zeros, zeros, q_internal)

    assert q_cooling.shape == (2, 8760)
    assert np.allclose(e_cooling_kwh, [1.0, 2.0])
    assert np.allclose(p_peak_kw, [1.0, 2.0])


def test_calc_cooling_demand_percentile_caps_peaks() -> None:
    # A flat year with two extreme spikes; capping at the 98th percentile removes the spikes.
    q = np.full(8760, 100.0)
    q[:2] = 50_000.0

    _, p_peak_percentile_kw, _, e_capped_kwh = calc_cooling_demand_percentile(q, n_percentile=98)

    uncapped_kwh = q.sum() / 1000
    # The 98th-percentile cap is far below the spike, so the capped energy is lower than the raw total.
    assert e_capped_kwh < uncapped_kwh
    assert p_peak_percentile_kw == pytest.approx(np.percentile(q, 98) / 1000)


def test_calc_cooling_demand_percentile_averages_over_years() -> None:
    # Two identical years -> the multi-year averages match the single-year result.
    one_year = np.full(8760, 100.0)
    one_year[:2] = 50_000.0
    two_years = np.concatenate([one_year, one_year])

    _, p_peak_two_years, _, e_two_years = calc_cooling_demand_percentile(two_years, n_percentile=98)
    _, p_peak_one_year, _, e_one_year = calc_cooling_demand_percentile(one_year, n_percentile=98)

    assert p_peak_two_years == pytest.approx(p_peak_one_year)
    assert e_two_years == pytest.approx(e_one_year)


def test_calc_cooling_demand_percentile_sorts_each_year_descending() -> None:
    q = np.tile(np.arange(8760, dtype=float), 2)

    q_sorted, _, q_capped, _ = calc_cooling_demand_percentile(q, n_percentile=98, include_time_series=True)

    assert q_sorted is not None
    assert q_capped is not None
    assert q_sorted.shape == q.shape
    # Each year is sorted independently, largest first.
    assert np.allclose(q_sorted[:8760], np.arange(8759, -1, -1))
    # Capping never raises a value.
    assert (q_capped <= q).all()


def test_calc_cooling_demand_metrics_for_df_runs_end_to_end(
    buildings: pd.DataFrame,
    time_series_full_year: dict,
    global_parameters: dict,
) -> None:
    result = calc_cooling_demand_metrics_for_df(buildings, time_series_full_year, global_parameters)

    # Positive aggregate metrics, and the hourly series are only returned when explicitly requested.
    assert result["E_cooling_kWh"].iloc[0] > 0
    assert result["P_cooling_peak_kW"].iloc[0] > 0
    assert "Q_cooling_demand_Wh" not in result.columns


def test_calc_cooling_demand_metrics_for_df_returns_hourly_series_on_request(
    buildings: pd.DataFrame,
    time_series_full_year: dict,
    global_parameters: dict,
) -> None:
    result = calc_cooling_demand_metrics_for_df(
        buildings,
        time_series_full_year,
        global_parameters,
        include_time_series=True,
    )

    q_cooling = result["Q_cooling_demand_Wh"].iloc[0]
    assert q_cooling.shape == (8760,)
    assert (q_cooling >= 0).all()
    # The five heat flows that make up the demand are returned alongside it.
    assert result["Q_transmission_Wh"].iloc[0].shape == (8760,)
    assert result["Q_cooling_capped_at_98th_percentile_Wh"].iloc[0].shape == (8760,)


def test_calc_cooling_demand_metrics_for_df_rejects_partial_years(
    buildings: pd.DataFrame,
    time_series_full_year: dict,
    global_parameters: dict,
) -> None:
    """A weather series that isn't a whole number of years fails loud, not with a cryptic reshape."""
    # 8760 + 24 hours: one extra day, as a leftover Feb 29 would produce.
    partial = {key: np.concatenate([value, value[:24]]) for key, value in time_series_full_year.items()}

    with pytest.raises(ValueError, match="whole number of 8760-hour years"):
        calc_cooling_demand_metrics_for_df(buildings, partial, global_parameters)


def test_calc_cooling_demand_metrics_for_df_is_chunk_invariant(
    building: pd.Series,
    time_series_full_year: dict,
    global_parameters: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Chunking is a memory optimisation; it must not change a single number.
    stock = pd.DataFrame([building.to_dict() | {"volume_m3": 900.0 * (1 + i)} for i in range(5)])

    unchunked = calc_cooling_demand_metrics_for_df(stock.copy(), time_series_full_year, global_parameters)
    monkeypatch.setattr("cdm.thermodynamic.MAX_CHUNK_CELLS", 8760 * 2)  # forces chunks of two buildings
    chunked = calc_cooling_demand_metrics_for_df(stock.copy(), time_series_full_year, global_parameters)

    assert np.allclose(unchunked["E_cooling_kWh"], chunked["E_cooling_kWh"])
    assert np.allclose(unchunked["P_cooling_peak_98th_percentile_kW"], chunked["P_cooling_peak_98th_percentile_kW"])


def test_heat_flows_stay_float32_and_reductions_accumulate_in_float64(
    buildings: pd.DataFrame,
    time_series_full_year: dict,
    global_parameters: dict,
) -> None:
    """The hourly blocks must be born float32, or one float64 input silently promotes them all.

    An np.empty() left at its default float64 is the easy way to lose this unnoticed.
    """
    time_series = {name: np.asarray(series, dtype=FLOW_DTYPE) for name, series in time_series_full_year.items()}
    flows = [
        calc_Q_transmission(buildings, time_series, global_parameters),
        calc_Q_infiltration(buildings, time_series, global_parameters),
        calc_Q_ventilation(buildings, time_series, global_parameters),
        calc_Q_solar_radiation(buildings, time_series),
        calc_Q_internal_heat(buildings, time_series, global_parameters),
    ]
    assert [flow.dtype for flow in flows] == [np.float32] * 5

    Q_cooling, E_cooling, P_peak = calc_cooling_demand_from_thermal_flows(*flows)
    assert Q_cooling.dtype == np.float32
    # The hour-axis sums span 8760+ terms; float32 accumulation would lose precision.
    assert E_cooling.dtype == np.float64
    assert P_peak.dtype == np.float64


def test_float32_flows_match_float64_on_annual_totals(
    building: pd.Series,
    time_series_full_year: dict,
    global_parameters: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # float32 is a memory/speed choice, not a modelling one: the annual totals must not move.
    stock = pd.DataFrame([building.to_dict() | {"volume_m3": 900.0 * (1 + i)} for i in range(5)])

    monkeypatch.setattr("cdm.thermodynamic.FLOW_DTYPE", np.float64)
    reference = calc_cooling_demand_metrics_for_df(stock.copy(), time_series_full_year, global_parameters)
    monkeypatch.setattr("cdm.thermodynamic.FLOW_DTYPE", np.float32)
    actual = calc_cooling_demand_metrics_for_df(stock.copy(), time_series_full_year, global_parameters)

    for column in ("E_cooling_kWh", "P_cooling_peak_kW", "P_cooling_peak_98th_percentile_kW"):
        assert np.allclose(actual[column], reference[column], rtol=1e-6), column
