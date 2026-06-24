"""Unit tests for thermodynamic helper functions."""

import numpy as np
import pandas as pd
import pytest

from functions.thermodynamic import R_to_U, calc_Q_infiltration


def test_R_to_U_against_formula() -> None:
    # U = 1 / (1/alfa_i + Rc + 1/alfa_o)
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
