"""Benchmarks for the hourly weather time series preparation (cdm/time_series.py, cdm/uhi.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdm.time_series import add_seasonal_solar_radiation_boosts, add_seasonal_temperature_boosts
from cdm.uhi import add_UHI_fraction, theeuwes_weather_factor

if TYPE_CHECKING:
    import pandas as pd
    from pytest_codspeed import BenchmarkFixture


def test_theeuwes_weather_factor(benchmark: BenchmarkFixture, weather_df: pd.DataFrame) -> None:
    result = benchmark(theeuwes_weather_factor, weather_df)
    assert result.notna().all()


def test_add_UHI_fraction(benchmark: BenchmarkFixture, weather_df: pd.DataFrame) -> None:
    result = benchmark(add_UHI_fraction, weather_df, 0.2)
    assert result["UHI_fraction"].between(0.0, 1.0).all()


def test_add_seasonal_temperature_boosts(benchmark: BenchmarkFixture, weather_df: pd.DataFrame) -> None:
    result = benchmark(add_seasonal_temperature_boosts, weather_df, 0.5, 0.7, 1.0, 0.6)
    assert "delta_T_season_C" in result.columns


def test_add_seasonal_solar_radiation_boosts(benchmark: BenchmarkFixture, weather_df: pd.DataFrame) -> None:
    result = benchmark(add_seasonal_solar_radiation_boosts, weather_df, 0.05, 0.02)
    assert "P_sol_total_W_m2" in result.columns
