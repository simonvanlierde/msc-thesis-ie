"""Condition-scaled urban heat island realization.

The Habib et al. (2025) raster supplies each building's UHImax ceiling (the
location term of the Theeuwes et al. 2016 diagnostic, doi 10.1002/joc.4717).
This module supplies the *temporal* part: the diagnostic's weather kernel
(S.DTR^3/U)^(1/4), normalized over the reference period, times a
nocturnal-phased diurnal profile. The product, UHI_fraction in [0, 1], is the
share of a building's ceiling realized in a given hour:

    T_outdoor(b, h) = T_base(h) + UHI_max_C(b) * uhi_scale * UHI_fraction(h)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The fraction of the reference-period kernel distribution treated as "full
# realization". p98 rather than max so one freak day does not deflate all others.
_KERNEL_NORM_PERCENTILE = 98

# Fallback daily wind speed (m/s) when wind_speed_m_s is NaN for the whole
# reference period (backup-weather CSV has no FH column at all). Midpoint of
# the plausible coastal-station annual mean range documented in
# tests/test_weather_wind.py (3-10 m/s at Hoek van Holland).
_FALLBACK_WIND_M_S = 5.0


def _daily_aggregates(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Per-day S (mean radiation), DTR (temperature range) and U (mean wind).

    S is the hourly-mean radiation Q in J/cm^2/h (KNMI units), not W/m^2. The
    kernel is normalized by its own p98 below, so the unit constant cancels
    and using Q directly is fine.
    """
    day = weather_df["date"].dt.date
    grouped = weather_df.groupby(day)
    daily_wind = grouped["wind_speed_m_s"].mean()  # NaN for a day only if every hour that day is NaN
    # NaN-wind policy: the backup-weather fallback path has no FH column, so
    # wind_speed_m_s can be NaN for an entire day (gap) or the whole series
    # (full fallback). A leftover NaN would poison the whole kernel through
    # np.percentile in theeuwes_weather_factor's caller. Fill missing days
    # with the period's own mean wind; if the whole period is missing (full
    # fallback), fall back to a fixed plausible constant.
    period_mean_wind = daily_wind.mean()
    fallback = period_mean_wind if pd.notna(period_mean_wind) else _FALLBACK_WIND_M_S
    return pd.DataFrame(
        {
            "S": grouped["Q"].mean(),
            "DTR": grouped["T_outdoor_raw_C"].max() - grouped["T_outdoor_raw_C"].min(),
            "U": daily_wind.fillna(fallback).clip(lower=0.5),  # avoid division blow-up in dead calm
        },
    )


def theeuwes_weather_factor(weather_df: pd.DataFrame) -> pd.Series:
    """The Theeuwes weather kernel (S.DTR^3/U)^(1/4) per day, unnormalized."""
    daily = _daily_aggregates(weather_df)
    return (daily["S"] * daily["DTR"] ** 3 / daily["U"]) ** 0.25


def add_UHI_fraction(weather_df: pd.DataFrame, uhi_day_fraction: float) -> pd.DataFrame:
    """Add the hourly UHI_fraction column: normalized daily kernel x diurnal phase.

    Args:
        weather_df: hourly weather with date, H, T_outdoor_raw_C, Q, wind_speed_m_s.
        uhi_day_fraction: share of the nocturnal UHI realized in daytime hours
            (canopy UHI peaks at night; daytime is small but non-zero).

    Returns:
        The same DataFrame with UHI_fraction in [0, 1].
    """
    kernel = theeuwes_weather_factor(weather_df)
    norm = np.percentile(kernel.to_numpy(), _KERNEL_NORM_PERCENTILE)
    daily_factor = (kernel / norm).clip(0.0, 1.0)

    day = weather_df["date"].dt.date
    factor_by_hour = day.map(daily_factor).astype(float)
    is_daytime = weather_df["Q"].to_numpy() > 0  # radiation-based day/night: no astronomy dependency
    diurnal = np.where(is_daytime, uhi_day_fraction, 1.0)

    weather_df["UHI_fraction"] = factor_by_hour.to_numpy() * diurnal
    return weather_df
