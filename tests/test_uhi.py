"""Tests for the condition-scaled UHI fraction (Theeuwes kernel x diurnal phase)."""

import numpy as np
import pandas as pd

from cdm.uhi import add_UHI_fraction, theeuwes_weather_factor


def _day(date: str, t_amp: float, q_mid: float, wind: float) -> pd.DataFrame:
    """One synthetic day: sinusoidal T with amplitude t_amp, radiation q_mid at midday, constant wind."""
    hours = np.arange(1, 25)
    t = 15 + t_amp * np.sin((hours - 9) * np.pi / 12)
    q = np.where((hours > 6) & (hours < 22), q_mid * np.sin((hours - 6) * np.pi / 15).clip(0), 0.0)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(date) + pd.to_timedelta(hours - 1, unit="h"),
            "H": hours,
            "T_outdoor_raw_C": t,
            "Q": q,
            "wind_speed_m_s": wind,
        },
    )


def _year_like() -> pd.DataFrame:
    """A clear-calm day, an overcast-windy day, and 26 middling days (kernel normalization base)."""
    days = [_day("2021-06-01", t_amp=8, q_mid=250, wind=1.5), _day("2021-11-15", t_amp=2, q_mid=30, wind=9.0)]
    days += [_day(f"2021-05-{d:02d}", t_amp=5, q_mid=150, wind=4.0) for d in range(1, 27)]
    return pd.concat(days, ignore_index=True)


def test_fraction_bounded_and_complete() -> None:
    """UHI_fraction exists, has no NaN, and lies in [0, 1]."""
    out = add_UHI_fraction(_year_like(), uhi_day_fraction=0.25)
    frac = out["UHI_fraction"]
    assert frac.notna().all()
    assert float(frac.min()) >= 0.0
    assert float(frac.max()) <= 1.0


def test_clear_calm_night_realizes_ceiling() -> None:
    """On the most UHI-favourable day, the nocturnal fraction approaches 1."""
    out = add_UHI_fraction(_year_like(), uhi_day_fraction=0.25)
    best_day = out[out["date"].dt.date == pd.Timestamp("2021-06-01").date()]
    night = best_day[best_day["Q"] == 0]
    assert float(night["UHI_fraction"].max()) > 0.9


def test_windy_overcast_day_is_suppressed() -> None:
    """A windy, overcast, low-DTR day realizes little of the ceiling, day or night."""
    out = add_UHI_fraction(_year_like(), uhi_day_fraction=0.25)
    bad_day = out[out["date"].dt.date == pd.Timestamp("2021-11-15").date()]
    assert float(bad_day["UHI_fraction"].max()) < 0.35


def test_daytime_capped_by_day_fraction() -> None:
    """Daytime hours never exceed uhi_day_fraction of the day's realized factor (nocturnal phase)."""
    out = add_UHI_fraction(_year_like(), uhi_day_fraction=0.25)
    for _, day in out.groupby(out["date"].dt.date):
        daytime = day[day["Q"] > 0]["UHI_fraction"]
        nighttime = day[day["Q"] == 0]["UHI_fraction"]
        if len(daytime) and len(nighttime) and float(nighttime.max()) > 0:
            assert float(daytime.max()) <= 0.25 * float(nighttime.max()) + 1e-9


def test_weather_factor_monotonic_in_wind() -> None:
    """More wind, all else equal, means a smaller weather factor."""
    calm = _day("2021-06-01", t_amp=8, q_mid=250, wind=1.5)
    windy = _day("2021-06-01", t_amp=8, q_mid=250, wind=6.0)
    daily = pd.concat([calm, windy], ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    f_calm = theeuwes_weather_factor(calm)
    f_windy = theeuwes_weather_factor(windy)
    assert float(f_windy.iloc[0]) < float(f_calm.iloc[0])


def test_partial_nan_wind_day_does_not_poison_series() -> None:
    """One day with entirely missing wind (a gap in FH) still yields bounded, non-NaN fractions everywhere."""
    weather = _year_like()
    gap_day = weather["date"].dt.date == pd.Timestamp("2021-05-10").date()
    weather.loc[gap_day, "wind_speed_m_s"] = np.nan
    out = add_UHI_fraction(weather, uhi_day_fraction=0.25)
    assert out["UHI_fraction"].notna().all()
    assert float(out["UHI_fraction"].min()) >= 0.0
    assert float(out["UHI_fraction"].max()) <= 1.0


def test_full_nan_wind_column_falls_back_gracefully() -> None:
    """Backup-weather fallback path (no FH at all): wind_speed_m_s is NaN for every row."""
    weather = _year_like()
    weather["wind_speed_m_s"] = np.nan
    out = add_UHI_fraction(weather, uhi_day_fraction=0.25)
    frac = out["UHI_fraction"]
    assert frac.notna().all()
    assert float(frac.min()) >= 0.0
    assert float(frac.max()) <= 1.0
    # still discriminates on S/DTR alone: the clear day should realize more than the overcast one
    best_night = frac[(weather["date"].dt.date == pd.Timestamp("2021-06-01").date()) & (out["Q"] == 0)]
    bad_night = frac[(weather["date"].dt.date == pd.Timestamp("2021-11-15").date()) & (out["Q"] == 0)]
    assert float(best_night.max()) > float(bad_night.max())
