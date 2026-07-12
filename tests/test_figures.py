"""Smoke tests for the plotting helpers (visual output is not asserted)."""

import pandas as pd

from cdm.figures import get_indices_from_dates


def test_get_indices_from_dates_returns_matching_row_positions() -> None:
    time_series = pd.DataFrame({"date": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"])})

    start_index, end_index = get_indices_from_dates(time_series, "2022-01-02", "2022-01-04")

    assert start_index == 1
    assert end_index == 3
