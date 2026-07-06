"""Fetch the KNMI hourly weather series as a declared, cached pipeline input.

Wraps ``functions.time_series.get_raw_weather_data`` so the weather series is
fetched once (per station/year window from config) instead of being pulled live
inside every model run. Falls back to the committed backup automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from functions.time_series import get_raw_weather_data


def main() -> None:
    """Fetch the weather series for one station/window and write it to CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    weather = get_raw_weather_data(
        {
            "weather_data_start_year": args.start_year,
            "weather_data_end_year": args.end_year,
            "weather_station": args.station,
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output, index=False)


if __name__ == "__main__":
    main()
