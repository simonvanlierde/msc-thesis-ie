#!/usr/bin/env python3
"""Reconstruct the citywide diurnal & seasonal cooling-demand profile.

The thesis model computes hourly cooling per building but only exports annual
aggregates. This step re-runs the model's *own* heat-balance functions on a
stratified sample of the real buildings (from the dropped-in GPKGs), sums the
hourly cooling, averages the 5 weather years (2021-2025) into a typical year,
and calibrates the annual total per building use to the citywide archetype
totals (which carry the projected building-stock growth). It runs once per
scenario — the scenario parameters carry the climate boosts, comfort threshold
and UHI — so the dashboard's profiles follow the chosen path. Physics is the thesis code untouched; only the building MBR
(width/length) is recomputed from the footprint because the results GPKGs omit
it. Validated: per-building annual E_cooling reproduces the published value to
~0.03% median error (see dashboard/README).

Inputs (git-ignored, from Zenodo 10.5281/zenodo.8344580):
  data/output/geodata/buildings_with_CDM_results_<scenario>_full.gpkg
Committed inputs: weather + parameter CSVs under data/input/parameters/.

Output: dashboard/public/data/temporal.json
Run:    python dashboard/scripts/build_temporal.py   (needs the model env / .venv)
"""

# CLI build script (not the linted scientific package): prints, asserts, deferred imports are intentional.
# ruff: noqa: B007, C901, D103, EXE001, PLC0415, PLR0912, PLR0915, PLR2004, T201
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "dashboard" / "public" / "data" / "temporal.json"
SCENARIOS = ["SQ", "2030", "2050_L", "2050_M", "2050_H"]
SAMPLE_PER_USE = 3000  # buildings per end-use; calibration fixes magnitude, sample sets shape
HOURS_PER_YEAR = 8760
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# meteorological seasons by month index (1-12)
SEASON_OF_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}


def main(
    geodata_dir: Path,
    out_path: Path,
    weather_csv: Path | None = None,
    results_dir: Path | None = None,
) -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from cdm.parameters import add_derived_parameters_to_buildings, add_parameters_to_buildings
    from cdm.readers import read_global_parameters, read_parameter_specific_data
    from cdm.thermodynamic import (
        calc_cooling_demand_from_thermal_flows,
        calc_Q_infiltration,
        calc_Q_internal_heat,
        calc_Q_solar_radiation,
        calc_Q_transmission,
        calc_Q_ventilation,
    )
    from cdm.time_series import create_time_series, get_raw_weather_data

    pdir = REPO / "data/input/parameters"
    for scen in SCENARIOS:
        gpkg = geodata_dir / f"buildings_with_CDM_results_{scen}_full.gpkg"
        if not gpkg.exists():
            print(f"missing {gpkg} — skipping temporal build.")
            return 0

    # The raw KNMI series depends only on the station and year range, which the scenarios
    # share; the per-scenario climate boosts are applied in create_time_series below.
    # Prefer the pipeline's fetched CSV (the fetch_weather rule); fall back to a live KNMI
    # fetch for standalone runs. The committed backup only covers 2018-2022, so the live
    # path fails loudly when KNMI is down and the requested window is newer.
    if weather_csv and weather_csv.exists():
        raw = pd.read_csv(weather_csv, parse_dates=["date"])
    else:
        gp_sq = read_global_parameters(pdir / "parameters.toml", "SQ")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = get_raw_weather_data(gp_sq)

    by_scenario: dict[str, dict] = {}
    legacy: dict = {}  # top-level SQ fields, kept for the OG-image build and validation

    for scen in SCENARIOS:
        gp = read_global_parameters(pdir / "parameters.toml", scen)
        bt = read_parameter_specific_data(pdir / "parameters_building_type.csv", scen)
        ec = read_parameter_specific_data(pdir / "parameters_energy_class.csv", scen)

        # --- read every building's use + published E_cooling, keep a stratified sample ---
        gdf = gpd.read_file(geodata_dir / f"buildings_with_CDM_results_{scen}_full.gpkg")
        gdf = gdf.reset_index(drop=True)
        geoms = list(gdf.geometry)  # shapely geometries, positionally aligned to allb
        allb = pd.DataFrame(gdf.drop(columns="geometry"))
        published_by_use = allb.groupby("end_use")["E_cooling_kWh"].sum().to_dict()

        rng = np.random.default_rng(42)  # fixed seed → reproducible build
        keep = []
        for use, grp in allb.groupby("end_use"):
            idx = grp.index.to_numpy()
            if len(idx) > SAMPLE_PER_USE:
                idx = rng.choice(idx, SAMPLE_PER_USE, replace=False)
            keep.extend(idx.tolist())
        sample = allb.loc[keep].copy()

        # recompute MBR width/length from the footprint (GPKG omits them)
        w, ln = [], []
        for i in keep:
            geom = geoms[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rect = geom.minimum_rotated_rectangle
            xs, ys = rect.exterior.coords.xy if rect.geom_type == "Polygon" else ([], [])
            s1 = np.hypot(xs[1] - xs[0], ys[1] - ys[0]) if len(xs) >= 3 else 0.0
            s2 = np.hypot(xs[2] - xs[1], ys[2] - ys[1]) if len(xs) >= 3 else 0.0
            if not (s1 > 0 and s2 > 0):  # degenerate footprint: fall back to axis-aligned bbox
                minx, miny, maxx, maxy = geom.bounds
                s1, s2 = max(maxx - minx, 0.1), max(maxy - miny, 0.1)
            w.append(min(s1, s2))
            ln.append(max(s1, s2))
        sample["MBR_width_m"], sample["MBR_length_m"] = w, ln

        sample = add_parameters_to_buildings(sample, gp, bt, ec)
        sample = add_derived_parameters_to_buildings(sample, gp)

        # --- scenario time series: climate boosts, UHI and comfort threshold live in gp ---
        ts = create_time_series(
            gp,
            raw,
            "data/input/parameters/multidirectional_solar_radiation_fractions.csv",
            "data/input/parameters/presence_load_factors.csv",
        )
        n_hours = len(ts["T_outdoor_C"])
        n_years = n_hours // HOURS_PER_YEAR

        # --- accumulate hourly cooling per use, plus sampled annual for calibration ---
        # The cdm flow functions are vectorised over buildings, returning (n_buildings, n_hours).
        # Chunked so the five flow arrays stay ~200 MB instead of ~1 GB each for the full sample.
        chunk = 256
        streams = {use: np.zeros(n_hours) for use in published_by_use}
        sampled_annual = dict.fromkeys(published_by_use, 0.0)
        for start in range(0, len(sample), chunk):
            batch = sample.iloc[start : start + chunk]
            Qt = calc_Q_transmission(batch, ts, gp)
            Qi = calc_Q_infiltration(batch, ts, gp)
            Qv = calc_Q_ventilation(batch, ts, gp)
            Qs = calc_Q_solar_radiation(batch, ts)
            Qh = calc_Q_internal_heat(batch, ts, gp)
            Qc, E, _ = calc_cooling_demand_from_thermal_flows(Qt, Qi, Qv, Qs, Qh)
            for use in streams:
                mask = (batch["end_use"] == use).to_numpy()
                if mask.any():
                    streams[use] += Qc[mask].sum(axis=0, dtype=np.float64)  # Wh per hour
                    sampled_annual[use] += float(E[mask].sum())  # kWh/yr

        # Calibrate each use's stream to the citywide archetype total (committed CDM_results
        # CSV, the same source as scenarios.json) rather than the per-building GPKG sum:
        # the archetype totals carry the scenario's projected building-stock growth that the
        # per-building geometry does not. Today's stock sets the shape, the projection the size.
        arch = pd.read_csv((results_dir or REPO / "data" / "output") / f"CDM_results_{scen}_full.csv")
        arch_use = np.where(
            arch["building_type"].str.lower().str.contains("office"),
            "office",
            "residential",
        )
        target_by_use = arch["E_cooling_kWh"].groupby(arch_use).sum().to_dict()
        for use in streams:
            scale = target_by_use[use] / sampled_annual[use] if sampled_annual[use] else 0.0
            streams[use] *= scale

        # typical year: average the whole-city hourly Wh across the 5 years
        typical = {use: s.reshape(n_years, HOURS_PER_YEAR).mean(axis=0) for use, s in streams.items()}

        dates = pd.to_datetime(ts["date"][:HOURS_PER_YEAR])
        month = dates.month.to_numpy()
        uses = sorted(typical)  # e.g. ["office", "residential"]

        monthly = {}
        for use in uses:
            arr = typical[use]
            monthly[use] = [round(float(arr[month == m + 1].sum()) / 1e9, 3) for m in range(12)]
        monthly["total"] = [round(sum(monthly[u][m] for u in uses), 3) for m in range(12)]

        # --- heatwave week: the 7 consecutive days with the most cooling in the full record ---
        # Taken from the real (calibrated) hourly streams, not the typical year, so the peak
        # keeps its true height instead of being averaged flat across the 5 weather years.
        total_stream = sum(streams[use] for use in uses)
        daily = total_stream[: n_years * HOURS_PER_YEAR].reshape(-1, 24).sum(axis=1)
        week_sums = np.convolve(daily, np.ones(7), "valid")
        h0 = int(week_sums.argmax()) * 24
        # KNMI hourly timestamps are UTC; shift to local clock time for the chart labels
        hw_dates = pd.to_datetime(ts["date"])[h0 : h0 + 7 * 24].tz_localize("UTC").tz_convert("Europe/Amsterdam")
        heatwave = {
            "dates": [d.strftime("%Y-%m-%dT%H:00") for d in hw_dates],
            "series": {use: [round(float(x) / 1e9, 5) for x in streams[use][h0 : h0 + 7 * 24]] for use in uses},
        }

        by_scenario[scen] = {"monthly": monthly, "heatwave": heatwave}
        ann = sum(sum(monthly[u]) for u in uses)
        print(
            f"{scen}: sample {len(sample)}, annual {ann:.0f} GWh, "
            f"heatwave {heatwave['dates'][0][:10]} - {heatwave['dates'][-1][:10]}",
        )

        if scen == "SQ":
            # Diurnal-by-season shapes, kept top-level for the OG-image heat band.
            hour = dates.hour.to_numpy()
            season = np.array([SEASON_OF_MONTH[m] for m in month])
            seasons = ["Summer", "Spring", "Autumn", "Winter"]
            diurnal = {}
            for se in seasons:
                smask = season == se
                per_use = {}
                for use in uses:
                    arr = typical[use]
                    per_use[use] = [round(float(arr[smask & (hour == h)].mean()) / 1e9, 5) for h in range(24)]
                per_use["total"] = [round(sum(per_use[u][h] for u in uses), 5) for h in range(24)]
                diurnal[se] = per_use
            legacy = {
                "weather_years": f"{int(gp['weather_data_start_year'])}-{int(gp['weather_data_end_year'])}",
                "sample_buildings": len(sample),
                "seasons": seasons,
                "uses": uses,
                "diurnal_by_season": diurnal,
                "monthly": monthly,
            }

    data = {
        "meta": {
            "source": "thesis heat-balance model re-run on a stratified building sample, "
            "per scenario, calibrated to the published citywide E_cooling",
            "weather_years": legacy["weather_years"],
            "sample_buildings": legacy["sample_buildings"],
            "validation": "per-building annual E_cooling reproduced to ~0.03% median error",
            "units": {
                "diurnal": "mean cooling power, GW",
                "monthly": "cooling energy, GWh",
                "heatwave": "cooling power, GW",
            },
        },
        "hour_of_day": list(range(24)),
        "months": MONTHS,
        # Top-level profile fields are the present-day (SQ) scenario, consumed by the
        # OG-image build; the dashboard reads by_scenario.
        "seasons": legacy["seasons"],
        "uses": legacy["uses"],
        "diurnal_by_season": legacy["diurnal_by_season"],
        "monthly": legacy["monthly"],
        "by_scenario": by_scenario,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Reconstruct the diurnal/seasonal cooling profile.")
    ap.add_argument(
        "--geodata-dir",
        type=Path,
        default=REPO / "data" / "output" / "geodata",
        help="dir with buildings_with_CDM_results_<scenario>_full.gpkg",
    )
    ap.add_argument("--out", type=Path, default=OUT, help="output JSON path")
    ap.add_argument("--weather-csv", type=Path, default=None, help="fetched KNMI weather CSV (skips the live fetch)")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="dir with the CDM_results_<scenario>_full.csv calibration targets (default: data/output)",
    )
    args = ap.parse_args()
    raise SystemExit(main(args.geodata_dir, args.out, args.weather_csv, args.results_dir))
