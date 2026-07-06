#!/usr/bin/env python3
"""Reconstruct the citywide diurnal & seasonal cooling-demand profile.

The thesis model computes hourly cooling per building but only exports annual
aggregates. This step re-runs the model's *own* heat-balance functions on a
stratified sample of the real buildings (from the dropped-in GPKG), sums the
hourly cooling, averages the 5 weather years (2018-2022) into a typical year,
and calibrates the annual total per building use to the published citywide
E_cooling. Physics is the thesis code untouched; only the building MBR
(width/length) is recomputed from the footprint because the results GPKG omits
it. Validated: per-building annual E_cooling reproduces the published value to
~0.03% median error (see dashboard/README).

Inputs (git-ignored, from Zenodo 10.5281/zenodo.8344580):
  data/output/geodata/buildings_with_CDM_results_SQ_full.gpkg
Committed inputs: weather + parameter CSVs under data/input/parameters/.

Output: dashboard/public/data/temporal.json
Run:    python dashboard/scripts/build_temporal.py   (needs the model env / .venv)
"""

# CLI build script (not the linted scientific package): prints, asserts, deferred imports are intentional.
# ruff: noqa: ANN002, ANN003, ANN202, B007, C901, D103, EXE001, PLC0415, PLR0915, PLR2004, RSE102, T201
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "dashboard" / "public" / "data" / "temporal.json"
SCEN = "SQ"  # present-day; params carry no climate boost, so this is "today's" profile
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


def main(geodata_dir: Path, out_path: Path) -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import requests

    from functions.data_handling import (
        add_derived_parameters_to_buildings,
        add_parameters_to_buildings,
        read_global_parameters,
        read_parameter_specific_data,
    )
    from functions.thermodynamic import (
        calc_cooling_demand_from_thermal_flows,
        calc_Q_infiltration,
        calc_Q_internal_heat,
        calc_Q_solar_radiation,
        calc_Q_transmission,
        calc_Q_ventilation,
    )
    from functions.time_series import create_time_series, get_raw_weather_data

    gpkg = geodata_dir / f"buildings_with_CDM_results_{SCEN}_full.gpkg"
    if not gpkg.exists():
        print(f"missing {gpkg} — skipping temporal build.")
        return 0

    par = REPO / f"data/input/parameters/parameters_{SCEN}"
    gp = read_global_parameters(par / "parameters_global.csv")
    bt = read_parameter_specific_data(par / "parameters_building_type.csv")
    ec = read_parameter_specific_data(par / "parameters_energy_class.csv")

    # --- read every building's use + published E_cooling, keep a stratified sample ---
    gdf = gpd.read_file(gpkg).reset_index(drop=True)
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

    # --- weather/time series: force the committed backup (offline, deterministic) ---
    orig_post = requests.post

    def _fail(*_a, **_k):
        raise requests.exceptions.Timeout()

    requests.post = _fail
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = get_raw_weather_data(gp)
    requests.post = orig_post
    ts = create_time_series(
        gp,
        raw,
        "data/input/parameters/multidirectional_solar_radiation_fractions.csv",
        "data/input/parameters/presence_load_factors.csv",
    )
    n_hours = len(ts["T_outdoor_C"])
    n_years = n_hours // HOURS_PER_YEAR

    # --- accumulate hourly cooling per use, plus sampled annual for calibration ---
    streams = {use: np.zeros(n_hours) for use in published_by_use}
    sampled_annual = dict.fromkeys(published_by_use, 0.0)
    for _, b in sample.iterrows():
        Qt = calc_Q_transmission(b, ts, gp)
        Qi = calc_Q_infiltration(b, ts, gp)
        Qv, _, _ = calc_Q_ventilation(b, ts, gp)
        Qs = calc_Q_solar_radiation(b, ts)
        Qh = calc_Q_internal_heat(b, ts, gp)
        Qc, E, _ = calc_cooling_demand_from_thermal_flows(Qt, Qi, Qv, Qs, Qh)
        streams[b["end_use"]] += Qc  # Wh per hour
        sampled_annual[b["end_use"]] += E  # kWh/yr

    # calibrate each use's stream so its annual total matches the published citywide value
    for use in streams:
        scale = published_by_use[use] / sampled_annual[use] if sampled_annual[use] else 0.0
        streams[use] *= scale

    # typical year: average the whole-city hourly Wh across the 5 years
    typical = {use: s.reshape(n_years, HOURS_PER_YEAR).mean(axis=0) for use, s in streams.items()}

    # --- aggregate to diurnal-by-season (mean GW) and monthly (GWh) ---
    dates = pd.to_datetime(ts["date"][:HOURS_PER_YEAR])
    hour = dates.hour.to_numpy()
    month = dates.month.to_numpy()
    season = np.array([SEASON_OF_MONTH[m] for m in month])
    seasons = ["Summer", "Spring", "Autumn", "Winter"]

    uses = sorted(typical)  # e.g. ["office", "residential"]

    diurnal = {}
    for se in seasons:
        smask = season == se
        per_use = {}
        for use in uses:
            arr = typical[use]
            per_use[use] = [round(float(arr[smask & (hour == h)].mean()) / 1e9, 5) for h in range(24)]
        per_use["total"] = [round(sum(per_use[u][h] for u in uses), 5) for h in range(24)]
        diurnal[se] = per_use

    monthly = {}
    for use in uses:
        arr = typical[use]
        monthly[use] = [round(float(arr[month == m + 1].sum()) / 1e9, 3) for m in range(12)]  # Wh -> GWh
    monthly["total"] = [round(sum(monthly[u][m] for u in uses), 3) for m in range(12)]

    data = {
        "meta": {
            "source": "thesis heat-balance model re-run on a stratified building sample, "
            "calibrated to published citywide E_cooling",
            "scenario": SCEN,
            "weather_years": f"{int(gp['weather_data_start_year'])}-{int(gp['weather_data_end_year'])}",
            "sample_buildings": len(sample),
            "validation": "per-building annual E_cooling reproduced to ~0.03% median error",
            "units": {"diurnal": "mean cooling power, GW", "monthly": "cooling energy, GWh"},
        },
        "hour_of_day": list(range(24)),
        "months": MONTHS,
        "seasons": seasons,
        "uses": uses,
        "diurnal_by_season": diurnal,
        "monthly": monthly,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, separators=(",", ":")))
    ann = sum(sum(monthly[u]) for u in uses)
    print(
        f"wrote {out_path}  (sample {len(sample)}, annual {ann:.0f} GWh, {out_path.stat().st_size / 1024:.1f} kB)",
    )
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Reconstruct the diurnal/seasonal cooling profile.")
    ap.add_argument(
        "--geodata-dir",
        type=Path,
        default=REPO / "data" / "output" / "geodata",
        help="dir with buildings_with_CDM_results_SQ_full.gpkg",
    )
    ap.add_argument("--out", type=Path, default=OUT, help="output JSON path")
    args = ap.parse_args()
    raise SystemExit(main(args.geodata_dir, args.out))
