#!/usr/bin/env python3
"""Per-neighbourhood hourly cooling intensity, for the animated year time-lapse.

Runs the thesis heat-balance model over *every* building (SQ scenario), assigns
each to its buurt, and accumulates hourly cooling into per-buurt streams. The 5
weather years (2018-2022) are averaged into a typical year, then reduced to a
compact month x hour grid (12 x 24 = 288 frames) of cooling *intensity*
(W per m² of floor area). A fixed colour scale over these frames makes winter
read pale and summer afternoons saturate — the "city turns red" effect.

Same physics as build_temporal.py (thesis code untouched; only the footprint MBR
is reconstructed). Inputs are the git-ignored Zenodo geodata; output is small:
  dashboard/public/data/cooling_frames.json

Run: python dashboard/scripts/build_hourly_frames.py   (needs the model env)
Slow-ish (~1-2 min for 59k buildings); skips cleanly if the geodata is absent.
"""

# CLI build script (not the linted scientific package): prints, asserts, deferred imports are intentional.
# ruff: noqa: ANN001, ARG005, COM812, D103, EXE001, PLC0415, PLR0915, PLR2004, T201
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIVISIONS = REPO / "data" / "input" / "geodata" / "GeographicDivisions_TheHague.gpkg"
OUT = REPO / "dashboard" / "public" / "data" / "cooling_frames.json"
SCEN = "SQ"
HOURS_PER_YEAR = 8760
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def mbr_sides(geom) -> tuple[float, float]:
    """(width, length) of a footprint's minimum rotated rectangle; bbox fallback."""
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rect = geom.minimum_rotated_rectangle
    if rect.geom_type == "Polygon":
        xs, ys = rect.exterior.coords.xy
        if len(xs) >= 3:
            s1 = np.hypot(xs[1] - xs[0], ys[1] - ys[0])
            s2 = np.hypot(xs[2] - xs[1], ys[2] - ys[1])
            if s1 > 0 and s2 > 0:
                return min(s1, s2), max(s1, s2)
    minx, miny, maxx, maxy = geom.bounds
    return max(maxx - minx, 0.1), max(maxy - miny, 0.1)


def main(divisions: Path, geodata_dir: Path, out_path: Path) -> int:
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
    if not (gpkg.exists() and divisions.exists()):
        print("missing geodata — skipping hourly-frames build.")
        return 0

    par = REPO / f"data/input/parameters/parameters_{SCEN}"
    gp = read_global_parameters(par / "parameters_global.csv")
    bt = read_parameter_specific_data(par / "parameters_building_type.csv")
    ec = read_parameter_specific_data(par / "parameters_energy_class.csv")

    # --- buildings + buurt assignment (centroid within buurt) ---
    buurten = gpd.read_file(divisions, layer="Neighbourhoods_TheHague")[["buurtcode", "geometry"]]
    buildings = gpd.read_file(gpkg)
    pts = buildings.copy()
    pts["geometry"] = buildings.geometry.representative_point()
    joined = gpd.sjoin(pts, buurten, predicate="within").drop(columns="index_right")
    joined["geometry"] = buildings.geometry.loc[joined.index]  # restore polygon for MBR

    df = pd.DataFrame(joined.drop(columns="geometry"))
    df["MBR_width_m"], df["MBR_length_m"] = zip(*(mbr_sides(g) for g in joined.geometry), strict=True)
    print(f"buildings assigned to buurten: {len(df)} across {df['buurtcode'].nunique()} buurten")

    df = add_parameters_to_buildings(df, gp, bt, ec)
    df = add_derived_parameters_to_buildings(df, gp)

    # --- hourly weather/time series (offline, committed backup) ---
    orig = requests.post
    requests.post = lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = get_raw_weather_data(gp)
    requests.post = orig
    ts = create_time_series(
        gp,
        raw,
        "data/input/parameters/multidirectional_solar_radiation_fractions.csv",
        "data/input/parameters/presence_load_factors.csv",
    )
    n_hours = len(ts["T_outdoor_C"])
    n_years = n_hours // HOURS_PER_YEAR

    # --- accumulate hourly cooling (Wh) and floor area per buurt ---
    codes = sorted(df["buurtcode"].unique())
    accum = {c: np.zeros(n_hours) for c in codes}
    area = dict.fromkeys(codes, 0.0)
    buurt_col = df["buurtcode"].to_numpy()
    for i, (_, b) in enumerate(df.iterrows()):
        Qt = calc_Q_transmission(b, ts, gp)
        Qi = calc_Q_infiltration(b, ts, gp)
        Qv, _, _ = calc_Q_ventilation(b, ts, gp)
        Qs = calc_Q_solar_radiation(b, ts)
        Qh = calc_Q_internal_heat(b, ts, gp)
        Qc, _, _ = calc_cooling_demand_from_thermal_flows(Qt, Qi, Qv, Qs, Qh)
        c = buurt_col[i]
        accum[c] += Qc
        area[c] += b["floor_area_total_m2"]
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(df)} buildings")

    # --- typical year, then month x hour intensity (W/m²) per buurt ---
    dates = pd.to_datetime(ts["date"][:HOURS_PER_YEAR])
    month = dates.month.to_numpy()
    hour = dates.hour.to_numpy()

    active = [c for c in codes if area[c] > 0]
    frames = [[0.0] * len(active) for _ in range(12 * 24)]  # index = month*24 + hour
    for bi, c in enumerate(active):
        typ = accum[c].reshape(n_years, HOURS_PER_YEAR).mean(axis=0)  # mean W over each hour
        wm2 = typ / area[c]
        for m in range(12):
            mmask = month == m + 1
            for h in range(24):
                frames[m * 24 + h][bi] = round(float(wm2[mmask & (hour == h)].mean()), 3)

    flat = np.array([v for fr in frames for v in fr])
    vmax = round(float(np.percentile(flat, 99)), 3)  # robust ceiling for the colour scale

    data = {
        "meta": {
            "source": "thesis heat-balance model run over all buildings, aggregated per buurt",
            "scenario": SCEN,
            "weather_years": f"{int(gp['weather_data_start_year'])}-{int(gp['weather_data_end_year'])}",
            "metric": "mean cooling intensity, W per m² floor area",
            "vmax": vmax,
            "frame_order": "month-major, hour-minor (index = month*24 + hour)",
        },
        "months": MONTHS,
        "hours": list(range(24)),
        "buurtcodes": active,
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, separators=(",", ":")))
    print(
        f"wrote {out_path}  ({len(active)} buurten x 288 frames, "
        f"vmax {vmax} W/m², {out_path.stat().st_size / 1024:.0f} kB)"
    )
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the per-buurt hourly time-lapse frames.")
    ap.add_argument("--divisions", type=Path, default=DIVISIONS, help="GeographicDivisions GPKG")
    ap.add_argument(
        "--geodata-dir",
        type=Path,
        default=REPO / "data" / "output" / "geodata",
        help="dir with buildings_with_CDM_results_SQ_full.gpkg",
    )
    ap.add_argument("--out", type=Path, default=OUT, help="output JSON path")
    args = ap.parse_args()
    raise SystemExit(main(args.divisions, args.geodata_dir, args.out))
