#!/usr/bin/env python3
"""Aggregate per-building cooling results to neighbourhood (buurt) level for the map.

Inputs (git-ignored, dropped in locally from the Zenodo dataset 10.5281/zenodo.8344580):
  data/output/geodata/buildings_with_CDM_results_{scenario}_full.gpkg  — 59k building polygons
  data/input/geodata/GeographicDivisions_TheHague.gpkg (layer Neighbourhoods_TheHague) — 114 buurten

Output:
  dashboard/public/data/cooling_by_buurt.geojson — one FeatureCollection, geometry per
  buurt (simplified, WGS84), cooling totals for every scenario as properties.

Each building is assigned to the buurt containing its centroid (buurten partition the
city, so no double counting). We ship buurt geometry once with per-scenario values, not
raw building polygons — 114 simplified polygons instead of 59k, keeping the map fast.

Run:  python dashboard/scripts/build_choropleth.py
If the geodata isn't present it exits cleanly so the JSON build still works without it.
"""

# CLI build script (not the linted scientific package): prints, asserts, deferred imports are intentional.
# ruff: noqa: ANN001, ANN202, D103, EXE001, PLC0415, T201
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIVISIONS = REPO / "data" / "input" / "geodata" / "GeographicDivisions_TheHague.gpkg"
BUILDINGS = REPO / "data" / "output" / "geodata"
OUT = REPO / "dashboard" / "public" / "data" / "cooling_by_buurt.geojson"

SCENARIOS = ["SQ", "2030", "2050_L", "2050_M", "2050_H"]
SIMPLIFY_M = 15  # geometry simplification tolerance, metres (source CRS EPSG:28992 is metric)
COORD_DP = 5  # ~1 m; plenty for a city choropleth, and keeps the file small

# Per-building GPKG field -> output property name; all summed within each buurt.
SUM_COLS = {
    "E_cooling_kWh": "E_cooling_kWh",
    "floor_area_total_m2": "floor_area_m2",
    "GHG_emissions_total_kgCO2eq": "GHG_kgCO2eq",
    "population": "population",
}


def main(divisions: Path, buildings_dir: Path, out_path: Path) -> int:
    import geopandas as gpd
    from shapely.geometry import mapping

    if not divisions.exists():
        print(f"missing {divisions} — skipping choropleth build.")
        return 0

    # --- load buurten (the aggregation units) ---
    buurten = gpd.read_file(divisions, layer="Neighbourhoods_TheHague")[["buurtcode", "buurtnaam", "geometry"]]
    values: dict[str, dict] = {
        r.buurtcode: {"buurtcode": r.buurtcode, "buurtnaam": r.buurtnaam} for r in buurten.itertuples()
    }

    # --- sum building results into buurten, per scenario (centroid within buurt) ---
    for scen in SCENARIOS:
        gpkg = buildings_dir / f"buildings_with_CDM_results_{scen}_full.gpkg"
        if not gpkg.exists():
            print(f"  {scen}: no GPKG, skipping this scenario")
            continue
        b = gpd.read_file(gpkg)[[*SUM_COLS, "geometry"]]
        b["geometry"] = b.geometry.representative_point()
        joined = gpd.sjoin(b, buurten[["buurtcode", "geometry"]], predicate="within")
        agg = joined.groupby("buurtcode")[list(SUM_COLS)].sum()
        n = joined.groupby("buurtcode").size()
        for code, row in agg.iterrows():
            out = values[code]
            for col, dst in SUM_COLS.items():
                out[f"{scen}__{dst}"] = round(float(row[col]), 1)
            out[f"{scen}__n_buildings"] = int(n[code])
        print(f"  {scen}: {int(n.sum())} buildings across {len(agg)} buurten")

    # --- simplify + reproject to WGS84, emit rounded GeoJSON ---
    buurten["geometry"] = buurten.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    buurten = buurten.to_crs("EPSG:4326")

    features = []
    for r in buurten.itertuples():
        props = values[r.buurtcode]
        if not any(k.endswith("__E_cooling_kWh") for k in props):
            continue  # buurt with no buildings (e.g. water) — drop it
        gj = _round_coords(mapping(r.geometry), COORD_DP)
        features.append({"type": "Feature", "properties": props, "geometry": gj})

    fc = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "buildings_with_CDM_results_*.gpkg aggregated to CBS buurten",
            "doi_data": "10.5281/zenodo.8344580",
            "scenarios": SCENARIOS,
            "note": "values are per-buurt sums of building-level model output",
        },
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, separators=(",", ":")))
    print(f"wrote {out_path}  ({len(features)} buurten, {out_path.stat().st_size / 1024:.0f} kB)")
    return 0


def _round_coords(geom: dict, dp: int) -> dict:
    def rnd(x):
        if isinstance(x, (int, float)):
            return round(x, dp)
        return [rnd(v) for v in x]

    geom["coordinates"] = rnd(geom["coordinates"])
    return geom


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate per-building cooling results to buurt GeoJSON.")
    ap.add_argument("--divisions", type=Path, default=DIVISIONS, help="GeographicDivisions GPKG")
    ap.add_argument("--geodata-dir", type=Path, default=BUILDINGS, help="dir with buildings_with_CDM_results_*.gpkg")
    ap.add_argument("--out", type=Path, default=OUT, help="output GeoJSON path")
    args = ap.parse_args()
    raise SystemExit(main(args.divisions, args.geodata_dir, args.out))
