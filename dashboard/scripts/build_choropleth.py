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

Uses fiona + shapely + pyproj directly (the pinned geopandas 0.13 is incompatible with
the installed fiona 1.10). Run:  python dashboard/scripts/build_choropleth.py
If the geodata isn't present it exits cleanly so the JSON build still works without it.
"""

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


def main() -> int:
    import fiona
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shp_transform
    from shapely.strtree import STRtree

    if not DIVISIONS.exists():
        print(f"missing {DIVISIONS.relative_to(REPO)} — skipping choropleth build.")
        return 0

    # --- load buurten, build a spatial index for point-in-polygon assignment ---
    codes: list[str] = []
    names: list[str] = []
    geoms = []
    with fiona.open(DIVISIONS, layer="Neighbourhoods_TheHague") as src:
        src_crs = src.crs
        for feat in src:
            p = feat["properties"]
            codes.append(p["buurtcode"])
            names.append(p["buurtnaam"])
            geoms.append(shape(feat["geometry"]))
    tree = STRtree(geoms)
    values: dict[str, dict] = {c: {"buurtcode": c, "buurtnaam": n} for c, n in zip(codes, names)}

    # --- sum building results into buurten, per scenario ---
    for scen in SCENARIOS:
        gpkg = BUILDINGS / f"buildings_with_CDM_results_{scen}_full.gpkg"
        if not gpkg.exists():
            print(f"  {scen}: no GPKG, skipping this scenario")
            continue
        acc: dict[str, dict] = {}
        assigned = 0
        with fiona.open(gpkg) as src:
            for feat in src:
                pt = shape(feat["geometry"]).representative_point()
                hits = tree.query(pt, predicate="within")
                if len(hits) == 0:
                    continue
                code = codes[int(hits[0])]
                p = feat["properties"]
                rec = acc.setdefault(code, {"n": 0})
                rec["n"] += 1
                for col in SUM_COLS:
                    rec[col] = rec.get(col, 0.0) + float(p.get(col) or 0.0)
                assigned += 1
        for code, rec in acc.items():
            out = values[code]
            for col, dst in SUM_COLS.items():
                out[f"{scen}__{dst}"] = round(rec[col], 1)
            out[f"{scen}__n_buildings"] = rec["n"]
        print(f"  {scen}: {assigned} buildings across {len(acc)} buurten")

    # --- simplify + reproject to WGS84, emit rounded GeoJSON ---
    import pyproj

    to_wgs84 = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform

    features = []
    for code, name, geom in zip(codes, names, geoms):
        props = values[code]
        if not any(k.endswith("__E_cooling_kWh") for k in props):
            continue  # buurt with no buildings (e.g. water) — drop it
        simplified = geom.simplify(SIMPLIFY_M, preserve_topology=True)
        wgs = shp_transform(to_wgs84, simplified)
        gj = _round_coords(mapping(wgs), COORD_DP)
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fc, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO)}  ({len(features)} buurten, {kb:.0f} kB)")
    return 0


def _round_coords(geom: dict, dp: int) -> dict:
    def rnd(x):
        if isinstance(x, (int, float)):
            return round(x, dp)
        return [rnd(v) for v in x]

    geom["coordinates"] = rnd(geom["coordinates"])
    return geom


if __name__ == "__main__":
    raise SystemExit(main())
