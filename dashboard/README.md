# Cooling for Comfort: dashboard

This dashboard shows the results of my MSc thesis model: how much cooling the buildings of The
Hague need, and the life-cycle climate impact of that cooling. A map colours the city's 112
neighbourhoods (buurten) by cooling demand, profiles show a summer day and the months, and stacked
bars split the life-cycle greenhouse-gas emissions per scenario.

**[Open the live dashboard →](https://simonvanlierde.github.io/msc-thesis-ie/)**

![The dashboard's map view: cooling demand across The Hague's 112 neighbourhoods, shown on the 2050 Medium scenario](docs/screenshot.png)

## Run it locally

```bash
cd dashboard
pnpm install
pnpm dev          # http://localhost:5183
pnpm check        # typecheck + lint + coverage + build
pnpm test         # unit tests (data transforms)
pnpm test:e2e     # accessibility, light + dark
pnpm og           # rebuild the social card + favicon
```

The committed JSON in `public/data/` drives everything, so the dashboard runs and deploys without
the model inputs. `scripts/build_og.mjs` draws the social card, favicon and apple-touch icon from
the thesis output, with the Playwright Chromium the accessibility tests already install, and skips
cleanly when its inputs are missing.

## Where the data comes from

Every number comes from the thesis model. Three small build scripts (`scripts/*.py`) convert
the model outputs into JSON and GeoJSON for the web:

| Script | Output | Source | Notes |
| --- | --- | --- | --- |
| `build_scenarios.py` | `public/data/scenarios.json` (~47 kB) | `results/CDM_results_*.csv` (committed) | Per-archetype cooling + LCA totals for all 5 scenarios. A self-check reproduces the README headline (offices: 11% area, 54% demand, 79% GHG) from the built data. |
| `build_choropleth.py` | `public/data/cooling_by_buurt.geojson` (~127 kB) | per-building GPKG + CBS buurten (fetched by the `fetch_cbs_buurten` rule) | 59,381 buildings aggregated to 112 buurten by centroid, geometry simplified and reprojected to WGS84. Buurt sums match the archetype totals to 0.00% for the present-day (SQ) scenario. |
| `build_temporal.py` | `public/data/temporal.json` (~33 kB) | per-building GPKGs + committed weather/parameter CSVs | Re-runs the thesis heat-balance model on a stratified building sample, once per scenario (climate, UHI, comfort threshold and renovation come from the scenario parameters), averages 2021–2025 weather into a typical year and exports the hottest week hour by hour. Calibrated per use to the citywide archetype totals, so the magnitudes include projected building-stock growth. Per-building annual `E_cooling` reproduces the published value to ~0.03% median error (SQ; ~0.8% for 2050 M). |

To rebuild them, you need the repo's Python environment and the per-building GeoPackages from the
Zenodo dataset ([10.5281/zenodo.8344580](https://doi.org/10.5281/zenodo.8344580)) in
`data/input/geodata/` and `data/output/geodata/` (Git ignores those files). The buurt boundaries
come from CBS (Statistics Netherlands) through the pipeline.

```bash
# from the repo root, in the model venv
cd dashboard
pnpm data         # runs the three build_*.py scripts
```

`build_scenarios.py` needs only the Python standard library, so it always works. The other two
skip cleanly when the geodata is missing. Each script takes `--results-dir`, `--geodata-dir`,
`--divisions` and `--out`, defaulting to the committed reference results in `data/output/`.
`snakemake dashboard_data` regenerates all three from the pipeline's `results/` outputs.

### Caveats

- The page opens on the **2050 medium** path, so the map first shows that scenario. The
  **present-day (SQ)** scenario is one click away.
- Only in SQ do the per-building geometry and the archetype totals describe the same buildings,
  so only there do they agree exactly. The future scenarios add growth of the building stock to
  the archetype totals, but the per-building geometry does not include that growth. So their map
  sums are ~10–12% below the scenario total. The map still shows them, with a label that says so.
- The temporal profiles use representative building geometry. Only the footprint's minimum
  bounding rectangle (MBR) is rebuilt; the physics is the thesis code, unchanged. Today's
  buildings set the shape of each profile. Its size is calibrated per use to the same citywide
  totals as the impact charts, so all sections agree. To use an exact hourly export from a
  notebook run instead, replace `temporal.json`.

## Accessibility

- Every control works with the keyboard, with a skip link and always-visible focus.
- Every chart and the map have an accessible label and a **data table** fallback, so nothing is
  only available as a chart or only on hover.
- The colours are safe for colour-blind readers, and colour never carries meaning on its own.
- The hand-designed light and dark themes are both checked for contrast.
- `@axe-core/playwright` tests the production build in **both themes** (`pnpm test:e2e`). The
  test fails on any serious or critical WCAG 2.2 A/AA violation.

## Stack and deploy

Vite, React and TypeScript, with [nivo](https://nivo.rocks) for the charts,
[MapLibre GL](https://maplibre.org) for the map and [Biome](https://biomejs.dev) to lint and
format. The fonts ([Newsreader](https://fonts.google.com/specimen/Newsreader) and
[Public Sans](https://public-sans.digital.gov)) are served from this site, with no font CDN. The
map draws over the [CARTO](https://carto.com) basemaps, which need no API key. If that CDN is
unreachable, the map falls back to a plain background, so the neighbourhood colours still work
offline. The CARTO and OpenStreetMap credits show on the map when the basemap loads.

[`.github/workflows/dashboard.yml`](../.github/workflows/dashboard.yml) publishes the static build
in `dist/` to GitHub Pages on every push to `main` that changes `dashboard/`. Pull requests run the
same quality and accessibility checks, but do not deploy.
