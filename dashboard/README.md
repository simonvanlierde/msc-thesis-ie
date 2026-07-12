# Cooling for Comfort — dashboard

An interactive dashboard for the MSc thesis model of building cooling demand and its
life-cycle climate impact in The Hague. It turns the thesis result files into three views:

- **Where** — a choropleth of cooling demand across the city's 112 neighbourhoods (buurten).
- **When** — summer-day and monthly profiles, switchable between cooling demand and the
  electricity drawn to meet it.
- **Impact** — the life-cycle greenhouse-gas breakdown, one stacked bar per scenario.

A plain-language summary panel states the headline finding up front, so nobody has to
read a chart to get it.

**[Open the live dashboard →](https://simonvanlierde.github.io/msc-thesis-ie/)**

![The dashboard's map view: cooling demand across The Hague's 112 neighbourhoods, shown on the 2050 Medium scenario](docs/screenshot.png)

## Where the data comes from

Every number traces to the thesis model. Three small build scripts
(`scripts/*.py`) convert the model outputs into web-friendly JSON/GeoJSON:

| Script | Output | Source | Notes |
| --- | --- | --- | --- |
| `build_scenarios.py` | `public/data/scenarios.json` (~47 kB) | `data/output/CDM_results_*.csv` (committed) | Per-archetype cooling + LCA totals for all 5 scenarios. A self-check reproduces the README headline (offices: 13% area, 34% demand, 65% GHG) from the built data. |
| `build_choropleth.py` | `public/data/cooling_by_buurt.geojson` (~127 kB) | per-building GPKG + CBS buurten (fetched by the `fetch_cbs_buurten` rule) | 59,381 buildings aggregated to 112 buurten by centroid, geometry simplified and reprojected to WGS84. Buurt sums match the archetype totals to 0.00% for the present-day (SQ) scenario. |
| `build_temporal.py` | `public/data/temporal.json` (~33 kB) | per-building GPKGs + committed weather/parameter CSVs | Re-runs the thesis heat-balance model on a stratified building sample, once per scenario (climate, UHI, comfort threshold and renovation come from the scenario parameters), averages 2021–2025 weather into a typical year and exports the hottest week hour by hour. Calibrated per use to the citywide archetype totals, so the magnitudes include projected building-stock growth. Per-building annual `E_cooling` reproduces the published value to ~0.03% median error (SQ; ~0.8% for 2050 M). |

The spatial and temporal builds need the per-building GeoPackages from the Zenodo dataset
([10.5281/zenodo.8344580](https://doi.org/10.5281/zenodo.8344580)), which are git-ignored;
the buurt boundaries come from CBS via the pipeline. The pre-built JSON in `public/data/`
is committed, so the dashboard runs and deploys without any of it.

### Caveats

- The page opens on the **2050 medium** path, so the map first shows that scenario;
  **present-day (SQ)** is one click away. Only in
  SQ do the per-building geometry and the archetype totals share the same underlying stock and
  agree exactly. Future scenarios add building-stock growth in the archetype totals that the
  current per-building geometry does not carry, so their map sums run ~10–12% below the scenario
  total — shown but labelled.
- The temporal profiles use representative building geometry (only the footprint MBR is
  reconstructed; the physics is the thesis code, unchanged). Today's stock sets each
  profile's shape; the magnitude is calibrated per use to the same citywide totals as the
  impact charts, so the sections agree. To swap in an exact hourly export from a notebook run,
  replace `temporal.json`.

## Run it locally

```bash
cd dashboard
pnpm install
pnpm dev          # http://localhost:5183
```

That's it — the committed JSON drives everything.

### Rebuild the data (optional)

Needs the repo's Python model environment and the Zenodo geodata dropped into
`data/input/geodata/` and `data/output/geodata/`:

```bash
# from the repo root, in the model venv
cd dashboard
pnpm data         # runs the three build_*.py scripts
```

`build_scenarios.py` alone has no heavy dependencies (stdlib only) and always works;
the others skip cleanly if the geodata is absent.

Each script takes `--results-dir` / `--geodata-dir` / `--divisions` / `--out` path
arguments (defaulting to the committed `data/output/` reference results), so they can
read from wherever the model writes. The Snakemake pipeline wires them in as a target:

```bash
# regenerate all three datasets from the pipeline's results/ outputs
snakemake dashboard_data
```

## Deploy

The build is a static SPA (`dist/`), published to GitHub Pages by
[`.github/workflows/dashboard.yml`](../.github/workflows/dashboard.yml) on every push to
`main` that touches `dashboard/`. Pull requests run the same quality and accessibility
gates without deploying.

Live at <https://simonvanlierde.github.io/msc-thesis-ie/>.

## Accessibility

Accessibility follows the same setup as `tide` / `credit-heatmap`:

- Keyboard-navigable controls (radio-group segmented controls, skip link, visible focus), each
  placed with the view it scopes rather than buried in a chart card.
- Every chart and the map carry an accessible label, and each view has a **data-table
  fallback** — nothing is available only as a chart or only on hover.
- A colourblind-safe palette: a single-hue blue sequential scale for the map, blue/orange
  for residential/office. Each hue has one job; magenta is reserved for keyboard focus and
  no chart uses it. Meaning never rides on colour alone — the map legend gives explicit
  numeric ranges, a value strip shows where every neighbourhood falls, and every value is
  in a table.
- Light and dark themes, both deliberately styled (not an auto-flip) and both contrast-checked.
  The theme is stamped before first paint by an inline script, so a saved override never
  flashes the other theme on load.
- `@axe-core/playwright` runs against the production build in **both themes** (`pnpm test:e2e`),
  asserting zero serious/critical WCAG 2 A/AA violations.

## Stack

Vite + React + TypeScript, [nivo](https://nivo.rocks) charts, [MapLibre GL](https://maplibre.org)
map, [Biome](https://biomejs.dev) (lint/format), Vitest (unit tests for the data transforms),
Playwright + axe (a11y). `pnpm check` also runs pre-commit, via the repo-root
[pre-commit](../.pre-commit-config.yaml) config. Vite is pinned to 7.x: Vite 8's rolldown
bundler currently breaks MapLibre's GeoJSON web worker in production builds.

Type is self-hosted: [Newsreader](https://fonts.google.com/specimen/Newsreader) (variable, for the
wordmark, headings and the hero headline) and [Public Sans](https://public-sans.digital.gov)
(variable, body and every figure — a serif on a stat-tile value reads as decoration), both subset
to `latin` in `public/fonts/` — no font CDN, no external request. The map draws
over [CARTO](https://carto.com)'s positron / dark-matter basemap (streets and place names, no API
key); if that CDN is unreachable it falls back to a plain background, so the choropleth still works
offline. CARTO/OpenStreetMap attribution shows on the map when the basemap loads.

```bash
pnpm check        # typecheck + lint + coverage + build
pnpm test         # unit tests (data transforms)
pnpm test:e2e     # accessibility, light + dark
pnpm og           # rebuild the social card + favicon (see note below)
```

The social-preview card (`public/og.png`), the favicon (`public/icon.svg`, the city silhouette)
and the apple-touch icon are generated by `scripts/build_og.mjs` from the thesis output itself —
the headline stat, the year's heat band, the buurt geometry. It renders with the Playwright
chromium already installed for the a11y tests, so it adds no dependency, and skips cleanly when
its inputs are absent. The band's 12 × 24 grid is reconstructed from `temporal.json`: each month
takes its season's diurnal shape, rescaled to that month's energy total. Only four seasonal
profiles exist, so months inside a season differ in magnitude but not in shape — invisible at
168 px, and it keeps the card on the same data the dashboard itself reads.
