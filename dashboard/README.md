# Cooling for Comfort — dashboard

An interactive dashboard for the MSc thesis model of building cooling demand and its
life-cycle environmental impact in The Hague. It turns the thesis result files into three
views for a mixed audience (researchers, policymakers, general public):

- **Where** — a choropleth of cooling demand across the city's 112 neighbourhoods (buurten).
- **Year** — a scrubbable hour-by-hour time-lapse: watch the city flush from pale (winter nights)
  to deep red (summer afternoons).
- **When** — diurnal and seasonal cooling-demand profiles.
- **Impact** — the life-cycle greenhouse-gas breakdown and how impacts split across building use.

A plain-language summary panel states the headline finding up front, so the point lands
without reading a chart.

<!-- Live demo: add the Cloudflare Pages URL here once deployed. -->

![The dashboard](docs/screenshot.png)

## The data is real thesis output

Every number traces to the thesis model — nothing is invented. Three small build scripts
(`scripts/*.py`) convert the model outputs into web-friendly JSON/GeoJSON:

| Script | Output | Source | Notes |
| --- | --- | --- | --- |
| `build_scenarios.py` | `public/data/scenarios.json` (~47 kB) | `data/output/CDM_results_*.csv` (committed) | Per-archetype cooling + LCA totals for all 5 scenarios. A self-check reproduces the README headline (offices: 13% area, 34% demand, 65% GHG) from the built data. |
| `build_choropleth.py` | `public/data/cooling_by_buurt.geojson` (~127 kB) | per-building GPKG + CBS buurten (fetched by the `fetch_cbs_buurten` rule) | 59,381 buildings aggregated to 112 buurten by centroid, geometry simplified and reprojected to WGS84. Buurt sums match the archetype totals to 0.00% for the present-day (SQ) scenario. |
| `build_temporal.py` | `public/data/temporal.json` (~3 kB) | per-building GPKG + committed weather/parameter CSVs | Re-runs the thesis heat-balance model on a stratified building sample, averages 2018–2022 weather into a typical year, calibrated to the published annual totals. Validated: per-building annual `E_cooling` reproduces the published value to **~0.03% median error**. |

The spatial and temporal builds need the per-building GeoPackages from the Zenodo dataset
([10.5281/zenodo.8344580](https://doi.org/10.5281/zenodo.8344580)), which are git-ignored;
the buurt boundaries come from CBS via the pipeline. The pre-built JSON in `public/data/`
is committed, so the dashboard runs and deploys without any of it.

### Caveats (kept honest)

- The page opens on the **2050 medium** path (the "choose your 2050" narrative pre-selects a
  future), so the map first shows that scenario; **present-day (SQ)** is one click away. Only in
  SQ do the per-building geometry and the archetype totals share the same underlying stock and
  agree exactly. Future scenarios add building-stock growth in the archetype totals that the
  current per-building geometry does not carry, so their map sums run ~10–12% below the scenario
  total — shown but labelled.
- The temporal profile uses representative building geometry (only the footprint MBR is
  reconstructed; the physics is the thesis code, unchanged) calibrated to the published
  totals. To swap in an exact hourly export from a notebook run, replace `temporal.json`.

## Run it locally

```bash
cd dashboard
pnpm install
pnpm dev          # http://localhost:5173
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

## Deploy to Cloudflare Pages

The build is a static SPA (`dist/`), same pattern as the `tide` repo (`wrangler.jsonc` →
`pages_build_output_dir`).

**Git integration (recommended — gives free PR previews):** in the Cloudflare dashboard,
connect the repo as a Pages project with:

- **Root directory:** `dashboard`
- **Build command:** `pnpm build`
- **Build output directory:** `dist`

Every pull request then gets an automatic preview deployment; merges to `main` publish to
production.

**Or from the CLI:**

```bash
pnpm build
pnpm dlx wrangler pages deploy dist
```

## Accessibility

Accessibility is a first-class requirement here, matching the setup in `tide` /
`credit-heatmap`:

- Keyboard-navigable controls (radio-group segmented controls, skip link, visible focus), gathered
  into a single row above the views they scope rather than buried in each chart card.
- Every chart and the map carry an accessible label, and each view has a **data-table
  fallback** — nothing is available only as a chart or only on hover. The year time-lapse
  gets two: a per-month summary and the full 12 × 24 matrix the carpet colours, in a
  focusable scroll region so a keyboard can pan it.
- Text set inside a coloured fill (the stacked bar's share labels) picks black or white by the
  fill's luminance, so it always clears 4.5:1 — asserted in `src/lib/palette.test.ts`, since
  axe cannot see SVG text.
- A colourblind-safe palette (validated data-viz reference palette): a single-hue blue
  sequential scale for the map, blue/orange for residential/office, a warm ramp for the
  year's heat. Colour has one job per hue — magenta is reserved for keyboard focus and used
  by no chart; warm saturated colour only ever means the data is hot. Meaning is never
  colour-alone — the map legend gives explicit numeric ranges, a value strip shows where
  every neighbourhood falls, and every value is in a table.
- Light and dark themes, both deliberately styled (not an auto-flip) and both contrast-checked.
  The theme is stamped before first paint by a tiny inline script, so a saved override never
  flashes the other theme on load.
- `@axe-core/playwright` runs against the production build in **both themes** (`pnpm test:e2e`),
  asserting zero serious/critical WCAG 2 A/AA violations.

## Stack

Vite + React + TypeScript, [nivo](https://nivo.rocks) charts, [MapLibre GL](https://maplibre.org)
map, [Biome](https://biomejs.dev) (lint/format), Vitest (unit tests for the data transforms),
Playwright + axe (a11y), lefthook (git hooks). Vite is pinned to 7.x: Vite 8's rolldown
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
