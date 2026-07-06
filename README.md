# Cooling for Comfort, Warming the World

**Residential and office cooling and its environmental impacts in The Hague, the Netherlands.**

[![CI](https://github.com/simonvanlierde/msc-thesis-ie/actions/workflows/ci.yml/badge.svg)](https://github.com/simonvanlierde/msc-thesis-ie/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Data DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.8344580-blue.svg)](https://doi.org/10.5281/zenodo.8344580)
[![Thesis](https://img.shields.io/badge/thesis-TU%20Delft%20repository-blue.svg)](https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a)

This repository contains the model behind my MSc Industrial Ecology thesis (joint degree of
Leiden University and TU Delft). It estimates how much cooling the building stock of The Hague
needs, and what that cooling costs in electricity, greenhouse-gas emissions and material use,
under current conditions and across future scenarios for 2030 and 2050.

The pipeline combines three layers:

1. **Geospatial data** — building footprints and attributes from the Dutch BAG (*Basisregistratie
   Adressen en Gebouwen*), processed into building archetypes.
2. **Thermodynamic modelling** — an hourly heat-balance model (transmission, infiltration,
   ventilation, solar gains and internal loads) driven by KNMI weather data, giving cooling
   energy and peak power demand per building.
3. **Environmental impact assessment** — life-cycle-based impacts (climate change, abiotic
   resource depletion, crustal scarcity) from both the operational energy and the cooling
   equipment itself.

## Key findings

![Cooling demand and emissions across scenarios](docs/scenario_overview.png)

- Offices occupy only **13%** of the floor area but account for **34%** of current cooling
  demand and **65%** of cooling-related greenhouse-gas emissions.
- Cooling already represents about **25%** of office electricity use, against **5.5%** for
  residential buildings.
- An estimated **77%** of cooling demand is currently unmet, and that gap falls hardest on
  economically disadvantaged neighbourhoods.
- Under a business-as-usual 2050 scenario, cooling energy demand roughly **doubles** relative
  to today, putting pressure on the Netherlands' 2050 net-zero target.

The full method and discussion are in the
[thesis](https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a).

## Interactive dashboard

An interactive web dashboard presents these results — a choropleth of cooling demand across
The Hague, the diurnal/seasonal demand profile, and the life-cycle impact breakdown, with a
plain-language summary for non-experts. It runs entirely on real model output.

<!-- Live demo: add the Cloudflare Pages URL here once deployed. -->

![Dashboard](dashboard/docs/screenshot.png)

```bash
cd dashboard && pnpm install && pnpm dev
```

See [`dashboard/README.md`](dashboard/README.md) for the data-build steps, accessibility
notes and the Cloudflare Pages deployment.

## Repository structure

| Path | Description |
| --- | --- |
| `functions/` | The model, split into focused modules (see below). |
| `main.ipynb` | End-to-end notebook: cooling demand, environmental impacts, sensitivity analysis and result figures. |
| `gis.ipynb` | Preparation of the BAG geospatial data and spatial visualisations. |
| `data/input/parameters/` | All model input parameters, organised per scenario. |
| `data/output/` | Aggregated model results per building type and energy label. |
| `docs/` | The headline figure and the script that regenerates it. |
| `tests/` | Unit tests for the geometric, thermodynamic and environmental functions. |
| `dashboard/` | Interactive web dashboard communicating the results (see below). |

Inside `functions/`:

- `data_handling.py` — reading, joining and preparing the building data.
- `geometric.py` — building geometry: facade orientation, window and wall areas.
- `thermodynamic.py` — the hourly heat-balance and cooling-demand calculations.
- `environmental.py` — life-cycle environmental impacts of cooling.
- `time_series.py` — weather data retrieval (KNMI) and time-series construction.
- `sensitivity_analysis.py` — scenario and one-at-a-time sensitivity analysis.
- `figures.py` — plotting helpers for the result figures.

## Getting started

The large spatial input datasets are hosted separately on Zenodo
([10.5281/zenodo.8344580](https://doi.org/10.5281/zenodo.8344580)) and are not part of this
repository.

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
git clone https://github.com/simonvanlierde/msc-thesis-ie.git
cd msc-thesis-ie
uv sync                 # creates a locked environment from uv.lock
uv run jupyter lab      # open main.ipynb and gis.ipynb
```

### With pip

```bash
git clone https://github.com/simonvanlierde/msc-thesis-ie.git
cd msc-thesis-ie
python -m venv .venv && source .venv/bin/activate
pip install .
jupyter lab
```

The Snakemake workflow below acquires the spatial inputs from official PDOK APIs and
the EP-Online energy-label export (the latter needs a free API key, see below).

### Regenerating the headline figure

```bash
uv run python docs/make_overview_figure.py
```

## Reproducible Snakemake Pipeline

The existing notebook analysis is also declared as a Snakemake workflow. The
workflow wraps the same `functions/` model code and notebook outputs; it does
not replace the scientific calculations with new implementations.

![Snakemake workflow DAG](docs/pipeline_dag.svg)

Create the runner environment once. The workflow still supports `uv` for normal
development, but the Snakemake runner uses conda-forge packages because the GIS
stack depends on native GDAL/PROJ/GEOS libraries.

```bash
conda env create -f workflow/envs/cooling-demand.yml
conda activate cooling-demand-model
```

Then reproduce the declared outputs (scenario results + overview figure) with one command:

```bash
snakemake --sdm conda --cores 4
```

For a fast end-to-end check, run the smoke profile (a tiny municipality, one
weather year — finishes in minutes):

```bash
snakemake --configfile config/smoke.yaml --cores 4
```

The heavy cooling-technology-mix sensitivity is **opt-in** (30 tech pairs × a
20-step model sweep), kept out of the default target:

```bash
snakemake --cores 4 cooling_mix
```

Alternatively, a hermetic container bundles the pinned environment:

```bash
docker build -t cooling-demand .
docker run --rm -e EP_ONLINE_API_KEY -v "$PWD:/work" cooling-demand snakemake --cores 4
```

The pipeline stages are:

| Rule | Existing analysis step | Main inputs | Main outputs |
| --- | --- | --- | --- |
| `fetch_city_boundary` | Municipal extent by name | PDOK Bestuurlijke gebieden OGC API `gemeentegebied` | city boundary GeoJSON + CRS84 bbox |
| `fetch_bag_residences` | Official BAG residence/use acquisition | PDOK BAG OGC API `verblijfsobject`, city bbox | `data/raw/pdok_bag/verblijfsobject_{city}.geojson` |
| `discover_pdok_3d_height_tiles` | Height-tile discovery | PDOK 3D Basisvoorziening OGC API `hoogtestatistieken_gebouwen`, city bbox | height-tile manifest |
| `download_pdok_3d_height_tiles` | Official height geodata acquisition | PDOK 3D Basisvoorziening tile download links | local GeoPackage ZIP tiles and manifest |
| `provide_ep_online_energy_labels` | Credentialed energy-label source boundary | EP-Online public export | `data/raw/ep_online/current/energy_labels.csv` |
| `fetch_weather` | KNMI weather-series retrieval | KNMI hourly API (station + year window), committed backup fallback | `results/weather/knmi_{station}_{start}_{end}.csv` |
| `prepare_bag_geodata` | Scripted replacement for the BAG/geodata joins in `gis.ipynb` | PDOK BAG residences, PDOK 3D height tiles, EP-Online labels, city boundary (clip) | `results/geodata/BAG_buildings_with_residence_data_full.gpkg` |
| `thermodynamic_model` | cooling-demand model from `main.ipynb` | processed BAG geodata, `parameters.toml` (per scenario), weather CSV, load-factor inputs | `results/intermediate/buildings_with_cooling_demand_{scenario}_full.gpkg` |
| `lca` | environmental-impact and aggregation steps from `main.ipynb` | cooling-demand geodata and scenario parameters | `results/CDM_results_{scenario}_full.csv` and CDM geodata |
| `scenario_overview_figure` | README headline figure script | scenario result CSVs | `results/figures/scenario_overview.png` |
| `cooling_mix_sensitivity` (opt-in) | cooling-technology-mix sensitivity cells in `main.ipynb`, extracted to `scripts/run_cooling_mix_sensitivity.py` | processed BAG geodata, SQ parameters, weather CSV | `results/cooling_mix_elasticities_table.csv` |

Each rule writes a log under `results/logs/` and the model stages a runtime
benchmark under `results/benchmarks/`. Large raw spatial inputs are not
committed. PDOK data is licensed under Public Domain Mark 1.0 (BAG, boundaries)
and CC BY 4.0 (3D Basisvoorziening).

### Configuring the city and weather window

The active city and weather window are declared in `config/sources.yaml`:

```yaml
city:
  name: "'s-Gravenhage" # official municipality name (fetched from PDOK)
  weather_station: 330 # nearest KNMI station
weather:
  start_year: 2018
  end_year: 2022
```

The municipal boundary and bounding box are fetched from PDOK by name, so
switching cities is just a name change (plus the nearest KNMI station). Buildings
are clipped to the boundary, so results cover exactly that municipality. Override
without editing the file, e.g. `snakemake --config city='{name: Rotterdam, weather_station: 344}' --cores 4`.

### Where outputs go

The pipeline writes everything it generates under `results/` (configurable via
`results_dir` in `config/sources.yaml`), leaving the committed reference results
in `data/output/` and `docs/` untouched. This makes a pipeline run safe to
repeat and lets you check reproduction directly:

```bash
diff results/CDM_results_SQ_full.csv data/output/CDM_results_SQ_full.csv
```

`results/` and the fetched `data/raw/` inputs are git-ignored. To refresh the
committed README figure from the reference results, run the figure script with
its defaults: `uv run python docs/make_overview_figure.py`.

### EP-Online energy labels

`provide_ep_online_energy_labels` downloads the full public label export via the
EP-Online v5 API. Put your key in a `.env` file at the repo root:

```bash
EP_ONLINE_API_KEY=your-key-here
```

The rule resolves the signed download URL, streams the ZIP, and extracts the CSV
to `data/raw/ep_online/current/energy_labels.csv` (~1.5 GB uncompressed). It
skips the download when a valid file is already present. The key is read from the
environment or `.env` and is never committed. Request a key at
<https://www.ep-online.nl/PublicData>.

Notes:

- PDOK 3D height-attribute column names were verified against a 2025
  `hoogtestatistieken_gebouwen` tile (`identificatie`, `status`,
  `oorspronkelijkbouwjaar`, `rf_h_ground`, `rf_h_roof_70p`). If a future
  vintage changes the schema, `scripts/gis/prepare_pdok_model_geodata.py` fails
  loudly listing the available columns instead of guessing.
- The cooling-technology-mix sensitivity is a heavy stage: 30 ordered
  technology pairs, each a 20-step mix sweep running the full model over the
  building stock. Lower `--calculation-steps` in the rule for a faster,
  coarser table.

To refresh the committed DAG after editing `Snakefile`, run:

```bash
# Reproduces the committed SVG (no Graphviz binary required)
snakemake --dag | python scripts/dot_to_simple_svg.py > docs/pipeline_dag.svg

# Alternative, if Graphviz `dot` is installed (styled differently)
snakemake --dag | dot -Tsvg > docs/pipeline_dag.svg
```

## Development

The project uses the [Astral](https://astral.sh/) toolchain:

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run ty check                # type check (informational)
uv run pytest                  # run the test suite
```

The same checks run in CI on every push and pull request
([workflow](.github/workflows/ci.yml)).

## Citation

If you use this work, please cite the thesis and dataset (see
[`CITATION.cff`](CITATION.cff)):

> van Lierde, S. (2024). *Cooling for Comfort, Warming the World: Residential and Office Cooling
> and its Environmental Implications in The Hague.* MSc thesis, Industrial Ecology, Leiden
> University & TU Delft.

## Acknowledgements

Supervised by Prof. ir. Peter Luscuere and Dr. Benjamin Sprecher.

## License

Released under the [GNU General Public License v3.0](LICENSE).
