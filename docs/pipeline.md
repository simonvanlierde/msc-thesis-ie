# Reproducible pipeline

The analysis in the notebooks is also a [Snakemake](https://snakemake.readthedocs.io/) workflow of
27 rules. It calls the same model code in `cdm/` and does not reimplement the science.

![Snakemake workflow DAG](pipeline_dag.svg)

```bash
uv sync                                                  # once
uv run snakemake --cores 4                               # scenario results + overview figure
uv run snakemake --configfile config/smoke.yaml --cores 4  # fast end-to-end check (minutes), under results/smoke/
```

Or run it in a container, so nothing depends on your own setup:

```bash
docker build -t cooling-demand .
docker run --rm -e EP_ONLINE_API_KEY -v "$PWD:/work" cooling-demand snakemake --cores 4
```

The pipeline writes everything it makes under `results/` (`results_dir` in `config/sources.yaml`
changes that). The committed reference results in `data/output/` and `docs/` stay untouched, so
you can repeat a run safely and compare its output with the reference directly:

```bash
diff results/CDM_results_SQ_full.csv data/output/CDM_results_SQ_full.csv
```

Git ignores `results/` and the fetched inputs in `data/raw/`. Each rule writes a log under
`results/logs/`, and the model stages record their run time under `results/benchmarks/`.

## Opt-in targets

These targets are heavy, so I kept them out of the default `all` target:

```bash
# Journal-paper figures (F1, F2, F5) and the manuscript numbers (results/paper_numbers.json).
uv run snakemake --cores 4 paper

# One-at-a-time sensitivity analysis of the SQ scenario: 21 steps per parameter, single-threaded
# (about 20 hours on the full stock). Set --config sa_steps=N for a coarser sweep.
uv run snakemake --cores 4 sensitivity

# Rebuild the dashboard data in dashboard/public/data from the scenario results.
uv run snakemake --cores 4 dashboard_data

# Cooling-technology-mix sensitivity: 30 ordered technology pairs × a 20-step mix sweep,
# each running the full model over the building stock. Lower --calculation-steps for a
# faster, coarser table.
uv run snakemake --cores 4 cooling_mix

# Clip the RIVM urban-heat-island raster to the city (downloads a ~1.95 GB national ZIP once).
uv run snakemake --cores 4 data/input/geodata/UHI_effect_TheHague.tif

# Execute the analysis notebooks headless, keeping the executed copies as artifacts. gis needs
# the UHI raster above; main needs a sample subset or ample RAM (its figures keep the full hourly
# time series in memory).
uv run snakemake --cores 4 notebooks
```

## City and weather window

Both are declared in `config/sources.yaml`:

```yaml
city:
  name: "'s-Gravenhage" # official municipality name (fetched from PDOK)
  weather_station: 330 # nearest KNMI station
weather:
  start_year: 2021
  end_year: 2025
```

The pipeline fetches the municipal boundary and its bounding box from PDOK by name, and clips the
buildings to that boundary, so the results cover exactly that municipality. To switch cities,
change the name and pick the nearest KNMI station, or override the settings without editing the
file:

```bash
uv run snakemake --config city='{name: Rotterdam, weather_station: 344}' --cores 4
```

## EP-Online energy labels

`provide_ep_online_energy_labels` downloads the full public export of energy labels through the
EP-Online v5 API. Request a key at <https://www.ep-online.nl/PublicData>. Then put it in a `.env`
file at the repo root:

```bash
EP_ONLINE_API_KEY=your-key-here
```

The rule extracts the CSV to `data/raw/ep_online/current/energy_labels.csv` (~1.5 GB
uncompressed), and skips the download when a valid file is already there. It reads the key from
the environment or from `.env`, and the key is never committed.

## Stages

| Rule | Existing analysis step | Main inputs | Main outputs |
| --- | --- | --- | --- |
| `fetch_city_boundary` | Municipal extent by name | PDOK Bestuurlijke gebieden OGC API `gemeentegebied` | city boundary GeoJSON + CRS84 bbox |
| `fetch_bag_residences` | Official BAG residence/use acquisition | PDOK BAG OGC API `verblijfsobject`, city bbox | `data/raw/pdok_bag/verblijfsobject_{city}.gpkg` |
| `discover_pdok_3d_height_tiles` | Height-tile discovery | PDOK 3D Basisvoorziening OGC API `hoogtestatistieken_gebouwen`, city bbox | height-tile manifest |
| `download_pdok_3d_height_tiles` | Official height geodata acquisition | PDOK 3D Basisvoorziening tile download links | local GeoPackage ZIP tiles and manifest |
| `provide_ep_online_energy_labels` | Credentialed energy-label source boundary | EP-Online public export | `data/raw/ep_online/current/energy_labels.csv` |
| `fetch_weather` | KNMI weather-series retrieval | KNMI hourly API (station + year window), committed backup fallback | `results/weather/knmi_{station}_{start}_{end}.csv` |
| `prepare_bag_geodata` | Scripted replacement for the BAG/geodata joins in `gis.ipynb` | PDOK BAG residences, PDOK 3D height tiles, EP-Online labels, city boundary (clip) | `results/geodata/BAG_buildings_with_residence_data_nouhi_full.gpkg` |
| `fetch_uhi_habib` | Fetch + extract the Habib et al. UHImax raster for the city | 4TU national UHImax zip (cached, ~4.5 GB one-time) | `data/input/geodata/UHImax_habib_TheHague_5m.tif` |
| `add_uhi_to_buildings` | Vectorized per-building neighbourhood-mean UHImax join | BAG geodata (no-UHI), Habib UHImax raster | `results/geodata/BAG_buildings_with_residence_data_full.gpkg` |
| `thermodynamic_model` | cooling-demand model from `main.ipynb` | processed BAG geodata, `parameters.toml` (per scenario), weather CSV, load-factor inputs | `results/intermediate/buildings_with_cooling_demand_{scenario}_full.gpkg` |
| `lca` | environmental-impact and aggregation steps from `main.ipynb` | cooling-demand geodata and scenario parameters | `results/CDM_results_{scenario}_full.csv` and CDM geodata |
| `scenario_overview_figure` | README headline figure script | scenario result CSVs | `results/figures/scenario_overview.png` |
| `cooling_mix_sensitivity` (opt-in) | cooling-technology-mix sensitivity cells in `main.ipynb`, extracted to `scripts/run_cooling_mix_sensitivity.py` | processed BAG geodata, SQ parameters, weather CSV | `results/cooling_mix_elasticities_table.csv` |
| `fetch_uhi_raster` (opt-in) | RIVM urban-heat-island raster, previously a manual download | national UHI ZIP (RIVM), city bbox | `data/input/geodata/UHI_effect_TheHague.tif` (clipped) |
| `run_main_notebook` / `run_gis_notebook` (opt-in) | execute the analysis notebooks headless; keep the executed copy | pipeline outputs the notebook reads, `cdm/`, the notebook | `results/notebooks/{main,gis}.executed.ipynb` |

PDOK data is licensed under Public Domain Mark 1.0 (BAG, boundaries) and CC BY 4.0 (3D
Basisvoorziening). I checked the column names of the PDOK 3D height data against a 2025
`hoogtestatistieken_gebouwen` tile (`identificatie`, `status`, `oorspronkelijkbouwjaar`,
`rf_h_ground`, `rf_h_roof_70p`). If a later release changes these columns,
`scripts/gis/prepare_pdok_model_geodata.py` stops with an error that lists the columns it found. It
does not guess.
