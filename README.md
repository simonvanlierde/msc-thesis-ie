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

Then download the spatial datasets from Zenodo into `data/input/geodata/` before running
`gis.ipynb`.

### Regenerating the headline figure

```bash
uv run python docs/make_overview_figure.py
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
