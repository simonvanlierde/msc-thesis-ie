# Cooling for Comfort, Warming the World

**Residential and office cooling and its environmental impacts in The Hague, the Netherlands.**

[![CI](https://github.com/simonvanlierde/msc-thesis-ie/actions/workflows/ci.yml/badge.svg)](https://github.com/simonvanlierde/msc-thesis-ie/actions/workflows/ci.yml)
[![Dashboard](https://github.com/simonvanlierde/msc-thesis-ie/actions/workflows/dashboard.yml/badge.svg)](https://github.com/simonvanlierde/msc-thesis-ie/actions/workflows/dashboard.yml)
[![codecov](https://codecov.io/gh/simonvanlierde/msc-thesis-ie/branch/main/graph/badge.svg)](https://codecov.io/gh/simonvanlierde/msc-thesis-ie)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/simonvanlierde/msc-thesis-ie?utm_source=badge)
[![Dashboard](https://img.shields.io/badge/dashboard-live-brightgreen.svg)](https://simonvanlierde.github.io/msc-thesis-ie/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Data DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.8344580-blue.svg)](https://doi.org/10.5281/zenodo.8344580)
[![Thesis](https://img.shields.io/badge/thesis-TU%20Delft%20repository-blue.svg)](https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a)

This is the model behind my MSc Industrial Ecology thesis (Leiden University & TU Delft). It
estimates how much cooling the buildings of The Hague need, and what that cooling costs in
electricity, greenhouse-gas emissions and materials, today and in scenarios for 2030 and 2050.

**[Open the live dashboard →](https://simonvanlierde.github.io/msc-thesis-ie/)**

[![Dashboard](dashboard/docs/screenshot.png)](https://simonvanlierde.github.io/msc-thesis-ie/)

## Key findings

These are the results of the current model, and the dashboard shows the same ones. The model uses
weather from 2021–2025 with the urban heat island added, and the carbon intensity of the 2025
electricity grid. `scripts/paper_numbers.py` prints these numbers from `results/`. The 2023 thesis
used older weather and grid data, so its numbers differ.

![Cooling demand and emissions across scenarios](docs/scenario_overview.png)

- The Hague needs about **527 GWh** of cooling a year. Offices have only **11%** of the floor
  area, but they use **54%** of that cooling and cause **79%** of the greenhouse-gas emissions
  from cooling.
- An estimated **54%** of the cooling demand is not met today. The gap is worst in the
  lower-income neighbourhoods, where the heat island is strongest.
- By 2050, cooling demand changes by **−7%** in the low scenario and by **+79%** in the high
  one, compared with today (**+24%** in the middle one). Emissions fall in every 2050 scenario
  except the high one, where they rise by **76%**.

## How the model works

The model sorts the buildings in the Dutch BAG register into archetypes, runs an hourly
heat-balance model on each one with KNMI weather, and estimates the life-cycle impacts on climate
change, abiotic resource depletion and crustal scarcity, from both the electricity the cooling
uses and the cooling equipment itself. The
[thesis](https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a) has the
full method and discussion.

## Quick start

The large spatial input datasets are on Zenodo
([10.5281/zenodo.8344580](https://doi.org/10.5281/zenodo.8344580)), not in this repository. The
Snakemake pipeline can also fetch them from the official PDOK APIs (the Dutch government's
geodata service).

```bash
git clone https://github.com/simonvanlierde/msc-thesis-ie.git
cd msc-thesis-ie
uv sync                                                    # locked environment from uv.lock
uv run snakemake --configfile config/smoke.yaml --cores 4  # fast end-to-end check (minutes)
uv run snakemake --cores 4                                 # scenario results + overview figure
uv run --with jupyterlab jupyter lab                       # open main.ipynb and gis.ipynb
```

- [docs/pipeline.md](docs/pipeline.md): the pipeline, its opt-in targets, the container, the city
  and weather settings, and the EP-Online key.
- [docs/development.md](docs/development.md): repository layout, checks and git hooks.
- [dashboard/README.md](dashboard/README.md): the dashboard's data, accessibility and deploy.

## Citation

If you use this work, please cite the thesis and the dataset (see [`CITATION.cff`](CITATION.cff)):

> van Lierde, S. (2024). *Cooling for Comfort, Warming the World: Residential and Office Cooling and
> its Environmental Implications in The Hague.* MSc thesis, Industrial Ecology, Leiden University &
> TU Delft.

Supervised by Prof. ir. Peter Luscuere and Dr. Benjamin Sprecher.

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
