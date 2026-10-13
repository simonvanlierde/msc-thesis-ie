# Development

Install with `uv sync`. Without uv, `pip install .` in a virtual environment also works.

| Path | Description |
| --- | --- |
| `cdm/` | The model package. Each module's docstring says what it covers. |
| `main.ipynb` | End-to-end notebook: cooling demand, environmental impacts, sensitivity analysis, result figures. |
| `gis.ipynb` | Preparation of the BAG geospatial data and spatial visualisations. |
| `Snakefile`, `config/` | The reproducible pipeline and its city / weather configuration ([pipeline.md](pipeline.md)). |
| `data/input/parameters/` | All model input parameters, organised per scenario. |
| `data/output/` | Committed reference results per building type and energy label. |
| `dashboard/` | The interactive web dashboard ([its README](../dashboard/README.md)). |
| `scripts/` | Data acquisition and pipeline entry points. |
| `docs/` | The headline figure and the script that regenerates it. |
| `tests/` | Unit tests for the geometric, thermodynamic and environmental functions. |

## Checks

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run ty check                # type check (informational)
uv run pytest                  # tests
uv run pre-commit install      # git hooks
```

CI runs the same checks, plus a Snakemake dry run, on every pull request
([workflow](../.github/workflows/ci.yml)). On
commit, the hooks run ruff (lint and format), the dashboard formatter and `ty`. On push, they run
the test suite and the dashboard checks. `ty` reports its findings without blocking, as in CI.

## Regenerating the figures

```bash
uv run python docs/make_overview_figure.py

# Reproduces the committed SVG (no Graphviz binary required)
uv run snakemake --dag | uv run python scripts/dot_to_simple_svg.py > docs/pipeline_dag.svg
```
