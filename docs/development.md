# Development

Python 3.15 is required. Install with `uv sync --locked`.
Run `pnpm -C dashboard install --frozen-lockfile` for the dashboard.
`just check` runs the Python and dashboard checks.

The locked `immutables` dependency builds from source on Python 3.15.
Local installation needs a C compiler and C library headers.
The Dockerfile installs these build dependencies and removes them after installation.

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
| `benchmarks/` | CodSpeed performance benchmarks of the pipeline stages. |

## Checks

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
uv run ty check                # type check
uv run pytest                  # tests
uv run prek install            # git hooks
```

CI and `just check` install the optional `dataset` group so the release allowlist and archive tests run too.
To run those checks locally, use `uv run --group dataset pytest tests/test_dataset_release.py`.
The committed building fixture uses real building inputs with synthetic energy labels;
[the fixture notes](../tests/fixtures/README.md) describe its source and limits.

CI runs the same checks and a Snakemake dry run on Python 3.15 for every pull request
([workflow](../.github/workflows/ci.yml)). On
commit, the hooks run ruff (lint and format), the dashboard formatter and `ty`. On push, they run
the test suite and the dashboard checks. Type errors fail both the hook and CI.
Coverage includes branches. Unexpected warnings fail tests;
the one documented filter covers a deprecated operator inside Rasterio 1.5.

## Sampling profiles

Python 3.15 includes a [sampling profiler](https://docs.python.org/3.15/library/profiling.sampling.html).
Create the output directory before you run it:

```bash
mkdir -p results
```

Sample the model-stage benchmarks:

```bash
uv run python -m profiling.sampling run --flamegraph -o results/profile.html -m pytest benchmarks/ -q
```

Replace the benchmark invocation with the workload to measure. Keep profiles under the ignored `results/` directory.
The runtime upgrade alone does not establish a speed improvement for this model.

## Benchmarks

`benchmarks/` holds [pytest-codspeed](https://codspeed.io/docs/reference/pytest-codspeed) benchmarks
for the pipeline stages: parameter assignment, time series, heat balance, impacts and aggregation.
They run the real SQ parameters and one year of the committed backup weather on a 200-building
synthetic stock, so they need no Zenodo data. Without `--codspeed` they run once each as ordinary tests.

```bash
uv run pytest benchmarks/                                                     # run as tests
codspeed run --mode simulation -- uv run pytest benchmarks/ --codspeed        # measure locally
```

The [CodSpeed workflow](../.github/workflows/codspeed.yml) measures them with CPU simulation on every
push to `main` and every pull request that touches the Python code or its inputs, and reports
performance changes on the PR.

For real-stock sensitivity profiles and peak-memory measurements, use the commands in
[performance.md](performance.md). These complement the synthetic CI benchmarks.

## Regenerating the figures

```bash
uv run python docs/make_overview_figure.py

# Reproduces the committed SVG (no Graphviz binary required)
uv run snakemake --dag | uv run python scripts/dot_to_simple_svg.py > docs/pipeline_dag.svg
```
