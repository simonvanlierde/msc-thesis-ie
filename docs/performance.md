# Profile sensitivity runs and full-stock memory

Run these commands from the repository root with Python 3.14 and the locked dependencies.
The profiler runs the production sensitivity drivers without plotting or retaining scientific outputs.
It uses SQ parameters and the committed 2022 weather backup, with no KNMI request.

Keep profiles, reports and generated packages under ignored `results/`.
The source GeoPackage contains energy labels whose redistribution permission is unresolved.
The packaging command creates a private copy of the open-input subset; it does not publish it.

## Reproduce a measurement

On Linux with a user systemd session, cap each run at 8 GiB with no swap.
Set the three thread variables before importing NumPy.
Use a fresh output name for every run: existing profiles and release folders are refused.

```sh
mkdir -p results/profiling
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout 600 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONHASHSEED=0 PYTHONFAULTHANDLER=1 \
  uv run --locked --group dataset python -m scripts.profile_sensitivity \
  --source results/geodata/BAG_buildings_with_residence_data_full.gpkg \
  --profile results/profiling/threshold-200.prof \
  sensitivity --sample 200 --steps 3 > results/profiling/threshold-200.json
```

Omit `--sample 200` to run every building selected by `read_buildings`.
Use `--variable SEER` for the driver that calculates cooling demand once, then varies impacts.
The default `T_thresh_C` sweep recalculates cooling demand at each step.
Both sweeps use three points by default. A seed of 42 fixes the sample.

To measure full release packaging, replace the final command arguments with:

```sh
uv run --locked --group dataset python -m scripts.profile_sensitivity \
  --source results/geodata/BAG_buildings_with_residence_data_full.gpkg \
  --profile results/profiling/package-full.prof \
  package --output results/profiling/dataset-full > results/profiling/package-full.json
```

Apply the same systemd, timeout and environment prefix to that command.
Packaging reads every source row before excluding invalid geometries.
For a small packaging control, save the committed 2,000-row fixture under layer `BAG_buildings_full`:

```sh
uv run --locked python - <<'PY'
import geopandas as gpd
stock = gpd.read_file("tests/fixtures/bag_2000.gpkg", layer="buildings")
stock.to_file("results/profiling/source-2000.gpkg", layer="BAG_buildings_full", driver="GPKG")
PY
```

Read CPU costs from the profile:

```sh
uv run --locked python - <<'PY'
import pstats
pstats.Stats("results/profiling/threshold-200.prof").strip_dirs().sort_stats("cumulative").print_stats(20)
PY
```

Each successful run prints one JSON object with elapsed seconds, peak resident memory in MiB,
workload size, Python and dependency versions, JIT state, thread settings, and source/lock checksums.
The timer includes input reads and profiling overhead but excludes imports and provenance checksums.
Peak memory covers the entire fresh process, including imports and checksums.
Progress bars are disabled so their background thread does not distort the call profile.
Exit codes are 0 for success, 1 for invalid scientific inputs, 2 for command usage, and 3 for execution failures.

## Diagnostic result

The full source has 245,655 rows. The production filter selects 64,393 labelled, in-use residential
and office buildings. These counts describe different workloads; full-stock cooling demand uses
all 64,393 eligible buildings, while packaging reads all 245,655 source rows.

The first diagnostic runs on a shared Linux development machine used CPython 3.14.4 with JIT off,
NumPy 2.5.3, pandas 3.0.6, GeoPandas 1.1.4 and Shapely 2.1.2. The CPU was an AMD Ryzen 7 H 255.
Each BLAS/OpenMP limit was one thread. Processes were fresh and the filesystem cache was warm.
Processes were not pinned to CPUs. Each path was measured in three fresh processes.
The table gives the median, observed range and highest peak memory. These observations locate costs;
they are not a before/after benchmark or a promise of run time.

| Path | Workload | Median seconds (range) | Peak memory |
| --- | --- | --- | --- |
| Threshold SA | 200 buildings, 8,760 hours, 3 steps | 1.25 (1.20–1.32) | 325 MiB |
| Threshold SA | 64,393 buildings, 8,760 hours, 3 steps | 29.7 (29.5–34.8) | 538 MiB |
| SEER SA | 200 buildings, 8,760 hours, 3 steps | 1.19 (1.09–1.28) | 306 MiB |
| SEER SA | 64,393 buildings, 8,760 hours, 3 steps | 9.00 (8.71–12.1) | 426 MiB |
| Package | 2,000 source rows | 0.280 (0.254–0.321) | 214 MiB |
| Package | 245,655 source rows; 245,508 valid rows | 15.0 (14.1–15.8) | 941 MiB |

The first full threshold profile spent 27.6 of 29.7 seconds in cooling-demand metrics.
NumPy percentile partitioning accounted for 5.88 seconds within that stage.
The full SEER profile called the demand stage once, as intended; its three impact steps cost about 0.3 seconds together.
Packaging spent 6.16 seconds in ZIP compression and 2.58 seconds writing CSV.
The existing thermal chunks kept measured peak memory below 1 GiB in both full-stock paths.
These profiles do not identify an urgent memory fix.

The experiment covers two driver families for one weather year. It does not reproduce every
variable, the production multi-year weather window, hourly-result retention, or the historical 20-hour job.
Measure that exact workload before choosing process counts, parallelism or a new sensitivity method.
Use the existing CodSpeed benchmarks for regression detection; shared-machine timings cannot establish a speedup.
