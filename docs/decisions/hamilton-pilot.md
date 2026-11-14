# Keep Snakemake's direct cooling-demand calls

The Hamilton pilot preserves numerical results but adds orchestration code. Keep
`scripts/run_cdm_stage.py` calling the model directly. Do not add Hamilton to the
production dependencies.

## Acceptance criteria

Adopt Hamilton only if both conditions hold:

1. Its aggregate CSV outputs match the existing smoke run with relative tolerance
   `1e-9` and absolute tolerance zero.
2. It halves the cooling-demand glue, counting adapters moved to other files, or
   its generated graph replaces a manually maintained source-input list.

## Trial

Tested on 14 November 2026, from commit `57c4eaa`, with Python 3.14.4 and
[`apache-hamilton==1.90.0`](https://pypi.org/project/apache-hamilton/1.90.0/).
The trial used the documented
[`Builder().with_modules(...).build()` and `execute()` API](https://hamilton.apache.org/concepts/driver/).
It ran without Hamilton caching or parallel execution.

The candidate had nine functions. Their argument names declared these edges:

| Node | Dependencies | Existing operation |
| --- | --- | --- |
| `parameters` | arguments | `_load_parameters` |
| `buildings` | arguments | `read_buildings` |
| `raw_weather` | arguments | `pd.read_csv` |
| `parameterized_buildings` | buildings, parameters | `add_parameters_to_buildings` |
| `cooling_technologies` | parameters | `calculate_environmental_parameters_for_cooling_technologies` |
| `equipped_buildings` | parameterized buildings, cooling technologies | `add_cooling_technology_data_to_buildings` |
| `time_series` | parameters, weather, arguments | `create_time_series` |
| `cooling_demand` | equipped buildings, time series, parameters, arguments | `calc_cooling_demand_metrics_for_df` |
| `output` | cooling demand, arguments | `_drop_array_columns` and GeoPackage write |

Each execution read a fresh stock and weather frame. This matters because the
parameter and time-series helpers mutate their inputs. The original `run_lca`
processed each candidate and direct output separately.

The stock was the committed 2,000-row BAG fixture, with independent synthetic
energy labels cycling from 1 to 12. Weather came from KNMI station 330 in 2020:
8,760 hours, matching the smoke profile's weather window and station. This was
**not** the full Schiermonnikoog smoke stock, and does not establish a city estimate.

All five scenarios (`SQ`, `2030`, `2050_L`, `2050_M`, `2050_H`) passed
`pandas.testing.assert_frame_equal(..., rtol=1e-9, atol=0)` for both the complete
cooling-demand GeoDataFrames and the downstream aggregate CSVs. A call without
required arguments failed before execution, confirming input validation. The
exact-stock smoke criterion remains unverified.

## Why the candidate was removed

The existing cooling-demand function occupies 38 physical lines and contains
17 Python AST statements, including its definition and docstring. The candidate
module alone occupies 45 lines and contains 27 statements, before its driver
construction and execution. It also retains `_load_parameters`. Moving these
adapters out of the entry point shortens that file but increases the total glue.
The candidate therefore fails the simplification criterion.

`MODEL_SRC` in the Snakefile already discovers every `cdm/*.py` source file with
a glob. There is no hand-maintained module list to replace. Hamilton's function
graph describes node dependencies, not every source file imported by nested
helpers. Its [cache documentation](https://hamilton.apache.org/concepts/caching/)
explicitly excludes nested calls from code versioning. The outer Snakemake rule
must still track those helpers and `uv.lock`.

The comparison was a numerical and maintenance experiment, not a performance
benchmark. Its single sequential timings include warm-cache effects and cannot
support a speed claim. The experimental adapters and Hamilton dependency are
absent from production. Reconsider only when a larger dataflow offers a concrete
reduction in maintained orchestration code.
