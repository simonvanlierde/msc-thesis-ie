"""Profile real sensitivity runners or private dataset packaging in a fresh process."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import requests

from cdm.readers import read_buildings, read_global_parameters, read_parameter_specific_data
from cdm.sensitivity_analysis import (
    run_SA_for_variable_in_cooling_technology_parameters,
    run_SA_for_variable_in_global_parameters,
)
from cdm.time_series import get_raw_weather_data
from scripts.build_dataset_release import build_release
from scripts.run_sensitivity import PARAMETER_DIR, PARAMETERS_TOML, build_static_parameters


def positive_int(value: str) -> int:
    """Reject empty scientific workloads before loading inputs."""
    number = int(value)
    if number < 1:
        msg = "must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return number


def sensitivity(source: Path, layer: str, sample: int | None, steps: int, variable: str) -> dict:
    """Run the production SA path with seeded real stock and offline 2022 weather."""
    buildings = read_buildings(source, layer)
    stock_rows = len(buildings)
    if sample is not None:
        buildings = buildings.sample(n=min(sample, stock_rows), random_state=42).reset_index(drop=True)
    if buildings.empty:
        msg = "no modelled buildings remain after filtering"
        raise ValueError(msg)
    params: dict = {"global": read_global_parameters(PARAMETERS_TOML, "SQ")}
    for kind in ("building_type", "energy_class", "cooling_technology"):
        params[kind] = read_parameter_specific_data(PARAMETER_DIR / f"parameters_{kind}.csv", "SQ")
    params["global"]["weather_data_start_year"] = params["global"]["weather_data_end_year"] = 2022
    # Exercise the production backup conversion without contacting KNMI.
    with patch("requests.post", side_effect=requests.exceptions.Timeout), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        weather = get_raw_weather_data(params["global"])
    static = build_static_parameters(
        buildings,
        weather,
        params,
        str(PARAMETER_DIR / "multidirectional_solar_radiation_fractions.csv"),
        str(PARAMETER_DIR / "presence_load_factors.csv"),
    )
    # Progress threads distort the call profile and are not useful in batch output.
    with patch("cdm.sensitivity_analysis.tqdm", lambda values: values):
        if variable == "T_thresh_C":
            results = run_SA_for_variable_in_global_parameters(variable, 21, 27, params["global"], static, steps)
        else:
            results = run_SA_for_variable_in_cooling_technology_parameters(
                variable,
                0.5,
                3,
                params["cooling_technology"],
                static,
                steps,
            )
    if results.shape != (steps, 5) or results.isna().any().any():
        msg = "sensitivity output has missing rows, impacts or values"
        raise ValueError(msg)
    return {
        "stock_rows": stock_rows,
        "buildings": len(buildings),
        "weather_hours": len(weather),
        "weather_year": 2022,
        "scenario": "SQ",
        "sample_seed": 42,
        "steps": steps,
        "variable": variable,
    }


def main() -> int:
    """Write one JSON measurement and a cProfile file, without publishing scientific outputs."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Exit codes: 0 success, 1 invalid inputs, 2 usage, 3 I/O or execution failure.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Private full-stock GeoPackage.")
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="New cProfile output file; keep under ignored results/.",
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    sa = commands.add_parser("sensitivity", help="Profile one real SA runner without generating figures.")
    sa.add_argument("--layer", default="BAG_buildings_full")
    sa.add_argument("--sample", type=positive_int, help="Seeded sample size; omit for the full modelled stock.")
    sa.add_argument("--steps", type=positive_int, default=3)
    sa.add_argument("--variable", choices=("T_thresh_C", "SEER"), default="T_thresh_C")
    package = commands.add_parser("package", help="Measure the complete private open-input packaging path.")
    package.add_argument("--output", type=Path, required=True, help="New private release directory.")
    args = parser.parse_args()
    if args.mode == "package" and args.profile.resolve() == (args.output.parent / f"{args.output.name}.zip").resolve():
        parser.error("--profile must not replace the measured archive")
    try:
        if args.profile.exists():
            raise FileExistsError(args.profile)  # noqa: TRY301 - Keep overwrite refusal at the CLI boundary.
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        started = time.perf_counter()
        with profiler:
            if args.mode == "sensitivity":
                workload = sensitivity(args.source, args.layer, args.sample, args.steps, args.variable)
            else:
                build_release(args.source, args.output)
                inventory = json.loads((args.output / "inventory.json").read_text(encoding="utf-8"))
                workload = {key: inventory[key] for key in ("source_rows", "rows", "unique_bag_ids")}
        elapsed = time.perf_counter() - started
        profiler.dump_stats(args.profile)
        with args.source.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        report = {
            "mode": args.mode,
            "workload": workload,
            "wall_seconds": elapsed,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 if sys.platform != "darwin" else 1024 * 1024),
            "python": sys.version,
            "platform": platform.system(),
            "source_sha256": checksum,
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "pandas", "geopandas", "shapely")
            },
            "jit_enabled": sys._jit.is_enabled() if hasattr(sys, "_jit") else False,  # noqa: SLF001 - Public CPython diagnostic API.
            "threads": {
                name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "profiling": "cProfile; overhead included; fresh process; imports excluded from wall time",
        }
    except ValueError as error:
        print(f"error: {str(error)!r}", file=sys.stderr)  # noqa: T201
        return 1
    except Exception as error:  # noqa: BLE001 - CLI boundary maps execution failures to the documented exit code.
        print(f"error: {str(error)!r}", file=sys.stderr)  # noqa: T201
        return 3
    print(json.dumps(report))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
