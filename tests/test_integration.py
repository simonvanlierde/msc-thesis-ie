"""End-to-end regression guard for the cooling-demand pipeline.

Runs the full model (parameter assignment -> cooling technologies -> time series ->
cooling demand -> environmental impacts) under the SQ scenario, using the *real* SQ
parameter files and the *real* local weather backup, and checks the total annual
cooling energy demand against a pinned reference so a refactor that shifts the result
by more than the tolerance fails.

Two variants run the same harness on different building stocks:

* ``synthetic`` — a deterministic 2000-building stock generated in-process. Always
  available, so this is the guard that runs in CI (the real BAG geometry is a large
  gitignored Zenodo product CI never sees).
* ``real BAG sample`` — a deterministic 2000-building sample of the real Hague stock.
  Skipped when the BAG GeoPackage is not checked out locally.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from functions.readers import read_buildings, read_global_parameters, read_parameter_specific_data
from functions.sensitivity_analysis import run_CDM_model_for_SA
from functions.time_series import get_raw_weather_data

PARAMETER_DIR = Path("data/input/parameters")
PARAMETERS_TOML = PARAMETER_DIR / "parameters.toml"
SCENARIO = "SQ"
# The committed backup weather file covers these years. The test pins the weather window to it so it stays
# reproducible (and the goldens valid) even when the SQ scenario's production weather years change.
BACKUP_WEATHER_YEARS = (2018, 2022)
REAL_BAG_GPKG = Path("data/output/geodata/BAG_buildings_with_residence_data_full.gpkg")
REAL_BAG_LAYER = "BAG_buildings_full"

SAMPLE_SIZE = 2000
SAMPLE_SEED = 42
TOLERANCE = 0.05

# Golden references: total annual cooling energy demand (kWh) under SQ, captured from
# the pipeline. A refactor should not move either by more than TOLERANCE.
REFERENCE_SYNTHETIC_TOTAL_E_COOLING_KWH = 158_602_353.45
REFERENCE_REAL_BAG_TOTAL_E_COOLING_KWH = 29_663_259.43

# Skip cleanly if the real parameter / weather inputs are not checked out.
REQUIRED_INPUTS = [
    PARAMETERS_TOML,
    PARAMETER_DIR / "parameters_building_type.csv",
    PARAMETER_DIR / "parameters_energy_class.csv",
    PARAMETER_DIR / "parameters_cooling_technology.csv",
    PARAMETER_DIR / "multidirectional_solar_radiation_fractions.csv",
    PARAMETER_DIR / "presence_load_factors.csv",
    PARAMETER_DIR / "raw_weather_data_2018_2022_HvH.csv",
]


def _make_synthetic_stock(n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """Build a deterministic stock of ``n`` buildings with all raw attributes the model reads."""
    rng = np.random.default_rng(seed)
    end_use = np.where(rng.random(n) < 0.9, "residential", "office")
    height_m = rng.uniform(6.0, 40.0, n)
    floors = np.maximum(1, np.round(height_m / 3.0)).astype(int)
    floor_area_ground_m2 = rng.uniform(50.0, 500.0, n)
    floor_area_total_m2 = floor_area_ground_m2 * floors
    width_m = np.sqrt(floor_area_ground_m2 / 2.0)
    return pd.DataFrame(
        {
            "end_use": end_use,
            "height_m": height_m,
            "construction_year": rng.integers(1900, 2021, n),
            "energy_label_int": rng.integers(1, 13, n),  # labels 1-12 map to the four energy classes
            "floor_area_ground_m2": floor_area_ground_m2,
            "floor_area_total_m2": floor_area_total_m2,
            "number_of_residences": np.where(
                end_use == "residential",
                np.maximum(1, np.round(floor_area_total_m2 / 80.0)),
                0,
            ).astype(int),
            "volume_m3": floor_area_ground_m2 * height_m,
            "MBR_azimuth": rng.uniform(0.0, 180.0, n),
            "MBR_width_m": width_m,
            "MBR_length_m": 2.0 * width_m,
        },
    )


def _run_sq_total_cooling_demand(monkeypatch: pytest.MonkeyPatch, buildings: pd.DataFrame) -> float:
    """Run the full SQ pipeline on ``buildings`` (forcing the local weather backup); return total E_cooling_kWh."""

    def fake_post(*_args: object, **_kwargs: object) -> None:
        raise requests.exceptions.Timeout

    monkeypatch.setattr(requests, "post", fake_post)

    global_parameters = read_global_parameters(PARAMETERS_TOML, SCENARIO)
    global_parameters["weather_data_start_year"], global_parameters["weather_data_end_year"] = BACKUP_WEATHER_YEARS
    building_type_parameters = read_parameter_specific_data(PARAMETER_DIR / "parameters_building_type.csv", SCENARIO)
    energy_class_parameters = read_parameter_specific_data(PARAMETER_DIR / "parameters_energy_class.csv", SCENARIO)
    cooling_technology_parameters = read_parameter_specific_data(
        PARAMETER_DIR / "parameters_cooling_technology.csv",
        SCENARIO,
    )

    with pytest.warns(UserWarning, match="backup"):
        raw_weather_data = get_raw_weather_data(global_parameters)

    result, _impact_summary = run_CDM_model_for_SA(
        buildings,
        raw_weather_data,
        global_parameters,
        building_type_parameters,
        energy_class_parameters,
        cooling_technology_parameters,
        str(PARAMETER_DIR / "multidirectional_solar_radiation_fractions.csv"),
        str(PARAMETER_DIR / "presence_load_factors.csv"),
    )
    return result["E_cooling_kWh"].sum()


def _assert_within_tolerance(total_kwh: float, reference_kwh: float) -> None:
    """Fail if ``total_kwh`` drifts from ``reference_kwh`` by more than TOLERANCE."""
    relative_change = abs(total_kwh - reference_kwh) / reference_kwh
    assert relative_change < TOLERANCE, (
        f"Total SQ cooling demand {total_kwh:,.0f} kWh drifted {relative_change:.1%} "
        f"from the reference {reference_kwh:,.0f} kWh (tolerance {TOLERANCE:.0%})."
    )


@pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_INPUTS),
    reason="real SQ parameter / weather inputs are not checked out",
)
def test_full_pipeline_sq_synthetic_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    total_kwh = _run_sq_total_cooling_demand(monkeypatch, _make_synthetic_stock())
    _assert_within_tolerance(total_kwh, REFERENCE_SYNTHETIC_TOTAL_E_COOLING_KWH)


@pytest.mark.skipif(
    not (REAL_BAG_GPKG.exists() and all(path.exists() for path in REQUIRED_INPUTS)),
    reason="real BAG GeoPackage is not checked out (gitignored Zenodo product)",
)
def test_full_pipeline_sq_real_bag_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    stock = read_buildings(REAL_BAG_GPKG, REAL_BAG_LAYER)
    buildings = stock.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED).reset_index(drop=True)
    total_kwh = _run_sq_total_cooling_demand(monkeypatch, buildings)
    _assert_within_tolerance(total_kwh, REFERENCE_REAL_BAG_TOTAL_E_COOLING_KWH)
