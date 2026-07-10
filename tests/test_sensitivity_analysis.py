"""Unit tests for the pure post-processing helpers of the sensitivity analysis.

The model-running wrappers (``run_SA_*``) and plotting helpers require the full
pipeline and the external geospatial data, so only the deterministic DataFrame
transforms are unit-tested here.
"""

import numpy as np
import pandas as pd
import pytest

from cdm.sensitivity_analysis import calculate_elasticity_for_SA_results, normalize_SA_results


def test_normalize_SA_results_divides_by_reference_row() -> None:
    sa_results = pd.DataFrame(
        {"GHG emissions (kg CO2eq/m2)": [10.0, 20.0, 40.0]},
        index=[1.0, 2.0, 4.0],
    )
    normalized = normalize_SA_results(sa_results, ref_value_in_SA_results=2.0)

    # Values are divided by the reference row, and units are stripped from the column label.
    assert normalized["GHG emissions"].tolist() == pytest.approx([0.5, 1.0, 2.0])


def test_normalize_SA_results_snaps_to_the_nearest_index_value() -> None:
    """A reference value that isn't exactly on the (linspace) index must not raise KeyError."""
    sa_results = pd.DataFrame(
        {"GHG emissions (kg CO2eq/m2)": [10.0, 20.0, 40.0]},
        index=[1.0, 2.0, 4.0],
    )
    # 2.1 is not an index value; it snaps to 2.0 and divides by that row.
    normalized = normalize_SA_results(sa_results, ref_value_in_SA_results=2.1)

    assert normalized["GHG emissions"].tolist() == pytest.approx([0.5, 1.0, 2.0])


def test_calculate_elasticity_for_SA_results() -> None:
    sa_results = pd.DataFrame(
        {"Electricity use (kWh/m2)": [10.0, 11.0]},
        index=[1.0, 1.1],
    )
    elasticity = calculate_elasticity_for_SA_results(sa_results)

    # Elasticity = (%change in output) / (%change in input); a 10% rise in both gives unit elasticity.
    assert np.isnan(elasticity["Electricity use"].iloc[0])
    assert elasticity["Electricity use"].iloc[1] == pytest.approx(1.0)
