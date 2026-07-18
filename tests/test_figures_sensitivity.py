"""Unit tests for the pure post-processing helpers of the sensitivity-analysis plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: post_process_SA_results plots as a side effect and calls plt.show()

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cdm import figures_sensitivity
from cdm.figures_sensitivity import post_process_SA_results, select_reference_row


def test_select_reference_row_skips_the_poisoned_zero_start_row() -> None:
    """A zero-start sweep has an infinite index pct_change at row 1, poisoning its elasticity to 0.0."""
    index = pd.Index([0.0, 1.5, 3.0])

    assert select_reference_row(index, reference=1.0) == pytest.approx(3.0)


def test_select_reference_row_uses_row_one_when_it_is_valid() -> None:
    """A non-zero-start sweep has a finite index pct_change at row 1, so it is a legitimate candidate."""
    index = pd.Index([1.0, 2.0, 3.0])

    assert select_reference_row(index, reference=2.2) == pytest.approx(2.0)


def test_select_reference_row_still_lands_on_a_valid_row_when_reference_is_out_of_range() -> None:
    index = pd.Index([0.0, 1.0, 2.0, 4.0])

    # Rows 0 and 1 are poisoned (NaN and inf index pct_change); only 2.0 and 4.0 are valid candidates.
    assert select_reference_row(index, reference=-50.0) == pytest.approx(2.0)


def test_select_reference_row_falls_back_to_nearest_value_for_a_degenerate_single_row_sweep() -> None:
    index = pd.Index([5.0])

    assert select_reference_row(index, reference=1.0) == pytest.approx(5.0)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*:UserWarning")
def test_post_process_SA_results_reports_nonzero_elasticities_for_a_zero_start_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: a raw argmin at the read-out site silently reproduces the uhi_scale,0.0,... bug.

    Zero-start sweep [0, 1.5, 3] with reference 1.0 poisons row 1 (index pct_change = inf there). If
    someone re-inlines ``SA_results.index[np.abs(SA_results.index - ref).argmin()]`` in
    ``post_process_SA_results`` instead of calling ``select_reference_row``, this again picks row 1
    and every returned elasticity goes back to exactly 0.0.
    """
    monkeypatch.setattr(figures_sensitivity, "SA_IMAGE_DIR", str(tmp_path))
    sa_results = pd.DataFrame(
        {
            "Cooling energy demand (kWh/m2)": [10.0, 15.0, 25.0],
            "Electricity use (kWh/m2)": [5.0, 8.0, 14.0],
            "GHG emissions (kg CO2eq/m2)": [2.0, 3.5, 6.0],
        },
        index=[0.0, 1.5, 3.0],
    )

    elasticities = post_process_SA_results(
        SA_results=sa_results,
        reference_values={"SQ": 1.0},
        variable_name_print="synthetic test variable",
        variable_unit_print="x reference",
        include_scenario_lines_in_plots=False,
    )

    assert np.isfinite(elasticities).all()
    assert (elasticities.abs() > 0).all()
