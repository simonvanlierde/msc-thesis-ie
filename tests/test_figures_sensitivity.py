"""Unit tests for the pure post-processing helpers of the sensitivity-analysis plots."""

import pandas as pd
import pytest

from cdm.figures_sensitivity import select_reference_row


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
