"""Unit tests for the run_cdm_stage helpers."""

import numpy as np
import pandas as pd

from scripts.run_cdm_stage import _drop_array_columns


def test_drop_array_columns_drops_a_column_whose_first_cell_is_none() -> None:
    """Array columns must be found by scanning, not by iloc[0]: a None first cell hid them before."""
    buildings = pd.DataFrame(
        {
            "scalar": [1.0, 2.0],
            # First cell None, second an array: GeoPackage cannot serialize this column.
            "hourly": [None, np.arange(3)],
        },
    )

    result = _drop_array_columns(buildings)

    assert list(result.columns) == ["scalar"]


def test_drop_array_columns_keeps_plain_object_columns() -> None:
    """String/object columns that never hold arrays must survive."""
    buildings = pd.DataFrame({"id_BAG": ["a", "b"], "value": [1.0, 2.0]})

    result = _drop_array_columns(buildings)

    assert list(result.columns) == ["id_BAG", "value"]
