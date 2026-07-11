"""Unit tests for the curated sensitivity-analysis spec table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sensitivity as rs


def test_sa_specs_are_well_formed():
    specs = rs.SA_SPECS
    assert len(specs) == 15
    kinds = {s.kind for s in specs}
    assert kinds == {"global", "cooling_tech", "market_penetration"}
    # exactly one market-penetration and the expected cooling-tech vars
    assert sum(s.kind == "market_penetration" for s in specs) == 1
    ct = {s.variable_name for s in specs if s.kind == "cooling_tech"}
    assert ct == {"SEER", "refrigerant_leakage_rate_relative", "material_density_kg_kW", "average_lifetime_yr"}
    # ranges are ordered start < end
    for s in specs:
        assert s.start < s.end, s.variable_name


def test_global_reference_values_pull_from_scenarios():
    refs = rs.reference_values_for(
        rs.SASpec("T_thresh_C", "global", 15, 30, "threshold temperature", "°C"),
        global_param_dict={"SQ": {"T_thresh_C": 24.0}, "2050_H": {"T_thresh_C": 26.0}},
    )
    assert refs == {"SQ": 24.0, "2050_H": 26.0}


def test_cooling_tech_reference_values_are_the_per_scenario_mean():
    # The cooling-tech runner indexes on mean(variable) * multiplier, so the reference is the mean.
    refs = rs.reference_values_for(
        rs.SASpec("SEER", "cooling_tech", 0.5, 3, "SEER", "×"),
        global_param_dict={"SQ": {}, "2050_H": {}},
        cooling_tech_param_dict={
            "SQ": [{"SEER": 3.0}, {"SEER": 5.0}],
            "2050_H": [{"SEER": 4.0}, {"SEER": 8.0}],
        },
    )
    assert refs == {"SQ": 4.0, "2050_H": 6.0}


def test_market_penetration_reference_is_weighted_mpr_percent():
    # Weighted total market-penetration rate in percent: shares summed per building type,
    # weighted by prevalence, times 100.
    import pandas as pd

    refs = rs.reference_values_for(
        rs.SASpec("market_penetration", "market_penetration", 0, 5.86, "MPR", "%"),
        global_param_dict={"SQ": {}},
        building_type_param_dict={
            "SQ": [
                {"building_type": "res", "cooling_technology_share_A": 0.2, "cooling_technology_share_B": 0.1},
                {"building_type": "office", "cooling_technology_share_A": 0.5, "cooling_technology_share_B": 0.0},
            ],
        },
        building_type_prevalence=pd.Series({"res": 0.75, "office": 0.25}),
    )
    # res: (0.2+0.1)=0.3*0.75=0.225 ; office: 0.5*0.25=0.125 ; sum=0.35 -> 35%
    assert refs["SQ"] == 35.0
