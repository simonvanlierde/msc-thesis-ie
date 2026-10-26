"""Snakemake's paper_numbers rule must reproduce the pinned paper_numbers.json.

The rule's inputs are the per-archetype scenario results. The current ones come from the
full pipeline on the real BAG stock, which CI does not have, so this test runs the rule on
the committed reference results in data/output/ instead. That checks scripts/paper_numbers.py
and its Snakemake wiring, not the model: a change to the model does not move these numbers.
The rest of `snakemake paper` (the F2 figure) needs the per-building GeoPackage, so it is left out.

To repin after an intended change to paper_numbers.py, copy the rule's new JSON over
tests/paper_numbers_pinned.json.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
REFERENCE_RESULTS_DIR = REPO_ROOT / "data/output"
PINNED = REPO_ROOT / "tests/paper_numbers_pinned.json"
SCENARIOS = ["SQ", "2030", "2050_L", "2050_M", "2050_H"]


def test_snakemake_paper_numbers_matches_pinned_json(tmp_path: Path) -> None:
    for scenario in SCENARIOS:
        shutil.copy(REFERENCE_RESULTS_DIR / f"CDM_results_{scenario}_full.csv", tmp_path)
    output = tmp_path / "paper_numbers.json"

    # --nolock: a pipeline run in the same checkout must not block (or be blocked by) the test.
    # --drop-metadata: keep the test's run out of .snakemake's rerun records.
    subprocess.run(  # noqa: S603 -- fixed argv, no user input
        [
            sys.executable,
            "-m",
            "snakemake",
            "--cores",
            "1",
            "--nolock",
            "--drop-metadata",
            "--config",
            f"results_dir={tmp_path}",
            "--",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    # Exact: same CSVs and the locked numpy/pandas give the same floats, so even a last-digit edit fails.
    assert json.loads(output.read_text()) == json.loads(PINNED.read_text())
