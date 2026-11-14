from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from scripts import check_dependency_licenses as licenses

if TYPE_CHECKING:
    from pathlib import Path


def test_verified_mit_wheels() -> None:
    licenses.check_metadata_gaps()


def test_updated_wheel_needs_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(licenses, "distribution", lambda _: SimpleNamespace(version="0.0", files=["LICENSE"]))
    with pytest.raises(ValueError, match="Review the shipped licence"):
        licenses.check_metadata_gaps()


def test_changed_license_needs_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    license_file = tmp_path / "LICENSE"
    license_file.write_text("Changed licence terms")
    monkeypatch.setattr(
        licenses,
        "distribution",
        lambda name: SimpleNamespace(
            version=licenses.VERIFIED_MIT[name][0],
            files=["LICENSE"],
            locate_file=lambda _: license_file,
        ),
    )
    with pytest.raises(ValueError, match="Review the shipped licence"):
        licenses.check_metadata_gaps()
