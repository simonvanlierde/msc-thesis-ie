from types import SimpleNamespace

import pytest

from scripts import check_dependency_licenses as licenses


def test_verified_mit_wheels() -> None:
    licenses.check_metadata_gaps()


def test_updated_wheel_needs_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(licenses, "distribution", lambda _: SimpleNamespace(version="0.0", files=["LICENSE"]))
    with pytest.raises(ValueError, match="Review the shipped licence"):
        licenses.check_metadata_gaps()
