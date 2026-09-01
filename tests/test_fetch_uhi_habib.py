"""Tests for the Habib et al. (2025) UHImax raster fetch."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import rasterio

from scripts.gis import fetch_uhi_habib

RASTER = Path("data/input/geodata/UHImax_habib_TheHague_5m.tif")


@pytest.mark.skipif(not RASTER.exists(), reason="raster not fetched yet (network rule)")
def test_habib_raster_is_valid() -> None:
    """The fetched UHImax raster is RD New, 5 m, plausible-valued, and covers The Hague."""
    with rasterio.open(RASTER) as src:
        assert src.crs.to_epsg() == 28992
        assert abs(src.res[0] - 5.0) < 0.01
        band = src.read(1, masked=True)
        valid = band.compressed()
        assert valid.size > 0
        # canopy-air UHImax for a Dutch city: positive, and far below the old 8.3 C surface figure
        assert float(valid.min()) >= 0.0
        assert float(np.percentile(valid, 99)) < 12.0
        # The Hague city-centre coordinate (RD New) must fall inside the raster bounds
        x, y = 81400, 455200
        assert src.bounds.left < x < src.bounds.right
        assert src.bounds.bottom < y < src.bounds.top


def test_extract_member_reextracts_when_size_mismatched(tmp_path: Path) -> None:
    """A dest left over from an interrupted extract (wrong size) is overwritten, not trusted."""
    archive = tmp_path / "archive.zip"
    content = b"hello world"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("member.tif", content)
    dest = tmp_path / "out.tif"
    dest.write_bytes(b"truncated")  # wrong size

    fetch_uhi_habib.extract_member(archive, "member.tif", dest)

    assert dest.read_bytes() == content


def test_extract_member_skips_when_size_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dest that already matches the member's size is not re-extracted."""
    archive = tmp_path / "archive.zip"
    content = b"hello world"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("member.tif", content)
    dest = tmp_path / "out.tif"
    dest.write_bytes(content)

    calls: list[str] = []
    monkeypatch.setattr(fetch_uhi_habib.shutil, "copyfileobj", lambda *_a, **_k: calls.append("copied"))

    fetch_uhi_habib.extract_member(archive, "member.tif", dest)

    assert calls == []


class _FakeResponse:
    """Minimal stand-in for requests.Response, enough for download_file's usage."""

    def __init__(
        self,
        *,
        json_data: object = None,
        headers: dict[str, str] | None = None,
        content: bytes = b"",
    ) -> None:
        self._json_data = json_data
        self.headers = headers or {}
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> object:
        return self._json_data

    def iter_content(self, chunk_size: int) -> list[bytes]:  # noqa: ARG002 -- fixed test payload, chunking irrelevant
        return [self._content]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class _FakeSession:
    """Stand-in for retrying_session(), enough for download_file's .get/.head calls."""

    def __init__(self, *, file_name: str, content: bytes, content_length: int | None = None) -> None:
        self.files_json = [{"name": file_name, "download_url": "https://example.test/download"}]
        self.content = content
        self.content_length = len(content) if content_length is None else content_length
        self.stream_get_calls = 0

    def get(self, url: str, timeout: int | None = None, stream: bool | None = None) -> _FakeResponse:  # noqa: ARG002 -- fixed test double, url/timeout unused
        if stream:
            self.stream_get_calls += 1
            return _FakeResponse(content=self.content)
        return _FakeResponse(json_data=self.files_json)

    def head(self, url: str, timeout: int | None = None, allow_redirects: bool | None = None) -> _FakeResponse:  # noqa: ARG002 -- fixed test double, url/timeout/redirects unused
        return _FakeResponse(headers={"content-length": str(self.content_length)})


def test_download_file_skips_when_cached_size_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached file whose size matches the HEAD-reported size is not re-downloaded."""
    content = b"x" * 20
    session = _FakeSession(file_name="a.zip", content=content)
    monkeypatch.setattr(fetch_uhi_habib, "retrying_session", lambda: session)
    dest = tmp_path / "a.zip"
    dest.write_bytes(content)

    fetch_uhi_habib.download_file("https://example.test/files", "a.zip", dest)

    assert session.stream_get_calls == 0
    assert dest.read_bytes() == content


def test_download_file_redownloads_when_cached_size_mismatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated prior download (wrong size) is redownloaded rather than trusted as cached."""
    content = b"x" * 20
    session = _FakeSession(file_name="a.zip", content=content)
    monkeypatch.setattr(fetch_uhi_habib, "retrying_session", lambda: session)
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"truncated")  # simulates an interrupted previous download

    fetch_uhi_habib.download_file("https://example.test/files", "a.zip", dest)

    assert session.stream_get_calls == 1
    assert dest.read_bytes() == content
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_file_raises_on_post_download_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that doesn't match the size HEAD promised raises and leaves no partial file."""
    session = _FakeSession(file_name="a.zip", content=b"short", content_length=999)
    monkeypatch.setattr(fetch_uhi_habib, "retrying_session", lambda: session)
    dest = tmp_path / "a.zip"

    with pytest.raises(RuntimeError, match="size mismatch"):
        fetch_uhi_habib.download_file("https://example.test/files", "a.zip", dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
