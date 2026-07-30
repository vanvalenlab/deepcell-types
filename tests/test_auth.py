"""Unit tests for the download / integrity layer
(``deepcell_types.utils._auth`` + the model/baseline registries). These guard
network-free, security-relevant code paths that were previously untested: the
hash-algorithm dispatch, the cache-hit and missing-token branches of
``fetch_data``, and the registry digest shapes. Archive extraction moved to
``tests/test_archive.py``.
"""

import hashlib
import requests

import pytest

from deepcell_types.utils import _auth
from deepcell_types.utils._auth import _hash_file, fetch_data
from deepcell_types.utils import (
    _latest,
    _model_registry,
    list_model_versions,
)


# --- _hash_file: algorithm dispatch by digest length ------------------------


def test_hash_file_dispatches_md5_and_sha256(tmp_path):
    f = tmp_path / "blob.bin"
    payload = b"deepcell-types integrity check"
    f.write_bytes(payload)

    algo, digest = _hash_file(f, "0" * 32)  # 32 hex -> md5
    assert algo == "md5"
    assert digest == hashlib.md5(payload).hexdigest()

    algo, digest = _hash_file(f, "0" * 64)  # 64 hex -> sha256
    assert algo == "sha256"
    assert digest == hashlib.sha256(payload).hexdigest()


def test_hash_file_rejects_unknown_digest_length(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unrecognized file_hash length"):
        _hash_file(f, "abc123")  # neither 32 nor 64 hex chars


# --- fetch_data: cache-hit and missing-token branches (no network) ----------


def test_fetch_data_returns_cached_file_on_hash_match(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    payload = b"cached checkpoint bytes"
    (cache_dir / "model.pt").write_bytes(payload)
    digest = hashlib.md5(payload).hexdigest()

    # Hash matches -> returns the cached path without ever needing a token.
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    out = fetch_data("models/model.pt", cache_subdir="models", file_hash=digest)
    assert out == cache_dir / "model.pt"


def test_fetch_data_requires_token_on_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    # No cached file -> falls through to the token check, which must raise
    # (never hits the network in the test).
    with pytest.raises(ValueError, match="DEEPCELL_ACCESS_TOKEN"):
        fetch_data("models/missing.pt", cache_subdir="models", file_hash="0" * 32)


class _Response:
    def __init__(self, *, status=200, json_data=None, text="", chunks=(), headers=None):
        self.status_code = status
        self._json_data = json_data
        self.text = text
        self._chunks = chunks
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks


def _mock_download(monkeypatch, post_response, get_response):
    monkeypatch.setenv("DEEPCELL_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(_auth.requests, "post", lambda *args, **kwargs: post_response)
    monkeypatch.setattr(_auth.requests, "get", lambda *args, **kwargs: get_response)


def test_fetch_data_reports_non_json_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    _mock_download(
        monkeypatch,
        _Response(status=502, json_data=ValueError("not json"), text="gateway"),
        _Response(),
    )
    with pytest.raises(requests.HTTPError, match="502"):
        fetch_data("models/model.pt")


def test_fetch_data_rejects_missing_download_url(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    _mock_download(monkeypatch, _Response(json_data={}), _Response())
    with pytest.raises(ValueError, match="missing download URL"):
        fetch_data("models/model.pt")


def test_fetch_data_preserves_cache_on_interrupted_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    cached = tmp_path / "models" / "model.pt"
    cached.parent.mkdir()
    cached.write_bytes(b"previous")

    def interrupted():
        yield b"partial"
        raise requests.ConnectionError("connection lost")

    _mock_download(
        monkeypatch,
        _Response(json_data={"url": "https://example.invalid/model"}),
        _Response(chunks=interrupted()),
    )
    with pytest.raises(requests.ConnectionError, match="connection lost"):
        fetch_data("models/model.pt", cache_subdir="models", file_hash="0" * 32)
    assert cached.read_bytes() == b"previous"
    assert list(cached.parent.glob(".model.pt.*")) == []


def test_fetch_data_rejects_oversized_stream_and_preserves_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    cached = tmp_path / "model.pt"
    cached.write_bytes(b"previous")
    _mock_download(
        monkeypatch,
        _Response(json_data={"url": "https://example.invalid/model"}),
        _Response(chunks=[b"123", b"456"]),
    )
    with pytest.raises(ValueError, match="safety limit"):
        fetch_data("model.pt", file_hash="0" * 32, max_download_bytes=5)
    assert cached.read_bytes() == b"previous"


def test_fetch_data_rejects_digest_mismatch_without_replacing_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    cached = tmp_path / "model.pt"
    cached.write_bytes(b"previous")
    _mock_download(
        monkeypatch,
        _Response(json_data={"url": "https://example.invalid/model"}),
        _Response(chunks=[b"wrong"]),
    )
    with pytest.raises(ValueError, match="Integrity check failed"):
        fetch_data("model.pt", file_hash=hashlib.sha256(b"expected").hexdigest())
    assert cached.read_bytes() == b"previous"


def test_fetch_data_downloads_and_lands_atomically(tmp_path, monkeypatch):
    # The happy path: a valid streamed download with a matching hash lands at
    # download_location/fname with the exact bytes and leaves no temp file.
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    payload = b"fresh model bytes"
    digest = hashlib.md5(payload).hexdigest()
    _mock_download(
        monkeypatch,
        _Response(json_data={"url": "https://example.invalid/model"}),
        _Response(chunks=[payload[:5], payload[5:]]),
    )
    out = fetch_data("models/model.pt", cache_subdir="models", file_hash=digest)
    assert out == tmp_path / "models" / "model.pt"
    assert out.read_bytes() == payload
    assert list(out.parent.glob(".model.pt.*")) == []


def test_fetch_data_reuses_unhashed_cache_without_network(tmp_path, monkeypatch):
    # With no file_hash and an existing cached file, fetch_data must return the
    # cached path directly — never reaching the token check or the network.
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    (cache_dir / "corpus.zip").write_bytes(b"corpus")
    out = fetch_data("data/corpus.zip", cache_subdir="data")
    assert out == cache_dir / "corpus.zip"


def test_fetch_data_rejects_oversized_content_length_before_writing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    _mock_download(
        monkeypatch,
        _Response(json_data={"url": "https://example.invalid/model"}),
        _Response(headers={"Content-Length": "999"}, chunks=[b"x"]),
    )
    with pytest.raises(ValueError, match="exceeding the"):
        fetch_data("model.pt", max_download_bytes=5)
    assert not (tmp_path / "model.pt").exists()


def test_fetch_data_rejects_non_numeric_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(_auth, "_asset_location", tmp_path)
    _mock_download(
        monkeypatch,
        _Response(json_data={"url": "https://example.invalid/model"}),
        _Response(headers={"Content-Length": "not-a-number"}, chunks=[b"x"]),
    )
    with pytest.raises(ValueError, match="invalid Content-Length"):
        fetch_data("model.pt")
    assert not (tmp_path / "model.pt").exists()


# --- model registry shape ---------------------------------------------------


def test_model_registry_entries_are_well_formed():
    assert list_model_versions()[0] == _latest
    assert _latest in _model_registry
    for version, entry in _model_registry.items():
        assert isinstance(version, str)
        filename, file_hash = entry
        assert isinstance(filename, str) and filename.endswith(".pt")
        assert len(file_hash) in (32, 64)
        int(file_hash, 16)  # valid hex digest
