"""Unit tests for the download / integrity / archive-extraction layer
(``deepcell_types.utils._auth`` + the model/baseline registries). These guard
network-free, security-relevant code paths that were previously untested: the
hash-algorithm dispatch, the zip-slip / tar-traversal / tar-symlink rejection in
``extract_archive``, the cache-hit and missing-token branches of ``fetch_data``,
and the registry digest shapes.
"""

import hashlib
import io
import tarfile
import zipfile
import requests

import pytest

from deepcell_types.utils import _auth
from deepcell_types.utils._auth import _hash_file, extract_archive, fetch_data
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


# --- extract_archive: path-traversal / symlink rejection --------------------


def test_extract_archive_accepts_benign_zip(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/ok.txt", "hello")
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "inner" / "ok.txt").read_text() == "hello"


def test_extract_archive_rejects_zip_slip(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="escapes"):
        extract_archive(archive, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_extract_archive_accepts_benign_tar(tmp_path):
    archive = tmp_path / "ok.tar"
    data = b"hi"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("inner/ok.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "inner" / "ok.txt").read_bytes() == data


def test_extract_archive_rejects_tar_symlink_member(tmp_path):
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe tar member"):
        extract_archive(archive, dest)


def test_extract_archive_rejects_tar_traversal_member(tmp_path):
    archive = tmp_path / "evil2.tar"
    data = b"x"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="unsafe tar member"):
        extract_archive(archive, dest)


def test_extract_archive_rejects_non_archive(tmp_path):
    plain = tmp_path / "notes.txt"
    plain.write_text("not an archive")
    with pytest.raises(ValueError, match="not a recognized"):
        extract_archive(plain, tmp_path / "out")


def test_extract_archive_enforces_member_and_size_limits(tmp_path):
    archive = tmp_path / "bounded.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one", b"123")
        zf.writestr("two", b"456")

    with pytest.raises(ValueError, match="2 members"):
        extract_archive(archive, tmp_path / "members", max_members=1)
    with pytest.raises(ValueError, match="declared size"):
        extract_archive(archive, tmp_path / "member-size", max_member_bytes=2)
    with pytest.raises(ValueError, match="larger than"):
        extract_archive(archive, tmp_path / "total-size", max_total_bytes=5)


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
