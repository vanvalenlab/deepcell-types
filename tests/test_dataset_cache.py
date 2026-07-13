"""Tests for zarr cell-data cache provenance and negative entries."""

import json
import os
import pickle

import pytest

from deepcell_types.training import config as config_module
from deepcell_types.training.config import (
    archive_array_fingerprint,
    archive_metadata_fingerprint,
    cached_archive_metadata_fingerprint,
)
from deepcell_types.training.dataset import FullImageDataset


def test_cell_data_cache_keeps_unlabeled_negative_entries(tmp_path):
    cache_path = tmp_path / "cell-data.pkl"
    fingerprint = "archive123"
    payload = {
        "labeled": {
            "channel_names": ["CD3"],
            "cell_data": (["CD4T"], [1], [[10.0, 20.0]]),
        },
        "unlabeled": {
            "channel_names": ["CD3"],
            "cell_data": None,
        },
    }

    FullImageDataset._save_cell_data_cache(cache_path, payload, fingerprint)
    loaded = FullImageDataset._load_cell_data_cache(
        cache_path, ["labeled", "unlabeled"], fingerprint
    )

    assert loaded == payload


def test_cell_data_cache_rebuilds_when_expected_key_missing(tmp_path):
    cache_path = tmp_path / "cell-data.pkl"
    fingerprint = "archive123"
    FullImageDataset._save_cell_data_cache(
        cache_path,
        {
            "labeled": {
                "channel_names": ["CD3"],
                "cell_data": (["CD4T"], [1], [[10.0, 20.0]]),
            }
        },
        fingerprint,
    )

    loaded = FullImageDataset._load_cell_data_cache(
        cache_path, ["labeled", "unlabeled"], fingerprint
    )

    assert loaded is None


def test_cell_data_cache_rejects_fingerprint_mismatch(tmp_path):
    cache_path = tmp_path / "cell-data.pkl"
    FullImageDataset._save_cell_data_cache(
        cache_path,
        {
            "labeled": {
                "channel_names": ["CD3"],
                "cell_data": (["CD4T"], [1], [[10.0, 20.0]]),
            }
        },
        "old-fingerprint",
    )

    loaded = FullImageDataset._load_cell_data_cache(
        cache_path, ["labeled"], "new-fingerprint"
    )

    assert loaded is None


def test_cell_data_cache_rejects_legacy_bare_dict(tmp_path):
    cache_path = tmp_path / "cell-data.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "labeled": {
                    "channel_names": ["CD3"],
                    "cell_data": (["CD4T"], [1], [[10.0, 20.0]]),
                }
            },
            f,
        )

    loaded = FullImageDataset._load_cell_data_cache(
        cache_path, ["labeled"], "archive123"
    )

    assert loaded is None


def test_cell_data_cache_rejects_group_writable_file(tmp_path):
    cache_path = tmp_path / "cell-data.pkl"
    FullImageDataset._save_cell_data_cache(cache_path, {"labeled": {}}, "archive123")
    cache_path.chmod(0o660)

    loaded = FullImageDataset._load_cell_data_cache(
        cache_path, ["labeled"], "archive123"
    )

    assert loaded is None
    assert os.stat(cache_path).st_mode & 0o020


def test_cell_data_cache_rejects_symlink(tmp_path):
    # The cache path is predictable, so an attacker on a shared filesystem could
    # replace it with a symlink to a crafted pickle they control. O_NOFOLLOW
    # must refuse to follow the link and trigger a rebuild instead.
    real = tmp_path / "attacker.pkl"
    FullImageDataset._save_cell_data_cache(real, {"labeled": {}}, "archive123")
    cache_path = tmp_path / "cell-data.pkl"
    os.symlink(real, cache_path)

    loaded = FullImageDataset._load_cell_data_cache(
        cache_path, ["labeled"], "archive123"
    )

    assert loaded is None


def test_cell_data_cache_rebuilds_on_stale_import_error(tmp_path, monkeypatch):
    cache_path = tmp_path / "cell-data.pkl"
    cache_path.write_bytes(b"placeholder")

    def fail_load(_file):
        raise ModuleNotFoundError("removed cache class")

    monkeypatch.setattr(pickle, "load", fail_load)
    assert (
        FullImageDataset._load_cell_data_cache(cache_path, ["labeled"], "archive123")
        is None
    )


def test_validate_cell_data_lengths_rejects_mismatch():
    # cell_types / cell_indices / centroids of differing lengths would silently
    # misalign labels with cells; the loader must reject it.
    with pytest.raises(ValueError, match="must have equal lengths"):
        FullImageDataset._validate_cell_data_lengths(
            "ds1", (["CD4T", "CD8T"], [1], [[0.0, 0.0], [1.0, 1.0]])
        )


def test_validate_cell_data_lengths_accepts_equal():
    triple = (["CD4T"], [1], [[0.0, 0.0]])
    assert FullImageDataset._validate_cell_data_lengths("ds1", triple) == triple


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_archive_array_fingerprint_changes_when_chunk_changes(tmp_path):
    root = tmp_path / "archive.zarr"
    preprocessed = root / "ds1" / "preprocessed"
    _write_json(root / "zarr.json", {"attributes": {}})
    _write_json(preprocessed / "zarr.json", {"attributes": {"channel_names": ["CD3"]}})
    _write_json(preprocessed / "raw" / "zarr.json", {"shape": [1, 2, 2]})
    _write_json(preprocessed / "mask" / "zarr.json", {"shape": [2, 2]})
    chunk = preprocessed / "raw" / "c" / "0" / "0" / "0"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"one")

    before = archive_array_fingerprint(root, ["ds1"])
    chunk.write_bytes(b"changed")
    after = archive_array_fingerprint(root, ["ds1"])

    assert before != after


def test_archive_fingerprints_include_cell_type_info_chunks(tmp_path):
    root = tmp_path / "archive.zarr"
    preprocessed = root / "ds1" / "preprocessed"
    _write_json(root / "zarr.json", {"attributes": {}})
    _write_json(preprocessed / "zarr.json", {"attributes": {"channel_names": ["CD3"]}})
    _write_json(preprocessed / "raw" / "zarr.json", {"shape": [1, 2, 2]})
    _write_json(preprocessed / "mask" / "zarr.json", {"shape": [2, 2]})
    _write_json(
        preprocessed / "cell_type_info" / "cell_type" / "zarr.json",
        {"shape": [1]},
    )
    _write_json(
        preprocessed / "cell_type_info" / "cell_index" / "zarr.json",
        {"shape": [1]},
    )
    chunk = preprocessed / "cell_type_info" / "cell_type" / "c" / "0"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"CD4T")

    before_meta = archive_metadata_fingerprint(root)
    before_array = archive_array_fingerprint(root, ["ds1"])
    chunk.write_bytes(b"CD8T")
    after_meta = archive_metadata_fingerprint(root)
    after_array = archive_array_fingerprint(root, ["ds1"])

    assert before_meta != after_meta
    assert before_array != after_array


def _build_minimal_archive(root):
    preprocessed = root / "ds1" / "preprocessed"
    _write_json(root / "zarr.json", {"attributes": {}})
    _write_json(preprocessed / "zarr.json", {"attributes": {"channel_names": ["CD3"]}})
    _write_json(preprocessed / "raw" / "zarr.json", {"shape": [1, 2, 2]})
    _write_json(preprocessed / "mask" / "zarr.json", {"shape": [2, 2]})
    _write_json(
        preprocessed / "cell_type_info" / "cell_type" / "zarr.json",
        {"shape": [1]},
    )
    _write_json(
        preprocessed / "cell_type_info" / "cell_index" / "zarr.json",
        {"shape": [1]},
    )
    chunk = preprocessed / "cell_type_info" / "cell_type" / "c" / "0"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(b"CD4T")
    return chunk


def test_cached_fingerprint_matches_uncached_first_call(tmp_path):
    """Per-process memo must produce the same bytes as the canonical helper.

    On-disk cell-data caches are keyed by this fingerprint; any drift between
    the cached wrapper and the canonical helper would silently invalidate
    every consumer's pickle on next load.
    """
    root = tmp_path / "archive.zarr"
    _build_minimal_archive(root)

    config_module._FINGERPRINT_CACHE.pop(str(root.resolve()), None)
    direct = archive_metadata_fingerprint(root)
    cached = cached_archive_metadata_fingerprint(root)
    assert direct == cached


def test_cached_fingerprint_memoizes_per_path(tmp_path):
    """The cached wrapper must not re-walk the archive on repeated calls.

    Mutating a chunk between calls would change the uncached fingerprint, but
    the cached wrapper should keep returning the first-computed value for the
    process lifetime.
    """
    root = tmp_path / "archive.zarr"
    chunk = _build_minimal_archive(root)

    config_module._FINGERPRINT_CACHE.pop(str(root.resolve()), None)
    first = cached_archive_metadata_fingerprint(root)
    chunk.write_bytes(b"CD8T")
    second_uncached = archive_metadata_fingerprint(root)
    second_cached = cached_archive_metadata_fingerprint(root)

    assert first == second_cached, "memo should return the cached value"
    assert first != second_uncached, "uncached helper must still detect the mutation"
