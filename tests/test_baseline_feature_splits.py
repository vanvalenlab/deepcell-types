"""Regression tests for baseline feature split validation."""

import json

import numpy as np
import pytest
import zarr

import deepcell_types.training.config as config_module
import deepcell_types.training.utils as utils


class _Config:
    marker2idx = {"CD3": 0}
    ct2idx = {"CD4T": 0}


def _write_split(path, train, val, heldout=None):
    path.write_text(
        json.dumps(
            {
                "metadata": {},
                "train": train,
                "val": val,
                "heldout": heldout or {},
            }
        )
    )


def _features(labels=(0,)):
    return {
        "features": np.ones((len(labels), 1), dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "cell_indices": np.arange(len(labels), dtype=np.int64),
    }


def _patch_archive(monkeypatch, dataset_keys, per_dataset):
    monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: object())
    monkeypatch.setattr(config_module, "_discover_fov_keys", lambda zf: dataset_keys)
    monkeypatch.setattr(
        utils,
        "_extract_all_dataset_features",
        lambda **kwargs: per_dataset,
    )


def test_extract_features_rejects_split_fov_without_features(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    _write_split(split_file, {"ds1": ["ds1"]}, {"ds2": ["ds2"]})
    _patch_archive(monkeypatch, ["ds1", "ds2"], {"ds1": _features()})

    with pytest.raises(ValueError, match="produced no features"):
        utils.extract_features_from_zarr(
            zarr_dir=str(tmp_path / "archive.zarr"),
            dct_config=_Config(),
            split_file=str(split_file),
        )


def test_extract_features_rejects_processed_fov_absent_from_split(
    tmp_path, monkeypatch
):
    split_file = tmp_path / "split.json"
    _write_split(split_file, {"ds1": ["ds1"]}, {})
    _patch_archive(
        monkeypatch,
        ["ds1", "ds2"],
        {"ds1": _features(), "ds2": _features()},
    )

    with pytest.raises(ValueError, match="absent from the split file"):
        utils.extract_features_from_zarr(
            zarr_dir=str(tmp_path / "archive.zarr"),
            dct_config=_Config(),
            split_file=str(split_file),
        )


def test_extract_features_rejects_split_fov_absent_from_archive(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    _write_split(split_file, {"missing": ["missing"]}, {})
    _patch_archive(monkeypatch, ["ds1"], {})

    with pytest.raises(ValueError, match="not present in the archive"):
        utils.extract_features_from_zarr(
            zarr_dir=str(tmp_path / "archive.zarr"),
            dct_config=_Config(),
            split_file=str(split_file),
        )


def test_extract_features_rejects_filtered_split_fov(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    _write_split(split_file, {"ds1": ["ds1"]}, {"ds2": ["ds2"]})
    _patch_archive(monkeypatch, ["ds1", "ds2"], {"ds1": _features()})

    with pytest.raises(ValueError, match="excluded by keep_datasets"):
        utils.extract_features_from_zarr(
            zarr_dir=str(tmp_path / "archive.zarr"),
            dct_config=_Config(),
            split_file=str(split_file),
            keep_datasets=["ds1"],
        )


def test_extract_features_rebuilds_unreadable_split_cache(tmp_path, monkeypatch):
    split_file = tmp_path / "split.json"
    cache_file = tmp_path / "features.npz"
    _write_split(split_file, {"ds1": ["ds1"]}, {})
    cache_file.write_bytes(b"not an npz")
    _patch_archive(monkeypatch, ["ds1"], {"ds1": _features(labels=(0, 0))})

    out = utils.extract_features_from_zarr(
        zarr_dir=str(tmp_path / "archive.zarr"),
        dct_config=_Config(),
        split_file=str(split_file),
        cache_path=str(cache_file),
    )

    assert len(out["X_train"]) == 2
    assert cache_file.exists()
