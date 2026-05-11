"""Tests for the metadata-only archive contract validator."""

import json

from scripts.validate_archive_contract import validate_archive


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_fov(root, key, channels, raw_shape, mask_shape):
    preprocessed = root / key / "preprocessed"
    _write_json(preprocessed / "zarr.json", {"attributes": {"channel_names": channels}})
    _write_json(preprocessed / "raw" / "zarr.json", {"shape": raw_shape})
    _write_json(preprocessed / "mask" / "zarr.json", {"shape": mask_shape})


def test_validator_rejects_two_channel_collapse(tmp_path):
    root = tmp_path / "archive.zarr"
    _write_json(
        root / "zarr.json", {"attributes": {"all_standardized_channels": ["CD3"]}}
    )
    _write_fov(root, "fov1", ["CD3", "DAPI"], [2, 8, 8], [8, 8])

    report = validate_archive(root)

    assert any("only 2 channels" in error for error in report.errors)


def test_validator_rejects_raw_mask_shape_mismatch(tmp_path):
    root = tmp_path / "archive.zarr"
    _write_json(
        root / "zarr.json", {"attributes": {"all_standardized_channels": ["CD3"]}}
    )
    _write_fov(root, "fov1", ["CD3", "DAPI", "CD3"], [3, 8, 9], [8, 8])

    report = validate_archive(root)

    assert any("raw spatial shape" in error for error in report.errors)


def test_validator_rejects_missing_required_marker(tmp_path):
    # Strict canonical contract: validator only resolves exact matches
    # against the canonical registry. The FOV stores HO1 directly (the
    # canonical name); CD20 is genuinely missing.
    root = tmp_path / "archive.zarr"
    _write_json(
        root / "zarr.json",
        {"attributes": {"all_standardized_channels": ["CD3", "HO1"]}},
    )
    _write_fov(root, "fov1", ["CD3", "DAPI", "HO1"], [3, 8, 8], [8, 8])

    report = validate_archive(root, required_markers={"fov1": {"HO1", "CD20"}})

    assert any("missing required repaired markers" in error for error in report.errors)
    assert not any("HO1" in error and "missing" in error for error in report.errors)


def test_validator_rejects_non_canonical_marker(tmp_path):
    # The validator no longer alias-resolves HO-1 to HO1 — under the
    # strict contract, the FOV is reported as missing the required HO1
    # because HO-1 is not the canonical name.
    root = tmp_path / "archive.zarr"
    _write_json(
        root / "zarr.json",
        {"attributes": {"all_standardized_channels": ["CD3", "HO1"]}},
    )
    _write_fov(root, "fov1", ["CD3", "DAPI", "HO-1"], [3, 8, 8], [8, 8])

    report = validate_archive(root, required_markers={"fov1": {"HO1"}})

    assert any("HO1" in error and "missing" in error for error in report.errors)


def test_validator_rejects_split_fov_absent_from_archive(tmp_path):
    root = tmp_path / "archive.zarr"
    split = tmp_path / "split.json"
    _write_json(
        root / "zarr.json", {"attributes": {"all_standardized_channels": ["CD3"]}}
    )
    _write_fov(root, "fov1", ["CD3", "DAPI", "CD3"], [3, 8, 8], [8, 8])
    _write_json(split, {"train": {"missing": ["missing"]}, "val": {}})

    report = validate_archive(root, split_paths=[split])

    assert any("split FOVs are absent" in error for error in report.errors)
