"""Tests for the metadata-only archive contract validator."""

import json

from scripts.validate_archive_contract import (
    check_marker_index_order,
    validate_archive,
)


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


# ---------------------------------------------------------------------------
# Marker index-map guard: all_standardized_channels IS the released model's
# frozen marker->index map. Reorder/resize silently breaks the checkpoint.
# Regression guard for the 2026-06-01 incident (registry unioned to 327 +
# re-sorted, which broke loading deepcell-types_2026-05-17.pt).
# ---------------------------------------------------------------------------


def test_marker_order_check_passes_on_exact_match():
    expected = ["CD45", "PanCK", "CD3"]
    assert check_marker_index_order(["CD45", "PanCK", "CD3"], expected) == []


def test_marker_order_check_flags_reorder_same_set():
    expected = ["CD45", "PanCK", "CD3"]
    errors = check_marker_index_order(["CD45", "CD3", "PanCK"], expected)
    assert any("order diverges" in e and "index 1" in e for e in errors)


def test_marker_order_check_flags_resize():
    expected = ["CD45", "PanCK", "CD3"]
    errors = check_marker_index_order(["CD45", "PanCK", "CD3", "FoxP3"], expected)
    assert any("index-map size mismatch" in e for e in errors)
    assert any("unknown to the released model" in e for e in errors)


def test_marker_order_check_flags_missing_and_extra():
    expected = ["CD45", "PanCK", "CD3"]
    errors = check_marker_index_order(["CD45", "PanCK", "CgA"], expected)
    assert any("absent from" in e for e in errors)  # CD3 missing
    assert any("unknown to the released model" in e for e in errors)  # CgA extra


def test_validator_flags_marker_order_drift(tmp_path):
    # The archive's registry is reordered relative to the released model's
    # frozen order -> validate_archive must surface it as an error.
    root = tmp_path / "archive.zarr"
    _write_json(
        root / "zarr.json",
        {"attributes": {"all_standardized_channels": ["PanCK", "CD45", "CD3"]}},
    )
    _write_fov(root, "fov1", ["CD45", "DAPI", "CD3"], [3, 8, 8], [8, 8])

    report = validate_archive(root, expected_marker_order=["CD45", "PanCK", "CD3"])

    assert any("order diverges from the released model" in e for e in report.errors)


def test_validator_passes_when_marker_order_matches(tmp_path):
    root = tmp_path / "archive.zarr"
    _write_json(
        root / "zarr.json",
        {"attributes": {"all_standardized_channels": ["CD45", "PanCK", "CD3"]}},
    )
    _write_fov(root, "fov1", ["CD45", "DAPI", "CD3"], [3, 8, 8], [8, 8])

    report = validate_archive(root, expected_marker_order=["CD45", "PanCK", "CD3"])

    assert not any("order diverges" in e for e in report.errors)
    assert not any("index-map size mismatch" in e for e in report.errors)
